# helper/trainer.py
import os
import csv
import torch
from tqdm import tqdm
from transformers.optimization import Adafactor, get_scheduler
from model.layers import LoRALayer, MoEBlock
from helper.utils import evaluate, plot_metrics, visualize_expert_selection
from helper.logging import setup_logger

logger = setup_logger(__name__)

class Trainer:
    def __init__(
        self, 
        model: torch.nn.Module,
        model_wrapper,
        train_dataloader: torch.utils.data.DataLoader, 
        eval_dataloader: torch.utils.data.DataLoader = None, 
        tokenizer = None, 
        device: torch.device = None
    ):
        """
        初始化 Trainer。
        """
        self.model = model
        self.model_wrapper = model_wrapper
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 儲存訓練損失與驗證準確率
        self.train_losses = []
        self.val_accuracies = []

    def train(
        self, 
        num_epochs: int, 
        learning_rate: float, 
        output_dir: str, 
        accumulation_steps: int,
        max_grad_norm: float,
        lambda_orth: float,
        lambda_balance: float,
        plot_dir: str,
        dataset_name: str # 暫時沒用到
    ):
        """
        執行完整的訓練迴圈
        """
        # 優化器設定，使用 Adafactor (T5 官方推薦)
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            scale_parameter=False, 
            relative_step=False, 
            lr=learning_rate
        )

        # Linear Decay
        num_training_steps = len(self.train_dataloader) * num_epochs
        scheduler = get_scheduler(
            "linear", 
            optimizer=optimizer, 
            num_warmup_steps=0, 
            num_training_steps=num_training_steps
        )

        """
        [使用 Scaler]（適用於 V100/T4 等舊卡）
        目的: 混合精度設定，GradScaler 用於自動縮放梯度，避免 float16 Underflow
        Code: scaler = GradScaler()
        """

        # 準備 CSV 記錄檔
        os.makedirs(output_dir, exist_ok=True)
        seed_dir = os.path.dirname(output_dir)
        metrics_csv = os.path.join(seed_dir, 'metrics.csv')
        if not os.path.exists(metrics_csv):
            with open(metrics_csv, 'w', newline='') as f:
                csv.writer(f).writerow(['dataset', 'epoch', 'train_loss', 'val_accuracy'])
        
        # ========== Start Training ==========

        best_accuracy = 0.0
        for epoch in range(num_epochs):
            # 切換至訓練模式 (啟用 Dropout 等)
            self.model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            # 使用 tqdm 顯示進度條
            progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", disable=True)
            
            for step, batch in enumerate(progress_bar):
                # 將資料移至 GPU
                batch = {k: v.to(self.device) for k, v in batch.items()}

                try:
                    # 使用 bfloat16，適用於 RTX 30/40 系列
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        # 前向傳播 (Forward Pass)
                        outputs = self.model(
                            input_ids=batch["input_ids"], 
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"]
                        )

                        # Loss 計算（Cross Entropy）
                        loss = outputs.loss

                        # LoRA L2 Orthogonal Loss
                        if lambda_orth > 0:
                            loss += sum(
                                module.compute_orth_loss(lambda_orth) 
                                for module in self.model.modules() if isinstance(module, LoRALayer)
                            )

                        # MoE Load Balancing Loss
                        if lambda_balance > 0:
                            loss += sum(
                                module.compute_balance_loss(lambda_balance) 
                                for module in self.model.modules() if isinstance(module, MoEBlock)
                            )

                        # 梯度累積
                        loss = loss / accumulation_steps

                    # 反向傳播 (Backward Pass)
                    loss.backward()

                    """
                    [使用 FP16 + Scaler]（適用於 V100/T4 等舊卡）
                    Code:
                    with torch.autocast(dtype=torch.float16):
                        outputs = self.model(...)
                        loss = outputs.loss / accumulation_steps
                    scaler.scale(loss).backward()
                    """

                    # 參數更新，梯度累積處理
                    if (step + 1) % accumulation_steps == 0:

                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                        """
                        [使用 Scaler]（適用於 V100/T4 等舊卡）
                        Code:
                        # 先反縮放才能裁減梯度
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                        """

                except Exception as e:
                    logger.warning(f"[Warning] Step {step} Failed. Exception: {e}")
                    # logger.debug(f"[Debug] Input IDs: {batch['input_ids']}")
                    optimizer.zero_grad()
                    continue

                total_loss += loss.item() * accumulation_steps

            # 計算平均 Loss
            avg_loss = total_loss / len(self.train_dataloader)
            self.train_losses.append(avg_loss)
            logger.info(f">>> 【{dataset_name}】Epoch {epoch + 1} - Loss: {avg_loss:.4f}")

            # 驗證 (Validation)
            val_accuracy = 0.0
            if self.eval_dataloader is not None:
                val_accuracy = self.validate(dataset_name)

                # 若當前 Accuracy 優於歷史最佳，則保存模型
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    save_path = os.path.join(output_dir, f"best_model_epoch_{epoch + 1}")
                    
                    if self.model_wrapper:
                        logger.info(f"[System] Saving best model via Wrapper to {save_path}")
                        self.model_wrapper.save_pretrained(save_path)
                    else:
                        logger.info(f"[System] Saving best model via Raw PyTorch to {save_path}")
                        self.model.save_pretrained(save_path)
            else:
                logger.info(f"[System] 未提供驗證資料集 (eval_dataloader)，本輪 Epoch {epoch + 1} 跳過驗證步驟")
            
            self.val_accuracies.append(val_accuracy)
            
            # 寫入 CSV
            with open(metrics_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([dataset_name, epoch + 1, float(avg_loss), float(val_accuracy)])

        # ========== End Training ==========

        logger.info(f"Best Accuracy Achieved: {best_accuracy:.4f}")

        # 繪圖
        if not plot_dir:
            plot_dir = output_dir

        plot_metrics(self.train_losses, self.val_accuracies, plot_dir)

        # 獲取 MoE 統計數據並視覺化
        moe_usage = self.model_wrapper.get_moe_usage()
        visualize_expert_selection(moe_usage['encoder'], plot_dir, title_suffix="Encoder")
        visualize_expert_selection(moe_usage['decoder'], plot_dir, title_suffix="Decoder")
    
    def validate(self, dataset_name):
        """
        在驗證集上評估模型表現
        """
        if self.eval_dataloader is None:
            logger.info(f"[System] No eval_dataloader provided.")
            return 0.0

        accuracy = evaluate(self.model, self.eval_dataloader, self.tokenizer, dataset_name)
        
        return accuracy