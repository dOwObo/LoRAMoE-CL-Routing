# helper/trainer.py
import os
import csv
import torch
from tqdm import tqdm
from transformers.optimization import Adafactor
from transformers import get_scheduler

# 使用 Dynamic Expansion CL
# from model.layers import MoEBlock, LoRALayer
from helper.utils import evaluate, plot_metrics, visualize_expert_selection
from helper.logging import setup_logger

logger = setup_logger(__name__)

def load_balance_loss(scores):
    """
    計算負載均衡損失
    """
    # [batch_size, seq_len, num_experts] -> [seq_len, num_experts] -> [num_experts]
    expert_mean_usage = scores.mean(dim=0).mean(dim=0) 
    
    num_experts = scores.size(-1)
    balance_target = 1.0 / num_experts

    # 使用 MSE 計算當前分佈與均勻分佈的差異
    return ((expert_mean_usage - balance_target)**2).mean()

class Trainer:
    def __init__(
        self, 
        model, 
        train_dataloader, 
        eval_dataloader=None, 
        tokenizer=None, 
        labels_list=None, # [注意] 在 evaluate 中未使用
        device=None
    ):
        """
        初始化 Trainer。
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        self.labels_list = labels_list
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 儲存訓練損失與驗證準確率
        self.train_losses = []
        self.val_accuracies = []

    def train(
        self, 
        num_epochs=3, 
        learning_rate=1e-4, 
        output_dir="./", 
        accumulation_steps=1,
        max_grad_norm=1.0,
        lambda_orth=0.1 # [注意] 未使用
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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        metrics_csv_path = os.path.join(output_dir, 'metrics.csv')
        
        if not os.path.exists(metrics_csv_path):
            with open(metrics_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_accuracy'])
        
        # ========== Start Training ==========

        best_accuracy = 0.0
        for epoch in range(num_epochs):
            # 切換至訓練模式 (啟用 Dropout 等)
            self.model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            # 使用 tqdm 顯示進度條
            progress_bar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", disable=False)
            
            for step, batch in enumerate(progress_bar):
                # 將資料移至 GPU
                batch = {k: v.to(self.device) for k, v in batch.items()}

                try:
                    # 使用 bfloat16，適用於 RTX 30/40 系列
                    with torch.autocast(dtype=torch.bfloat16):
                        # 前向傳播 (Forward Pass)
                        outputs = self.model(
                            input_ids=batch["input_ids"], 
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"]
                        )

                        # Loss 計算
                        loss = outputs.loss

                        """
                        [Cross Entropy]（Baseline）
                        Code: loss = outputs.loss

                        [LoRA L2 Orthogonal Loss]（正交正則化）
                        目的: 強制不同專家的 LoRA 矩陣參數保持正交，避免專家學到重複特徵。
                        注意: 需在 Trainer.train 傳入 lambda_orth 係數 (例如 0.1)。
                        Code:
                        lora_orth_loss = sum(
                            module.compute_orth_loss(lambda_orth) 
                            for module in self.model.modules() if isinstance(module, LoRALayer)
                        )

                        [MoE Load Balancing Loss]（負載均衡損失）
                        目的: 避免 Router 傾向只選某幾個專家 (Collapse)，強制平均分配 Token。
                        注意: 需在檔案開頭定義 load_balance_loss()。
                        Code:
                        moe_lb_loss = sum(
                            load_balance_loss(module.last_scores) 
                            for module in self.model.modules() if isinstance(module, MoEBlock)
                        )
                        """

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

                total_loss += loss.item()

            # 計算平均 Loss
            avg_loss = total_loss / len(self.train_dataloader)
            self.train_losses.append(avg_loss)
            logger.info(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")

            # 驗證 (Validation)
            val_accuracy = 0.0
            if self.eval_dataloader is not None:
                val_accuracy = self.validate()

                # 若當前 Accuracy 優於歷史最佳，則保存模型
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    save_path = os.path.join(output_dir, f"best_model_epoch_{epoch + 1}")
                    self.model.save_pretrained(save_path)
            else:
                logger.info(f"[System] 未提供驗證資料集 (eval_dataloader)，本輪 Epoch {epoch + 1} 跳過驗證步驟")
            
            self.val_accuracies.append(val_accuracy)
            
            # 寫入 CSV
            with open(metrics_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, float(avg_loss), float(val_accuracy)])

        # ========== End Training ==========

        logger.info(f"Best Accuracy Achieved: {best_accuracy:.4f}")

        # 繪圖
        plot_metrics(self.train_losses, self.val_accuracies, output_dir)

        # 獲取 MoE 統計數據並視覺化
        moe_usage = self.model.get_moe_usage()
        all_counts = moe_usage['encoder'] + moe_usage['decoder']
        visualize_expert_selection(all_counts, output_dir=output_dir)

    def add_new_task(self, new_dataloader):
        """
        當新任務到來時的處理邏輯
        """
        logger.info("[System] Adding new task...")

        """
        [Dynamic Expansion CL]
        注意: 需在 LoRALayer/MoEBlock 實作 add_new_task()。
        Code:
        logger.info("[Model] Exploring Dynamic Expansion Strategy...")

        # 為所有 LoRALayer 增加新參數
        for module in self.model.modules():
            if isinstance(module, LoRALayer):
                module.add_new_task()

        # 為所有 MoEBlock 的專家層也新增 LoRA
        for module in self.model.modules():
            if isinstance(module, MoEBlock):
                module.add_new_task()

        [fixed architecture CL]（目前使用）
        注意: 這裡可以選擇是否要重置 optimizer (讓學習率回升) 或繼續沿用
        """

        # 更新 Trainer 的訓練資料為新任務的資料
        self.train_dataloader = new_dataloader
        logger.info("[System] New task added successfully.")
    
    def validate(self):
        """
        在驗證集上評估模型表現
        """
        if self.eval_dataloader is None:
            logger.info(f"[System] No eval_dataloader provided.")
            return 0.0

        accuracy = evaluate(self.model, self.eval_dataloader, self.tokenizer, self.labels_list)
        
        return accuracy