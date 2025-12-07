# helper/trainer.py
import torch
from torch.optim import AdamW
from transformers.optimization import Adafactor
from torch.cuda.amp import autocast, GradScaler
from transformers import get_scheduler
from tqdm import tqdm
from helper.utils import evaluate, visualize_expert_selection
from model.layers import MoEBlock

class Trainer:
    def __init__(
        self, 
        model, 
        train_dataloader, 
        eval_dataloader=None, 
        tokenizer=None, 
        labels_list=None, 
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

    def train(
        self, 
        num_epochs=3, 
        learning_rate=1e-4, 
        output_dir="./", 
        accumulation_steps=1,
        max_grad_norm=1.0
    ):
        """
        執行訓練過程，包含混合精度與梯度累積，並做梯度裁剪防止 NaN。
        """
        # optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            scale_parameter=False, 
            relative_step=False, 
            lr=learning_rate
        )
        num_training_steps = len(self.train_dataloader) * num_epochs
        scheduler = get_scheduler(
            "linear", 
            optimizer=optimizer, 
            num_warmup_steps=0, 
            num_training_steps=num_training_steps
        )
        
        scaler = GradScaler()
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            # ========== 進行訓練 ==========
            for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                try:
                    outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            labels=batch["labels"])
                    loss = outputs.loss / accumulation_steps

                    scaler.scale(loss).backward()

                    # 累積到一定步數再更新
                    if (step + 1) % accumulation_steps == 0:
                        # 先unscale，再梯度裁剪，接著再更新
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                        # 再呼叫 scheduler.step()
                        scheduler.step()

                except Exception as e:
                    print(f"[ERROR] Step {step} encountered an error: {e}")
                    print(f"Input IDs: {batch['input_ids']}")
                    continue

                # Debug: 打印 Loss
                if torch.isnan(loss):
                    print(f"[DEBUG] NaN Loss detected at step {step}. Input IDs: {batch['input_ids']}")
                    print(f"Current Learning Rate: {scheduler.get_last_lr()}")
                    optimizer.param_groups[0]["lr"] /= 2  # 動態調低學習率
                    print(f"Reducing Learning Rate to {scheduler.get_last_lr()}")

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")

            # ========== 驗證 ==========
            if self.eval_dataloader is not None:
                accuracy = self.validate()
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.save_model(output_dir, f"best_model_epoch_{epoch + 1}.bin")
            else:
                print("No eval_dataloader provided, skipping validation.")

        selection_counts = self.collect_selection_counts()
        visualize_expert_selection(selection_counts, output_dir=output_dir)
        print(f"Best Accuracy Achieved: {best_accuracy:.4f}")

    def validate(self):
        """
        在驗證集上評估模型 (evaluate)。
        """
        print("Evaluating...")
        
        if self.eval_dataloader is None:
            print("No eval_dataloader provided.")
            return 0.0

        accuracy = evaluate(self.model, self.eval_dataloader, self.tokenizer, self.labels_list)
        return accuracy

    def save_model(self, output_dir, model_name):
        """
        保存模型。
        """
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, model_name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


    def collect_selection_counts(self):
        selection_counts = []
        for name, module in self.model.named_modules():
            if isinstance(module, MoEBlock):
                counts = module.selection_counts.cpu().numpy().tolist()
                selection_counts.append(counts)
                module.selection_counts.zero_()  # 重置計數
        return selection_counts