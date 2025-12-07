# helper/utils.py
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from helper.logging import setup_logger

logger = setup_logger(__name__)

def collate_fn(batch):
    """
    DataLoader 的校對函式 (Collate Function)。
    將 List[Dict] 轉換為 Dict[Tensor]，準備輸入給模型。
    """
    
    # 將 list of dicts 轉換為 dict of lists，再轉 tensor
    input_ids = torch.tensor([example["input_ids"] for example in batch], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in batch], dtype=torch.long)
    labels = torch.tensor([example["labels"] for example in batch], dtype=torch.long)
    
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }

def evaluate(model, dataloader, tokenizer, labels_list):
    """
    讓模型生成文字，並計算 Accuracy，適用於 Seq2Seq 任務。
    """
    # 取得模型目前在哪個裝置 (CPU 或 GPU)
    device = next(model.parameters()).device

    correct = 0  # 答對題數
    total = 0    # 總題數

    # 切換至評估模式 (關閉 Dropout 等)
    model.eval()
    logger.info("[System] 開始評估 (Evaluation)...")

    with torch.no_grad():

        # 使用 tqdm 顯示進度條
        progress_bar = tqdm(dataloader, desc="Evaluating", disable=False)

        for batch_idx, batch in enumerate(progress_bar):
            
            # 確保資料在 GPU 上
            batch = {k: v.to(device) for k, v in batch.items()}

            # 模型生成
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=10
            )

            # Decode 預測結果
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Decode 真實標籤 (Gold Labels)
            gold_texts = []
            for label_seq in batch["labels"]:
                # 過濾掉 -100 (PyTorch CrossEntropyLoss 的忽略值)
                label_seq = label_seq[label_seq != -100]

                # 轉回文字並去除頭尾空白
                gold_text = tokenizer.decode(label_seq, skip_special_tokens=True).strip()
                gold_texts.append(gold_text)

            # [Debug] 印出前幾筆看預測 vs. 標籤
            if batch_idx < 5:
                logger.debug(f"[Debug] Batch {batch_idx} 樣本的預測 vs. 標籤:")
                for i, (p, g) in enumerate(zip(predictions, gold_texts)):
                    logger.debug(f"  Sample {i}: PRED='{p}' | GOLD='{g}'")

            # 比對計算 Accuracy
            for pred, gold in zip(predictions, gold_texts):
                # 轉小寫並去除空白
                if pred.strip().lower() == gold.lower():
                    correct += 1
                total += 1

    # 計算最終 Accuracy，避免分母為 0 的錯誤
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Accuracy: {accuracy:.4f}")

    return accuracy

def plot_metrics(train_losses, val_accuracies, output_dir):
    """
    繪製訓練過程中的 Loss 與 Accuracy 變化曲線，並存為圖片
    """
    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    epochs = range(1, len(train_losses) + 1)

    # 建立畫布
    plt.figure(figsize=(12, 5))

    # 處理 Tensor 與 Float 混合的情況
    train_losses = [l.item() if torch.is_tensor(l) else l for l in train_losses]

    # 子圖 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)

    # 子圖 2: Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)

    # 存檔
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"[System] Metrics plot saved to {plot_path}")

    # 關閉畫布釋放記憶體
    plt.close()

def visualize_expert_selection(selection_counts, output_dir="."):
    """
    繪製每一層 MoE 的專家被選中的頻率百分比分佈圖。
    """
    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍歷每一層 (Layer)
    for layer_idx, layer_selection_counts in enumerate(selection_counts):
        
        # 轉 numpy 方便計算
        counts = np.array(layer_selection_counts)
        total_tokens = counts.sum()

        # 計算頻率
        if total_tokens > 0:
            frequencies = counts / total_tokens
        else:
            frequencies = np.zeros_like(counts, dtype=float)

        # 建立畫布
        plt.figure(figsize=(8, 4))
        # 在柱狀圖上方標示具體百分比
        bars = plt.bar(range(len(frequencies)), frequencies, color='skyblue')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.1%}', ha='center', va='bottom', fontsize=9)
        # 設定標題與軸標籤
        plt.title(f"Layer {layer_idx} Expert Selection Frequency")
        plt.xlabel("Expert Index")
        plt.ylabel("Frequency (Ratio)")
        # 設定 y 軸範圍 0~110% 避免文字被切掉
        plt.ylim(0, 1.1)

        # 存檔
        output_path = os.path.join(output_dir, f"layer_{layer_idx}_selection.png")
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"[System] Layer {layer_idx} 專家分佈圖已儲存至: {output_path}")

        # 關閉畫布釋放記憶體
        plt.close()