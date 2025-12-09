# helper/utils.py
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from helper.logging import setup_logger

logger = setup_logger(__name__)

def collate_fn(batch):
    """
    DataLoader 的校對函式 (Collate Function)。
    將 List[Dict] 轉換為 Dict[Tensor]，準備輸入給模型。
    """
    input_ids = torch.tensor([example["input_ids"] for example in batch], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in batch], dtype=torch.long)
    labels = torch.tensor([example["labels"] for example in batch], dtype=torch.long)
    
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }

def evaluate(model, dataloader, tokenizer, dataset_name):
    """
    讓模型生成文字，並計算 Accuracy，適用於 Seq2Seq 任務。
    """
    # 取得模型目前在哪個裝置 (CPU 或 GPU)
    device = next(model.parameters()).device

    correct = 0  # 答對題數
    total = 0    # 總題數

    # 切換至評估模式 (關閉 Dropout 等)
    model.eval()
    logger.info(f"[System] 開始評估 {dataset_name} (Evaluation)...")

    with torch.no_grad():

        # 使用 tqdm 顯示進度條
        progress_bar = tqdm(dataloader, desc="Evaluating", disable=True)

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
    logger.info(f">>> 【{dataset_name}】Accuracy: {accuracy:.4f}")

    return accuracy

def plot_metrics(train_losses, val_accuracies, output_dir):
    """
    繪製訓練過程中的 Loss 與 Accuracy 變化曲線，並存為圖片
    """
    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    epochs = range(1, len(train_losses) + 1)
    train_losses = [l.item() if torch.is_tensor(l) else l for l in train_losses]

    # 建立畫布
    plt.figure(figsize=(12, 5))

    # 子圖 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()

    # 子圖 2: Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()

    # 存檔
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"[System] Metrics plot saved to {plot_path}")

    # 關閉畫布釋放記憶體
    plt.close()

def visualize_expert_selection(selection_counts, output_dir=".", title_suffix=""):
    """
    繪製專家被選中的長條圖(layer)與熱力圖(all)。
    """
    if not selection_counts:
        logger.warning(f"[Warning] {title_suffix} 無專家使用數據可供繪圖")
        return
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 將數據轉為 numpy array
    try:
        data = np.array(selection_counts)
    except Exception:
        data = np.array([sc.cpu().numpy() if hasattr(sc, 'cpu') else sc for sc in selection_counts])

    # 若維度不足，補齊維度
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    num_layers, num_experts = data.shape

    # 繪製熱力圖 (Heatmap)
    try:
        # 計算每一層的百分比 (Normalized by Layer)
        row_sums = data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 # 避免除以 0
        normalized_data = data / row_sums

        # 建立畫布 (根據專家數和層數動態調整)
        plt.figure(figsize=(num_experts * 1.5 + 2, num_layers * 0.5 + 2))
        # 繪製 Heatmap
        sns.heatmap(
            normalized_data, 
            annot=True,      # 顯示數值
            fmt=".1%",       # 百分比格式 (如 25.0%)
            cmap="YlGnBu",   # 0% 黃-綠-藍 100%
            vmin=0, vmax=1,  # 固定範圍 0~100%
            cbar_kws={'label': 'Selection Frequency'}
        )
        # 設定標題與軸標籤
        plt.title(f"MoE Expert Utilization Heatmap ({title_suffix})")
        plt.ylabel("Layer Index")
        plt.xlabel("Expert Index")

        # 存檔
        heatmap_path = os.path.join(output_dir, f"expert_heatmap_{title_suffix.lower()}.png")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)

        # 關閉畫布釋放記憶體
        plt.close()
    except Exception as e:
        logger.error(f"[Error] 繪製熱力圖失敗: {e}")

    # 繪製分層長條圖 (Bar Charts)
    bar_plot_dir = os.path.join(output_dir, f"layers_{title_suffix.lower()}")
    os.makedirs(bar_plot_dir, exist_ok=True)
    
    for layer_idx, counts in enumerate(data):
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
        plt.ylabel("Frequency")
        # 設定 y 軸範圍 0~110% 避免文字被切掉
        plt.ylim(0, 1.1)

        # 存檔
        output_path = os.path.join(bar_plot_dir, f"layer_{layer_idx}.png")
        plt.tight_layout()
        plt.savefig(output_path)

        # 關閉畫布釋放記憶體
        plt.close()

    logger.info(f"[System] {title_suffix} 專家分佈圖已儲存至: {output_dir}")