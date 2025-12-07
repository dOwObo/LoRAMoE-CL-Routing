# helper/utils.py
import torch
from tqdm import tqdm
# import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
def collate_fn(batch):
    """
    DataLoader 取出的 batch 經常會是一個 list，包含每筆資料的 dict。
    這裡示範將多筆資料組合成可以餵給模型的 tensor。
    """
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_ids = torch.tensor([example["input_ids"] for example in batch], dtype=torch.long).to(target_device)
    attention_mask = torch.tensor([example["attention_mask"] for example in batch], dtype=torch.long).to(target_device)
    labels = torch.tensor([example["labels"] for example in batch], dtype=torch.long).to(target_device)
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }

def evaluate(model, dataloader, tokenizer, labels_list):
    device = next(model.parameters()).device
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="evaluation")):
            batch = {k: v.to(device) for k, v in batch.items()}

            # T5 生成
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=10
            )
            # decode predictions
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # decode gold label
            gold_texts = []
            for label_seq in batch["labels"]:
                label_seq = label_seq[label_seq != -100]  # 去掉 -100
                gold_text = tokenizer.decode(label_seq, skip_special_tokens=True).strip()
                gold_texts.append(gold_text)

            # Debug: 印出前幾筆看看預測 vs. 標籤
            if batch_idx < 2:  # 前2個batch做檢查
                for i, (p, g) in enumerate(zip(predictions, gold_texts)):
                    print(f"[DEBUG] batch#{batch_idx} sample#{i}  PRED: '{p}' | GOLD: '{g}'")

            # 比對
            for pred, gold in zip(predictions, gold_texts):
                if pred.strip().lower() == gold.lower():
                    correct += 1
                total += 1

    accuracy = correct / total if total else 0
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

def visualize_expert_selection(selection_counts, output_dir="."):
    """
    繪製各層專家使用頻率。
    :param selection_counts: dict 或 list，存放各層的專家計數
    :param output_dir: 輸出檔案路徑
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for layer_idx, layer_selection_counts in enumerate(selection_counts):
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(layer_selection_counts)), layer_selection_counts, color='blue')
        plt.title(f"Layer {layer_idx} Expert Selection Frequency")
        plt.xlabel("Expert Index")
        plt.ylabel("Count")
        output_path = os.path.join(output_dir, f"layer_{layer_idx}_selection.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Layer {layer_idx} figure saved to {output_path}")