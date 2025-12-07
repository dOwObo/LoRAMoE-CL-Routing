# main.py
import torch
from torch.nn import CrossEntropyLoss

from model.custom_t5 import CustomT5Model
from dataset.data_processor import DataProcessor
# 使用 helper/utils.py 的 collate_fn
from helper.utils import collate_fn
from helper.trainer import Trainer
import argparse
import logging
import os
import json
import shutil
from helper.inference_efficiency import measure_inference_time, get_vram_usage
from transformers import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom T5 model with LoRA and MoE for Continual Learning.")
    parser.add_argument('--data_file', type=str, required=True, help='Path to the training data file (JSON).')
    parser.add_argument('--labels_file', type=str, required=True, help='Path to the labels file (JSON).')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pre-trained model to load (optional).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the fine-tuned model.')
    parser.add_argument('--eval_file', type=str, required=True, help='Path to the evaluation data file (JSON).')
    parser.add_argument('--eval_labels_files', type=str, required=True, help='Path to the labels file (JSON).')
    parser.add_argument('--test_data_files', type=str, nargs='*', default=[], help='List of test data files (JSON).')
    parser.add_argument('--test_labels_files', type=str, nargs='*', default=[], help='List of test labels files (JSON).')
    parser.add_argument('--seed', type=int, help='seed')
    return parser.parse_args()

def save_custom_model(custom_model, output_dir):
    """
    自訂保存流程：保存 state_dict 和自定義配置。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型的 state_dict
    state_dict_path = os.path.join(output_dir, "model_state_dict.pt")
    torch.save(custom_model.model.state_dict(), state_dict_path)
    logger.info(f"Model state_dict saved to {state_dict_path}")
    
    # 保存自定義配置
    config = {
        "num_experts": custom_model.num_experts,
        "expert_rank": custom_model.expert_rank
    }
    config_path = os.path.join(output_dir, "custom_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    logger.info(f"Custom config saved to {config_path}")

def load_custom_model(load_directory, device):
    """
    自訂加載流程：加載 state_dict 和自定義配置，並初始化模型。
    """
    custom_model = CustomT5Model.load_pretrained(load_directory, device=device)
    logger.info(f"Model loaded from {load_directory}")
    return custom_model

if __name__ == "__main__":
    args = parse_args()
    
    # 設定隨機種子
    set_seed(args.seed)

    data_file = args.data_file
    labels_file = args.labels_file
    model_path = args.model_path
    output_dir = args.output_dir
    eval_file = args.eval_file
    eval_labels_files = args.eval_labels_files
    test_data_files = args.test_data_files
    test_labels_files = args.test_labels_files
    
    # 直接刪除舊的 `output_dir` 並重新建立
    if os.path.exists(output_dir):
        print(f"🗑️ 刪除舊的輸出目錄: {output_dir}")
        shutil.rmtree(output_dir)  # **刪除整個目錄**
    os.makedirs(output_dir, exist_ok=True)  # **重新建立新的空目錄**
    print(f"✅ 已重新建立輸出目錄: {output_dir}")

    base_model_path = "./initial_model/t5-large"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # batch_size = 4
    # max_input_length = 256
    # max_label_length = 16

    # OLoRA 的配置 但max_input_length=512會OOM
    batch_size = 4
    max_input_length = 256
    max_label_length = 50

    # 讀取訓練資料
    train_processor = DataProcessor(
        data_file=data_file,
        labels_file=labels_file,
        peft_model_path=base_model_path,
        max_input_length=max_input_length,
        max_label_length=max_label_length
    )
    train_dataset = train_processor.get_dataset()
    train_dataloader = train_processor.get_dataloader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn  # utils.py 的 collate_fn
    )
    #DEL 創建訓練集的 500 筆子集
    train_subset_dataloader = train_processor.get_subset_dataloader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        subset_size=500,
        shuffle=True
    )

    # 驗證資料
    eval_processor = DataProcessor(
        data_file=eval_file,
        labels_file=eval_labels_files,
        peft_model_path=base_model_path,
        max_input_length=max_input_length,
        max_label_length=max_label_length
    )
    eval_dataset = eval_processor.get_dataset()
    eval_dataloader = eval_processor.get_dataloader(
        eval_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
    )
    #DEL 創建驗證集的 500 筆子集
    eval_subset_dataloader = eval_processor.get_subset_dataloader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        subset_size=500,
        shuffle=True
    )

    # 建立 T5 + LoRA 模型
    # 如果有指定模型路徑，則加載模型，否則初始化新模型
    if model_path:
        logger.info(f"Loading model from {model_path}...")
        if not os.path.exists(model_path):
            logger.error(f"指定的模型路徑不存在: {model_path}")
            raise FileNotFoundError(f"指定的模型路徑不存在: {model_path}")
        # custom_model = CustomT5Model.load_pretrained(model_path, device=device)
        custom_model = load_custom_model(model_path, device=device)
        logger.info("Model loaded successfully.")
    else:
        logger.info("Initializing new model...")
        custom_model = CustomT5Model(base_model_path, device=device, num_experts=4, expert_rank=8) # 設置專家數量和每個專家的秩
        logger.info("Model initialized successfully.")

    model = custom_model.model  # 這邊使用的是 T5ForConditionalGeneration

    # 凍結除了 LoRA/MoE 以外的參數
    # 凍結所有模型參數 then 解凍LoRA相關參數和專家層的參數
    logger.info("凍結除了 LoRA/MoE 以外的參數...")
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        if "experts" in name:
            param.requires_grad = True
            # print(name)
    logger.info("Parameter freezing completed.")
    
    # 檢索模型結構
    # print(model)
    # for name, param in model.named_parameters():
    #     if "router" in name or "experts" in name or "lora_A" in name or "lora_B" in name:
    #         print(name)
    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        # DEL
        # train_dataloader=train_subset_dataloader,
        # eval_dataloader=eval_subset_dataloader,
        tokenizer=train_processor.tokenizer,
        labels_list=train_processor.labels_list,
        device=device
    )
    
    allocated_before, total, usage_before = get_vram_usage()

    # 開始訓練
    # trainer.train(
    #     num_epochs=1,
    #     learning_rate=5e-6,       
    #     output_dir=output_dir,
    #     accumulation_steps=2
    # )

    # # OLoRA 的配置 
    trainer.train(
        num_epochs=3,
        learning_rate=5e-4,       
        output_dir=output_dir,
        accumulation_steps=64
    )

    # allocated_after, _, usage_after = get_vram_usage()

    # print(f"模型本身 GPU VRAM 使用量: {allocated_before} MiB / {total} MiB ({usage_before:.2%})")
    # print(f"模型訓練 GPU VRAM 使用量: {allocated_after} MiB / {total} MiB ({usage_after:.2%})")

    # 保存模型
    logger.info(f"Saving model to {output_dir}...")
    # custom_model.save_pretrained(output_dir)
    save_custom_model(custom_model, output_dir)
    logger.info("Model saved successfully.")



    # 測試資料
    if test_data_files and test_labels_files:
        if len(test_data_files) != len(test_labels_files):
            logger.error("測試數據文件和標籤文件的數量不匹配。")
            raise ValueError("測試數據文件和標籤文件的數量不匹配。")
        
        logger.info("Starting testing on provided datasets...")
        for test_data, test_labels in zip(test_data_files, test_labels_files):
            logger.info(f"Testing on dataset: {test_data}")
            test_processor = DataProcessor(
                data_file=test_data,
                labels_file=test_labels,
                peft_model_path=base_model_path, 
                max_input_length=max_input_length,
                max_label_length=max_label_length
            )
            test_dataset = test_processor.get_dataset()
            test_dataloader = test_processor.get_dataloader(
                test_dataset, 
                batch_size=batch_size, 
                collate_fn=collate_fn,
            )
            # DEL 創建測試集的 500 筆子集
            test_subset_dataloader = test_processor.get_subset_dataloader(
                test_dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                subset_size=500,
                shuffle=True
            )
            trainer.eval_dataloader = test_dataloader
            # DEL
            # trainer.eval_dataloader = test_subset_dataloader
            test_accuracy = trainer.validate()
            logger.info(f"Test Accuracy on {test_data}: {test_accuracy:.4f}")
    else:
        logger.info("No test datasets provided. Skipping testing.")

    # 測試推理時間
    # measure_inference_time(model, test_dataloader, device)