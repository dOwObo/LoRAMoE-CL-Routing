# main.py
import os
import csv
import shutil
import argparse
import random
import numpy as np
import torch
from dataset.data_processor import DataProcessor
from model.custom_t5 import CustomT5Model
from helper.utils import collate_fn
from helper.trainer import Trainer
from helper.logging import setup_logger
# from helper.inference_efficiency import measure_inference_time, get_vram_usage

logger = setup_logger(__name__)

def print_trainable_parameters(model):
    """
    可訓練參數統計
    """
    trainable_params = 0
    all_param = 0
    # LoRA & Experts (運算權重)
    lora_expert_trainable = 0
    lora_expert_total = 0
    # Router & Gate (路由權重)
    router_gate_trainable = 0
    router_gate_total = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        # 全部可訓練
        if param.requires_grad:
            trainable_params += num_params
        # 檢查 Router/Gate
        if "router" in name or "gate" in name:
            router_gate_total += num_params
            if param.requires_grad:
                router_gate_trainable += num_params
        # 檢查 LoRA/Expert
        elif "lora_" in name or "experts" in name:
            lora_expert_total += num_params
            if param.requires_grad:
                lora_expert_trainable += num_params

    # 計算百分比
    percent_trainable = 100 * trainable_params / all_param if all_param > 0 else 0
    percent_lora = 100 * lora_expert_trainable / lora_expert_total if lora_expert_total > 0 else 0
    percent_router = 100 * router_gate_trainable / router_gate_total if router_gate_total > 0 else 0

    logger.info("\n" + "="*15 + " Parameter Check " + "="*15)
    logger.info(f"Total Params:          {all_param:,}")
    logger.info(f"Trainable Params:      {trainable_params:,} ({percent_trainable:.4f}%)")
    logger.info(f"[LoRA / Experts] Weights:")
    logger.info(f"  - Total:               {lora_expert_total:,}")
    logger.info(f"  - Trainable:           {lora_expert_trainable:,}")
    logger.info(f"  - Status:              {'UNFROZEN' if lora_expert_trainable > 0 else 'FROZEN'} ({percent_lora:.2f}%)")
    logger.info(f"[Router / Gate] Weights:")
    logger.info(f"  - Total:               {router_gate_total:,}")
    logger.info(f"  - Trainable:           {router_gate_trainable:,}")
    logger.info(f"  - Status:              {'UNFROZEN' if router_gate_trainable > 0 else 'FROZEN'} ({percent_router:.2f}%)")
    logger.info("="*47 + "\n")

def set_seed(seed):
    """
    固定隨機亂數種子，確保實驗結果的可複現性 (Reproducibility)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    """
    解析命令列參數，將超參數參數化以便實驗調整
    """
    parser = argparse.ArgumentParser(description="Train a custom T5 model with LoRA and MoE for Continual Learning.")
    # 資料路徑設定
    parser.add_argument('--data_file', type=str, required=True, help='Path to training data (JSON).')
    parser.add_argument('--labels_file', type=str, required=True, help='Path to training labels (JSON).')
    parser.add_argument('--eval_file', type=str, required=True, help='Path to eval data (JSON).')
    parser.add_argument('--eval_labels_files', type=str, required=True, help='Path to eval labels (JSON).')
    parser.add_argument('--test_data_files', type=str, nargs='*', default=[], help='List of paths to test data files.')
    parser.add_argument('--test_labels_files', type=str, nargs='*', default=[], help='List of paths to test labels files.')
    # 儲存路徑設定
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save model.')
    parser.add_argument('--plot_dir', type=str, default=None, help='Specific directory to save plots.')
    parser.add_argument('--dataset_name', type=str, default="", help='Name of the dataset for logging/plotting.')
    # 模型架構設定
    parser.add_argument('--base_model_name', type=str, default="./initial_model/t5-large", help='Base model identifier (e.g., t5-small, t5-large).')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a pretrained custom model (for CL Task 2+).')
    parser.add_argument('--adapter_type', type=str, default="MoEBlock", choices=["LoRA", "MoEBlock"], help='Type of adapter: "LoRA" or "MoEBlock".')
    parser.add_argument('--dynamic_expansion', action='store_true', help='Enable Dynamic Expansion CL (freeze old, add new params).')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts in MoE.')
    parser.add_argument('--expert_rank', type=int, default=8, help='Rank of LoRA matrices.')
    parser.add_argument('--top_k', type=int, default=2, help='Top-K routing selection.')
    # 訓練參數設定
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device.')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Gradient accumulation steps.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping threshold.')
    # Loss 權重設定
    parser.add_argument('--lambda_orth', type=float, default=0.0, help='Weight for Orthogonal Loss.')
    parser.add_argument('--lambda_balance', type=float, default=0.0, help='Weight for MoE Load Balancing Loss.')
    # 其他設定
    parser.add_argument('--max_input_length', type=int, default=256, help='Max sequence length for input.')
    parser.add_argument('--max_label_length', type=int, default=50, help='Max sequence length for labels.')
    parser.add_argument('--debug', action='store_true', help='Debug mode (use small subset of data).')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # ========== 1. 環境設定 ==========
    
    # 設定隨機種子
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # [Debug]
    if torch.cuda.is_available():
        logger.info(f"[System] Currently using GPU: {torch.cuda.get_device_name(device)}")

    # 處理輸出目錄
    if os.path.exists(args.output_dir):
        logger.info(f"[System] 刪除舊的輸出目錄: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"[System] 已建立輸出目錄: {args.output_dir}")

    # ========== 2. 資料處理 ==========

    # 如果是載入舊模型 (CL Task 2+)，tokenizer 應該沿用舊模型的設定
    tokenizer_path = args.model_path if args.model_path else "./initial_model/" + args.base_model_name
    # 防呆：如果本地沒有 initial_model，改用 huggingface ID
    if not os.path.exists(tokenizer_path) and not args.model_path:
        tokenizer_path = args.base_model_name

    logger.info(f"[System] 正在處理訓練資料: {args.data_file}")
    train_processor = DataProcessor(
        data_file=args.data_file,
        labels_file=args.labels_file,
        peft_model_path=tokenizer_path,
        max_input_length=args.max_input_length,
        max_label_length=args.max_label_length
    )
    train_dataset = train_processor.get_dataset()

    logger.info(f"[System] 正在處理驗證資料: {args.eval_file}")
    eval_processor = DataProcessor(
        data_file=args.eval_file,
        labels_file=args.eval_labels_files,
        peft_model_path=tokenizer_path,
        max_input_length=args.max_input_length,
        max_label_length=args.max_label_length
    )
    eval_dataset = eval_processor.get_dataset()

    # 根據 Debug 模式決定 DataLoader
    if args.debug:
        logger.info("[System] 使用小型資料集進行快速測試")
        train_dataloader = train_processor.get_subset_dataloader(
            train_dataset, 
            args.batch_size, 
            collate_fn, 
            subset_size=100, 
            shuffle=True
        )
        eval_dataloader = eval_processor.get_subset_dataloader(
            eval_dataset, 
            args.batch_size, 
            collate_fn, 
            subset_size=100, 
            shuffle=False
        )
    else:
        train_dataloader = train_processor.get_dataloader(
            train_dataset, 
            args.batch_size, 
            collate_fn
        )
        eval_dataloader = eval_processor.get_dataloader(
            eval_dataset, 
            args.batch_size, 
            collate_fn
        )
        
    # ========== 3. 初始化模型 ==========

    # 持續學習 (Task 2+) 或載入微調好的模型
    if args.model_path:
        logger.info(f"[System] Loading pretrained custom model from: {args.model_path}")
        custom_model = CustomT5Model.load_pretrained(
            load_directory=args.model_path, 
            device=device
        )
        
        # 若 MoEBlock，重置專家選擇統計數據
        if args.adapter_type == "MoEBlock":
            custom_model.reset_moe_usage()
        # 若 dynamic_expansion，且載入舊模型，代表要為新任務擴充結構
        if args.dynamic_expansion:
            logger.info("[System] Detecting new task. Expanding model structure...")
            custom_model.expand_model_structure()
    # 新的訓練 (Task 1)
    else:
        logger.info(f"[System] Initializing new {args.adapter_type} model based on {args.base_model_name}...")
        custom_model = CustomT5Model(
            base_model_path=args.base_model_name,
            device=device,
            adapter_type=args.adapter_type,
            dynamic_expansion=args.dynamic_expansion,
            num_experts=args.num_experts,
            expert_rank=args.expert_rank,
            top_k=args.top_k
        )

    model = custom_model.model

    # ========== 4. 參數凍結與解凍 ==========

    logger.info("[System] 正在設定參數可訓練狀態 (Freeze/Unfreeze)...")
    
    for name, param in model.named_parameters():

        # Router/Gate 參數在 MoEBlock + Fixed-Architecture CL 解凍
        if args.adapter_type == "MoEBlock" and not args.dynamic_expansion and ("router" in name or "gate" in name):
            param.requires_grad = True
            continue

        # LoRA 參數
        if "lora_" in name:
            # 在 LoRA/MoEBlock + Fixed-Architecture CL 解凍
            if not args.dynamic_expansion:
                param.requires_grad = True
            # 在 LoRA + Dynamic-Expansion CL，且 Task 1 解凍
            elif args.dynamic_expansion and not args.model_path:
                param.requires_grad = True
            # 在 LoRA + Dynamic-Expansion CL，且 Task 2+
            else:
                # 已透過 expand_model_structure() 凍結舊參數，解凍新參數
                pass
        # 凍結基礎模型參數
        else:
            param.requires_grad = False
        
    logger.info("[Model] Parameter freezing completed.")

    # ========== 5. 初始化 Trainer ==========

    trainer = Trainer(
        model=model,
        model_wrapper=custom_model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=train_processor.tokenizer,
        device=device
    )

    # allocated_before, total, usage_before = get_vram_usage()

    # ========== 6. Training ==========

    # [Debug]
    print_trainable_parameters(model)

    logger.info("[System] Start Training...")
    trainer.train(
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        accumulation_steps=args.accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        lambda_orth=args.lambda_orth,
        lambda_balance=args.lambda_balance,
        plot_dir=args.plot_dir,
        dataset_name=args.dataset_name
    )

    # allocated_after, _, usage_after = get_vram_usage()

    # print(f"模型本身 GPU VRAM 使用量: {allocated_before} MiB / {total} MiB ({usage_before:.2%})")
    # print(f"模型訓練 GPU VRAM 使用量: {allocated_after} MiB / {total} MiB ({usage_after:.2%})")

    # ========== 7. 保存模型 ==========
    
    logger.info(f"[System] Saving model to {args.output_dir}...")
    custom_model.save_pretrained(args.output_dir)

    if train_processor.tokenizer:
        train_processor.tokenizer.save_pretrained(args.output_dir)

    # ========== 8. Testing ==========

    # 儲存測試結果
    test_results = os.path.join(os.path.dirname(args.output_dir), "cl_results.csv")
    if not os.path.exists(test_results):
        with open(test_results, 'w', newline='') as f:
            csv.writer(f).writerow(['current_task', 'test_on_dataset', 'accuracy'])

    if args.test_data_files:
        logger.info("[System] Starting Testing Phase...")
        
        if len(args.test_data_files) != len(args.test_labels_files):
            msg = "[Error] 測試數據文件和標籤文件的數量不匹配"
            logger.error(msg)
            raise ValueError(msg)

        for test_data, test_label in zip(args.test_data_files, args.test_labels_files):
            test_dataset_name = os.path.basename(os.path.dirname(test_data))
            logger.info(f"[System] 正在處理測試資料: {test_data}")
            test_processor = DataProcessor(
                data_file=test_data,
                labels_file=test_label,
                peft_model_path=tokenizer_path,
                max_input_length=args.max_input_length,
                max_label_length=args.max_label_length
            )
            test_dataset = test_processor.get_dataset()
            
            if args.debug:
                logger.info("[System] 使用小型資料集進行快速測試")
                test_dataloader = test_processor.get_subset_dataloader(
                    test_dataset, 
                    args.batch_size, 
                    collate_fn, 
                    subset_size=100, 
                    shuffle=False
                )
            else:
                test_dataloader = test_processor.get_dataloader(
                    test_dataset, 
                    args.batch_size, 
                    collate_fn
                )
            
            # 替換 Trainer 的 eval_loader 進行測試
            trainer.eval_dataloader = test_dataloader
            test_acc = trainer.validate(test_dataset_name)
            logger.info(f"Test Accuracy on {test_dataset_name}: {test_acc:.4f}")

            # 將測試結果寫入 CSV
            with open(test_results, 'a', newline='') as f:
                csv.writer(f).writerow([args.dataset_name, test_dataset_name, float(test_acc)])
    else:
        logger.warning("[Warning] No test files provided. Skipping testing.")

    # 測試推理時間
    # measure_inference_time(model, test_dataloader, device)

if __name__ == "__main__":
    main()