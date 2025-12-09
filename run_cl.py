import os
import sys
import subprocess
import logging
import shutil

# ================= 實驗配置 (Configuration) =================

# 1. 實驗種子
SEEDS = [438, 689, 251, 744, 329]

# 2. 基礎模型路徑
BASE_MODEL = "./initial_model/t5-large"

# 3. 資料夾設定
DATA_ROOT = "./CL_Benchmark"
RESULTS_ROOT = "O-LoRA"

# 4. 資料集與任務類型的映射
DATASET_TASK_MAP = {
    "dbpedia": "TC",
    "amazon":  "SC",
    "yahoo":   "TC",
    "agnews":  "TC"
}

# 5. 任務順序 (Order 1)
TASK_ORDER = ["dbpedia", "amazon", "yahoo", "agnews"]

# 6. 訓練參數
COMMON_ARGS = {
    "--adapter_type": "LoRA",
    "--dynamic_expansion": "",     # 空字串代表 True
    "--num_experts": "4",          # 預設值 4
    "--expert_rank": "8",          # 預設值 8
    "--top_k": "2",                # 預設值 2
    "--num_epochs": "1",           # O-LoRA 預設值 1，OrthMoE 預設值 3
    "--lr": "1e-3",                # O-LoRA 預設值 1e-3，OrthMoE 預設值 5e-4
    "--batch_size": "8",
    "--accumulation_steps": "8",
    "--lambda_orth": "0.5",
    "--lambda_balance": "0.0",
    "--max_input_length": "256",   # O-LoRA 預設值 512 (OOM)，OrthMoE 預設值 256
    "--max_label_length": "50",
    # "--debug": ""                  # 空字串代表 True
}

# ==========================================================

def setup_cl_logger(results_root):
    """
    設定 run_cl.py 專用的 Logger，路徑: results/cl.log
    """
    log_file = os.path.join(results_root, "cl.log")
    logger = logging.getLogger("CL_Runner")
    logger.setLevel(logging.INFO)
    
    # 清空之前的 Handler
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 1. File Handler (寫入檔案)
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8') # mode='w' 每次重跑該 seed 覆蓋舊 log
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 2. Stream Handler (輸出到終端機)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def run_subprocess(command, cl_logger, seed_log_path):
    """
    使用 subprocess 執行 main.py，並將詳細輸出: results/{seed}/run.log
    """
    # 將 list 轉為字串方便閱讀 log
    cmd_str = " ".join(command)
    cl_logger.info(f"執行指令: {cmd_str}")

    with open(seed_log_path, 'a', encoding='utf-8') as seed_logger:

        seed_logger.write(f"\n{'='*20} Executing Command {'='*20}\n")
        seed_logger.write(f"{cmd_str}\n")
        seed_logger.write(f"{'='*59}\n\n")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # 將錯誤輸出合併到標準輸出
            text=True,
            bufsize=1
        )

        # 即時讀取輸出
        for line in process.stdout:
            sys.stdout.write(line)
            seed_logger.write(line)
            seed_logger.flush()
        
        process.wait()
    
    if process.returncode != 0:
        cl_logger.error(f"❌ 任務執行失敗 (Return Code: {process.returncode})")
        raise RuntimeError("Subprocess failed")
    else:
        cl_logger.info("✅ 任務執行成功")

def main():
    # 確保根輸出目錄存在
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # 設定 order logger (results/cl.log)
    cl_logger = setup_cl_logger(RESULTS_ROOT)

    cl_logger.info("="*52)
    cl_logger.info(f"Seeds: {SEEDS}")
    cl_logger.info(f"Order: {TASK_ORDER}")

    for seed in SEEDS:
        cl_logger.info(f"============ Start Processing Seed: {seed} ============\n")

        # 定義該 Seed 的結果目錄: results/{seed}
        seed_dir = os.path.join(RESULTS_ROOT, str(seed))

        # 清空舊的 Seed 目錄
        if os.path.exists(seed_dir):
            cl_logger.warning(f"[System] 偵測到舊的 Seed 目錄，正在清空: {seed_dir}\n")
            shutil.rmtree(seed_dir)

        # 建立目錄 results/{seed}
        os.makedirs(seed_dir, exist_ok=True)

        # 定義 main.py 的詳細 Log 路徑: results/{seed}/run.log
        seed_log_path = os.path.join(seed_dir, "run.log")

        # 清空或是建立新的 seed run.log (若是重跑，先清空舊的)
        with open(seed_log_path, 'w', encoding='utf-8') as f:
            pass

        # 定義圖片目錄: results/{seed}/all_plots
        all_plots = os.path.join(seed_dir, "all_plots")
        os.makedirs(all_plots, exist_ok=True)

        # 初始化 CL 狀態變數
        previous_model_path = None
        accumulated_test_data = []
        accumulated_test_labels = []

        for step, dataset_name in enumerate(TASK_ORDER):
            task_type = DATASET_TASK_MAP.get(dataset_name)
            if not task_type:
                cl_logger.error(f"找不到資料集 {dataset_name} 的任務類型映射！")
                return

            cl_logger.info(f">>> [Step {step+1}/{len(TASK_ORDER)}] Dataset: {dataset_name} ({task_type})")

            # 1. 準備路徑
            dataset_path = os.path.join(DATA_ROOT, task_type, dataset_name)
            train_file = os.path.join(dataset_path, "train.json")
            eval_file = os.path.join(dataset_path, "dev.json")
            test_file = os.path.join(dataset_path, "test.json")
            labels_file = os.path.join(dataset_path, "labels.json")

            # 定義輸出目錄: results/{seed}/{dataset_name}
            output_dir = os.path.join(seed_dir, dataset_name)
            os.makedirs(output_dir, exist_ok=True)

            # 定義該任務的圖片目錄: results/{seed}/all_plots/{dataset_name}
            task_plot_dir = os.path.join(all_plots, dataset_name)
            os.makedirs(task_plot_dir, exist_ok=True)
            
            # 2. 累積測試資料 (Accumulated Testing)
            accumulated_test_data.append(test_file)
            accumulated_test_labels.append(labels_file)

            # 3. 組合 Command
            cmd = [
                "python", "main.py",
                "--data_file", train_file,
                "--labels_file", labels_file,
                "--eval_file", eval_file,
                "--eval_labels_files", labels_file,
                "--output_dir", output_dir,
                "--plot_dir", task_plot_dir,
                "--dataset_name", dataset_name,
                "--base_model_name", BASE_MODEL,
                "--seed", str(seed)
            ]

            # 加入通用參數
            for k, v in COMMON_ARGS.items():
                cmd.append(k)
                if v: # 如果有值就加，如果是 flag (空字串) 也加 key
                    cmd.append(v)

            # [CL 關鍵] 模型路徑處理
            if previous_model_path:
                # Task 2+: 使用上一個任務訓練好的模型
                cmd.extend(["--model_path", previous_model_path])
            else:
                # Task 1: 不加 --model_path，main.py 會讀取 base_model_name
                pass

            # [CL 關鍵] 測試資料列表 (包含過去所有任務 + 當前任務)
            if accumulated_test_data:
                cmd.append("--test_data_files")
                cmd.extend(accumulated_test_data)
                cmd.append("--test_labels_files")
                cmd.extend(accumulated_test_labels)

            # 4. 執行訓練
            try:
                run_subprocess(cmd, cl_logger, seed_log_path)
                
                # 更新 previous_model_path 為當前輸出的模型，供下一個任務使用
                previous_model_path = output_dir
                cl_logger.info(f"✅ {dataset_name} 訓練完成。模型已儲存至: {output_dir}\n")

            except RuntimeError:
                cl_logger.error(f"⛔ Seed {seed} 在任務 {dataset_name} 中斷。停止該 Seed 的後續任務。")
                break # 跳出 dataset loop，繼續下一個 seed

        cl_logger.info(f"================ Seed {seed} Finished! ================\n")

    cl_logger.info("🎉 所有實驗執行完畢！\n")

if __name__ == "__main__":
    main()