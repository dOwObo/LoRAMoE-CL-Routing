## 目錄結構
```text
.
├── run_cl.py                    # CL 實驗流程
├── main.py                      # 主程式（單任務實驗）
├── requirements.txt
│
├── model/
│   ├── custom_t5.py             # T5 模型封裝（支援 LoRA/MoE 切換與動態擴充）
│   ├── forward_modifier.py      # 修改 T5 FFN 層的 Forward 邏輯
│   └── layers.py                # 定義 LoRALayer, MoEBlock, Router 與 Loss 計算
│
├── dataset
│   └── data_processor.py        # 讀取資料、Tokenization 與 Prompt 構建
│
├── helper/
│   ├── trainer.py               # 訓練迴圈與驗證邏輯
│   ├── utils.py                 # 視覺化繪圖與計算 Accuracy
│   ├── logging.py               # 統一的 Log 格式設定
│   └── inference_efficiency.py  # 模型效能評估
│
├── initial_model/
│   └── t5-large/ ...            # HuggingFace 原始 T5-Large 權重
│
├── CL_Benchmark/
│   ├── TC/                      # Text Classification
│   │   ├── dbpedia/ ...
│   │   ├── yahoo/ ...
│   │   └── agnews/ ...
│   └── SC/                      # Sentiment Classification
│       └── amazon/ ...
│
├── configs/
│   ├── task.json                # 任務與資料集對應設定
│   └── instruction_config.json  # 定義各任務的 Prompt Template
│
└── results/
```

## 如何運行

#### 1. 建立 Conda 虛擬環境
```bash
# 建立虛擬環境
conda create -n loramoe-cl python=3.8

# 啟動虛擬環境
conda activate loramoe-cl
```

#### 2. 安裝必要套件
詳細套件版本請查看 `requirements.txt`。
```bash
# 安裝 PyTorch
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 安裝其餘套件
pip install -r requirements.txt
```

#### 3. 下載預訓練模型
從 Hugging Face 下載 `t5-large` 模型並放置於 `initial_model/` 目錄下。
```bash
# 確保 git-lfs 已安裝
git lfs install

# 下載模型
git clone https://huggingface.co/google-t5/t5-large initial_model/t5-large
```

#### 4. 執行程式（CL 實驗流程）
請先在 `run_cl.py` 上方設定實驗參數，再以背景方式執行以下指令。
```bash
nohup python run_cl.py > /dev/null 2>&1 &
```
* `--adapter_type`：選擇適配器類型 `"LoRA"` 或 `"MoEBlock"`。
* `--dynamic_expansion`：若啟用動態擴充，每個新任務會凍結舊參數並新增一組新參數。
* `--num_experts`：MoE 架構中的專家數量。
* `--expert_rank`：LoRA 矩陣的 Rank 大小。
* `--top_k`：Router 選擇的專家數量。
* `--num_epochs`：每個任務的訓練輪數。
* `--lr`：學習率 (Learning Rate)。
* `--batch_size`：單卡訓練的批次大小。
* `--accumulation_steps`：梯度累積步數。
* `--lambda_orth`：正交損失 (Orthogonal Loss) 的權重。
* `--lambda_balance`：負載均衡損失 (Load Balancing Loss) 的權重。
* `--debug`：若啟用除錯模式，將使用極少量數據進行測試流程。

#### 5. 輸出結果
```text
results/
├── cl.log                                  # CL 實驗流程日誌
└── {seed}/
    ├── run.log                             # 完整執行日誌
    ├── metrics.csv                         # Loss 與 Accuracy 紀錄
    ├── cl_results.csv                      # CL 測試結果 (評估災難性遺忘程度)
    ├── dbpedia/ ...                        # 訓練後的模型存檔
    ├── amazon/ ...
    ├── yahoo/ ...
    ├── agnews/ ...
    │
    └── all_plots/                          # 視覺化圖表
        ├── dbpedia/ ...
        │   ├── layers_encoder/ ...         # Encoder 各層專家的選擇頻率長條圖
        │   ├── layers_decoder/ ...         # Decoder 各層專家的選擇頻率長條圖
        │   ├── expert_heatmap_encoder.png  # Encoder 專家使用率熱力圖
        │   ├── expert_heatmap_decoder.png  # Decoder 專家使用率熱力圖
        │   └── training_metrics.png        # Loss 與 Accuracy 訓練曲線
        ├── amazon/ ...
        ├── yahoo/ ...
        └── agnews/ ...
```