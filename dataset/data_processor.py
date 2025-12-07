#  dataset/data_processor.py
import os
import json
import random
import pandas as pd
from torch.utils.data import DataLoader, Subset
from datasets import Dataset
from transformers import AutoTokenizer

from helper.logging import setup_logger

logger = setup_logger(__name__)

class DataProcessor:
    def __init__(self, 
                 data_file: str, 
                 labels_file: str, 
                 peft_model_path: str, 
                 max_input_length: int = 512, 
                 max_label_length: int = 32, 
                 config_dir: str = "configs"):
        """
        讀取資料、標籤映射、Tokenizer 初始化以及資料集的 Dataset/Task 判定
        """
        self.data_file = data_file
        self.labels_file = labels_file
        self.peft_model_path = peft_model_path
        self.max_input_length = max_input_length
        self.max_label_length = max_label_length
        self.config_dir = config_dir

        # 1. 讀取任務配置與 Prompt Template
        self.task_config = self.load_json(os.path.join(self.config_dir, "task.json"))
        self.instruction_config = self.load_json(os.path.join(self.config_dir, "instruction_config.json"))

        # 2. 讀取原始資料
        self.data_df = self.load_data(self.data_file)

        # 3. 讀取標籤
        self.labels_list = self.load_labels(self.labels_file)
        
        # 建立 Label <-> ID 的雙向映射字典
        self.label_to_id = {label: idx for idx, label in enumerate(self.labels_list)}
        self.id_to_label = {idx: label for idx, label in enumerate(self.labels_list)}

        # 取得專案根目錄
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 設定設定備用的基礎模型路徑
        base_model_path = os.path.join(ROOT_DIR, "initial_model/t5-large")

        # 檢查 peft_model_path 是否包含 config.json
        if os.path.exists(os.path.join(self.peft_model_path, "config.json")):
            config_path = self.peft_model_path
        else:
            logger.warning(f"[Warning] `{self.peft_model_path}` 缺少 `config.json`，改用 `{base_model_path}`")
            config_path = base_model_path

        # 4. 初始化 Tokenizer
        logger.info(f"[Data] 使用 tokenizer 設定檔: {config_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config_path)
        self.tokenizer.padding_side = "right"

        # 5. 確定資料集名稱和任務類型
        self.dataset_name = self.get_dataset_name(self.data_file)
        self.task = self.get_task_from_dataset(self.dataset_name)
        self.instruction = self.get_instruction(self.task)

        logger.info(f"[Data] 資料載入完成，Dataset: {self.dataset_name}, Task: {self.task}")

    def load_json(self, filepath):
        """
        加載 JSON 文件
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            logger.error(f"[Error] 配置文件不存在: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"[Error] 解析 JSON 文件失敗: {filepath}, 錯誤: {e}")
            raise

    def load_data(self, filepath):
        """
        加載 Pandas 數據文件
        """
        try:
            data_df = pd.read_json(filepath)
            return data_df
        except ValueError as e:
            logger.error(f"[Error] 讀取數據文件時出錯: {filepath}, 錯誤: {e}")
            raise
        except FileNotFoundError:
            logger.error(f"[Error] 數據文件不存在: {filepath}")
            raise

    def load_labels(self, filepath):
        """
        加載標籤文件並轉換為 Snake_Case
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                labels = json.load(f)

            # 將標籤轉換為 Snake_Case
            labels = [label.replace(" ", "_") for label in labels]
            
            # 檢查是否有重複標籤
            if len(labels) != len(set(labels)):
                msg = "[Error] 轉換後的標籤存在重複，請檢查標籤文件"
                logger.error(msg)
                raise ValueError(msg)
            return labels
        except FileNotFoundError:
            logger.error(f"[Error] 標籤文件不存在: {filepath}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"[Error] 解析標籤 JSON 文件失敗: {filepath}, 錯誤: {e}")
            raise

    def get_dataset_name(self, data_file):
        """
        從 data_file 路徑中提取數據集名稱
        """
        parts = data_file.split(os.sep)
        for part in parts:
            if part in self.get_all_datasets():
                return part
        msg = f"[Error] 無法從 {data_file} 中提取數據集名稱"
        logger.error(msg)
        raise ValueError(msg)

    def get_all_datasets(self):
        """
        從 task_config 中獲取所有數據集名稱
        """
        datasets = []
        # Question: task 沒用到能不能刪掉？
        for task, dataset_list in self.task_config.items():
            for dataset in dataset_list:
                datasets.append(dataset["dataset name"])

        return datasets

    def get_task_from_dataset(self, dataset_name):
        """
        根據數據集名稱從 task_config 中獲取對應的任務，例如：TC, SC
        """
        for task, dataset_list in self.task_config.items():
            for dataset in dataset_list:
                if dataset["dataset name"] == dataset_name:
                    return task
        msg = f"[Error] 未找到 {dataset_name} 對應的任務"
        logger.error(msg)
        raise ValueError(msg)

    def get_instruction(self, task):
        """
        根據任務從 instruction_config 中獲取對應的 Prompt
        """
        if task not in self.instruction_config:
            msg = f"[Error] 未找到任務 {task} 的指令"
            logger.error(msg)
            raise ValueError(msg)
        
        instructions = self.instruction_config[task]
        if not instructions:
            msg = f"[Error] 任務 {task} 的指令列表為空"
            logger.error(msg)
            raise ValueError(msg)
        
        # 假設每個任務只有一個指令，選擇第一個
        return instructions[0]["instruction"]

    def convert_label_to_id(self, example):
        """
        將文字標籤轉換為數字 ID
        """
        if example["label"] in self.label_to_id:
            example["label_id"] = self.label_to_id[example["label"]]
        else:
            msg = f"[Error] 未知標籤: {example['label']}"
            logger.error(msg)
            raise ValueError(msg)
        
        return example

    def preprocess_data(self, example):
        """
        將資料轉換為 Prompt 輸入
        """
        try:
            options = ", ".join(self.labels_list)
            input_text = (
                f"Task:{self.task}\nDataset:{self.dataset_name}\n"
                f"{self.instruction}"
                f"Option: {options}\n"
                f"{example['sentence']}\nAnswer:"
            )
            # 把產生的 prompt 直接加進 example，方便後續 tokenize
            example["input_text"] = input_text
        except KeyError as e:
            msg = f"[Error] 缺少必要的欄位: {e}"
            logger.error(msg)
            raise KeyError(msg)
        
        return example

    def tokenize_data(self, examples):
        """
        批次 Tokenize 函數，同時處理 Input Text 和 Labels
        """
        # 1. Tokenize input_text (Encoder 輸入)
        model_inputs = self.tokenizer(
            examples["input_text"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True
        )

        # 2. Tokenize labels (Decoder 目標)
        label_texts = [self.labels_list[idx] for idx in examples["label_id"]]

        # 使用 tokenizer 處理目標文字
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                label_texts,
                max_length=self.max_label_length,
                padding="max_length",
                truncation=True
            )

        # 3. 處理 Labels 的 Padding Mask
        label_masks = labels["attention_mask"]
        label_ids = labels["input_ids"]

        # 將 Padding 的 Label ID 設為 -100 以忽略 CrossEntropyLoss 計算
        for i in range(len(label_ids)):
            for j in range(len(label_ids[i])):
                if label_masks[i][j] == 0:
                    label_ids[i][j] = -100

        # 加到 model_inputs
        model_inputs["labels"] = label_ids

        return model_inputs

    def get_dataset(self):
        """
        執行完整的資料處理流程，回傳處理好的 Hugging Face Dataset
        """
        # 移除無效標籤
        invalid_labels = [label for label in self.data_df["label"] if label not in self.label_to_id]
        if invalid_labels:
            self.data_df = self.data_df[~self.data_df["label"].isin(invalid_labels)]
            logger.warning(f"[Warning] 資料包含無效標籤，已移除: {invalid_labels}")

        # 移除空句子
        invalid_sentences = self.data_df["sentence"].apply(lambda x: not x.strip())
        if invalid_sentences.any():
            self.data_df = self.data_df[~invalid_sentences]
            logger.warning(f"[Warning] 空句子被移除，Index: {self.data_df[invalid_sentences].index.tolist()}")

        # 轉換標籤為 ID
        self.data_df = self.data_df.apply(self.convert_label_to_id, axis=1)

        # 轉換成 Hugging Face Dataset
        hf_dataset = Dataset.from_pandas(self.data_df)

        # 進行 Prompt 組裝 (單筆)
        hf_dataset = hf_dataset.map(self.preprocess_data)

        # 進行 Tokenize (批次)
        hf_dataset = hf_dataset.map(
            self.tokenize_data,
            batched=True,
            # 只保留模型需要的欄位
            remove_columns=["sentence", "label", "label_id", "input_text"]  
        )

        return hf_dataset

    def get_dataloader(self, dataset, batch_size, collate_fn):
        """
        根據 Dataset 建立 DataLoader
        """
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=collate_fn
        )
    
    # [Testing]
    def get_subset_dataloader(self, dataset, batch_size, collate_fn, subset_size=500, shuffle=True):
        """
        從 dataset 中隨機抽取 subset_size 筆資料，並返回相應的 DataLoader
        """
        dataset_size = len(dataset)
        if subset_size > dataset_size:
            msg = f"[Error] subset_size ({subset_size}) 大於資料集大小 ({dataset_size})"
            logger.error(msg)
            raise ValueError(msg)
        
        # 隨機抽樣
        if shuffle:
            indices = random.sample(range(dataset_size), subset_size)
        # 固定取前 N 筆
        else:
            indices = list(range(subset_size))

        subset = Subset(dataset, indices)

        return self.get_dataloader(subset, batch_size, collate_fn)