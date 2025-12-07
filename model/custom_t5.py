# model/custom_t5.py
import torch
from transformers import T5ForConditionalGeneration
import os
import json
import logging

from .forward_modifier import apply_lora_to_ffn, apply_moe_to_ffn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomT5Model:
    def __init__(self, base_model_path: str, device: torch.device = None, num_experts: int = 4, expert_rank: int = 4):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(base_model_path).to(self.device)
        self.num_experts = num_experts     # <--- 新增用於儲存模型於下一輪訓練時載入可以抓到
        self.expert_rank = expert_rank     # <--- 新增用於儲存模型於下一輪訓練時載入可以抓到
        # apply_lora_to_ffn, apply_moe_to_ffn: 用於修改模型的前向傳播層（Feed-Forward Network）
        # 現在只有用了moe_to_ffn
        self.model = apply_moe_to_ffn(self.model, num_experts=num_experts, expert_rank=expert_rank)
        self.model.to(self.device)

    def save_pretrained(self, save_directory: str):
        """
        保存模型及其自定義配置。
        """
        os.makedirs(save_directory, exist_ok=True)

        # 使用 Hugging Face 的方法保存模型
        self.model.save_pretrained(save_directory)

        # 保存自定義配置
        config = {
            "num_experts": self.num_experts,
            "expert_rank": self.expert_rank
        }
        with open(os.path.join(save_directory, "custom_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    @classmethod
    def load_pretrained(cls, load_directory: str, device: torch.device = None):
        """
        從保存目錄加載模型及其自定義配置。
        """
        # 嘗試使用標準的 from_pretrained 方法
        try:
            # 這會嘗試載入 'pytorch_model.bin' 等標準檔案
            model = T5ForConditionalGeneration.from_pretrained(load_directory).to(device)
            logger.info("使用 from_pretrained 成功載入模型。")
        except OSError:
            logger.warning("標準模型權重文件未找到，嘗試從 'model_state_dict.pt' 載入。")
            # 如果找不到標準檔案，嘗試載入自定義的 state_dict
            model = T5ForConditionalGeneration.from_pretrained("./initial_model/t5-large").to(device)
            state_dict_path = os.path.join(load_directory, "model_state_dict.pt")
            if not os.path.exists(state_dict_path):
                raise FileNotFoundError(f"找不到 state_dict 文件: {state_dict_path}")
            state_dict = torch.load(state_dict_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            logger.info("從 'model_state_dict.pt' 成功載入模型權重。")

        # 載入自定義配置
        config_path = os.path.join(load_directory, "custom_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到自定義配置文件: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        num_experts = config.get("num_experts", 4)
        expert_rank = config.get("expert_rank", 8)

        # 初始化實例
        instance = cls(
            base_model_path="./initial_model/t5-large",  # 使用基礎模型路徑
            device=device,
            num_experts=num_experts,
            expert_rank=expert_rank
        )

        # 設定模型權重
        instance.model = model
        instance.model = apply_moe_to_ffn(instance.model, num_experts=num_experts, expert_rank=expert_rank)
        instance.model.to(device)

        return instance