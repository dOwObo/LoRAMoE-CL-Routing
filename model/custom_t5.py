# model/custom_t5.py
import os
import json
import torch
from transformers import T5ForConditionalGeneration
from model.layers import LoRALayer, MoEBlock
from model.forward_modifier import apply_lora_to_ffn, apply_moe_to_ffn
from helper.logging import setup_logger

logger = setup_logger(__name__)

class CustomT5Model:
    def __init__(
        self, 
        base_model_path: str, 
        device: torch.device = None,
        adapter_type: str = "MoEBlock", 
        dynamic_expansion: bool = False,
        num_experts: int = 4,  
        expert_rank: int = 8,  
        top_k: int = 2
    ):
        """
        自定義 T5 模型容器，支援 LoRA/MoEBlock 兩種模式切換
        """
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 1. 儲存超參數
        self.base_model_path = base_model_path
        self.adapter_type = adapter_type
        self.dynamic_expansion = dynamic_expansion
        self.num_experts = num_experts
        self.expert_rank = expert_rank
        self.top_k = top_k

        # 2. 載入基礎模型
        logger.info(f"[Model] 初始化 T5: {base_model_path}")
        self.model = T5ForConditionalGeneration.from_pretrained(base_model_path)

        # 3. 根據 adapter_type 決定使用哪種架構
        if adapter_type == "LoRA":
            logger.info(f"[Model] 將 FFN 替換成標準 LoRA 架構: Rank={expert_rank}")
            # 使用 expert_rank 作為 LoRA 的 rank
            self.model = apply_lora_to_ffn(
                self.model, 
                dynamic_expansion,
                rank=expert_rank
            )
        elif adapter_type == "MoEBlock":
            logger.info(f"[Model] 將 FFN 替換成 MoEBlock 架構: Experts={num_experts}, Rank={expert_rank}, TopK={top_k}")
            self.model = apply_moe_to_ffn(
                self.model, 
                dynamic_expansion,
                num_experts, 
                expert_rank,
                top_k
            )
        else:
            msg = f"[Error] Unknown adapter_type: {adapter_type}"
            logger.error(msg)
            raise ValueError(msg)
        
        self.model.to(self.device)
    
    def get_moe_usage(self):
        """
        收集全模型所有 MoEBlock 的專家使用數據
        """
        usage_data = {"encoder": [], "decoder": []}
        
        def get_counts(block, layer_idx):
            """
            安全地從 block 中提取 selection_counts
            """
            if len(block.layer) > layer_idx:
                ffn_layer = block.layer[layer_idx].DenseReluDense
                # 確認是否已經被替換為 MoEBlock (具備 selection_counts)
                if hasattr(ffn_layer, "selection_counts"):
                    # .cpu().numpy().tolist() 確保回傳的是標準 Python List，不佔用 GPU Graph
                    return ffn_layer.selection_counts.cpu().numpy().tolist()
            return None

        # 收集 Encoder (標準 T5 Encoder FFN 在 index 1)
        for block in self.model.encoder.block:
            counts = get_counts(block, 1)
            if counts is not None: 
                usage_data["encoder"].append(counts)

        # 收集 Decoder (標準 T5 Encoder FFN 在 index 2)
        for block in self.model.decoder.block:
            counts = get_counts(block, 2)
            if counts is not None: 
                usage_data["decoder"].append(counts)
            
        return usage_data

    def reset_moe_usage(self):
        """
        重置所有 MoE 層的統計數據，確保統計的是當前任務的專家使用率，而非歷史累積
        """
        count = 0

        # 重置 Encoder 的 MoE 層
        for block in self.model.encoder.block:
            if hasattr(block.layer[1].DenseReluDense, "reset_stats"):
                block.layer[1].DenseReluDense.reset_stats()
                count += 1

        # 重置 Decoder 的 MoE 層
        for block in self.model.decoder.block:
            if hasattr(block.layer[2].DenseReluDense, "reset_stats"):
                block.layer[2].DenseReluDense.reset_stats()
                count += 1

        logger.info(f"[Model] 已重置 {count} 個 MoE 層的統計數據 (歸零)")

    def expand_model_structure(self):
        """
        遍歷模型所有層，對 LoRA/MoEBlock 擴充參數結構
        """
        count = 0
        for module in self.model.modules():
            # LoRA 和 MoE 內部的 Expert LoRA，執行擴充
            if isinstance(module, LoRALayer):
                if hasattr(module, "add_new_task"):
                    module.add_new_task()
                    count += 1

        self.model.to(self.device)
        logger.info(f"[Model] Structure Expanded: {count} LoRA modules updated.")

    def save_pretrained(self, save_directory: str):
        """
        保存模型權重及其自定義配置。
        """
        os.makedirs(save_directory, exist_ok=True)

        # 偵測目前的擴充次數
        expansion_count = 1
        try:
            # 取得第一個 Block 的 FFN 層
            sample_layer = self.model.encoder.block[0].layer[1].DenseReluDense
            
            if isinstance(sample_layer, LoRALayer) and sample_layer.dynamic_expansion:
                expansion_count = len(sample_layer.lora_As)
            elif isinstance(sample_layer, MoEBlock) and sample_layer.experts[0].dynamic_expansion:
                expansion_count = len(sample_layer.experts[0].lora_As)
        except Exception as e:
            logger.warning(f"[Warning] 無法偵測擴充次數，預設為 1: {e}")

        # 1. 保存模型權重
        state_dict_path = os.path.join(save_directory, "model_state_dict.pt")
        torch.save(self.model.state_dict(), state_dict_path)
        logger.info(f"[System] 模型權重已保存: {state_dict_path}")

        # 2. 保存自定義配置
        config = {
            "base_model_path": self.base_model_path,
            "adapter_type": self.adapter_type,
            "dynamic_expansion": self.dynamic_expansion,
            "expansion_count":expansion_count,
            "num_experts": self.num_experts,
            "expert_rank": self.expert_rank,
            "top_k": self.top_k
        }
        config_path = os.path.join(save_directory, "custom_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        logger.info(f"[System] 自定義配置已保存: {config_path}")

    @classmethod
    def load_pretrained(cls, load_directory: str, device: torch.device = None):
        """
        從保存目錄加載模型及其自定義配置，讀取 Config -> 建立完整 MoE 架構 -> 載入權重。
        """
        # 1. 讀取自定義配置
        config_path = os.path.join(load_directory, "custom_config.json")
        if not os.path.exists(config_path):
            msg = f"[Error] 找不到自定義配置文件: {config_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 讀取參數
        base_model_path = config.get("base_model_path")
        adapter_type = config.get("adapter_type", "MoEBlock")
        dynamic_expansion = config.get("dynamic_expansion", False)
        expansion_count = config.get("expansion_count", 1)
        num_experts = config.get("num_experts", 4)
        expert_rank = config.get("expert_rank", 8)
        top_k = config.get("top_k", 2)

        # 2. 初始化實例，建立帶有 LoRA/MoEBlock 結構
        instance = cls(
            base_model_path=base_model_path,
            device=device,
            adapter_type=adapter_type,
            dynamic_expansion=dynamic_expansion,
            num_experts=num_experts,
            expert_rank=expert_rank,
            top_k=top_k
        )

        # 3. 手動擴充模型架構以匹配存檔
        if dynamic_expansion and expansion_count > 1:
            logger.info(f"[Model] 偵測到存檔包含 {expansion_count} 組歷史參數，正在重建模型結構...")
            # 初始建立時已有 1 組，所以需額外擴充 (count - 1) 次
            for i in range(expansion_count - 1):
                instance.expand_model_structure()
                logger.info(f"[Model] Restoring structure: Expanded to task {i+2}")

        # 4. 載入微調後的權重
        state_dict_path = os.path.join(load_directory, "model_state_dict.pt")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=device)
            keys_missing, keys_unexpected = instance.model.load_state_dict(state_dict, strict=False)

            # 檢查缺失
            if keys_missing:
                logger.warning(f"[Warning] 載入權重時缺失了部分 Key (可能正常，如未微調部分): {keys_missing[:5]}...")

            # 檢查多餘
            if keys_unexpected:
                logger.warning(f"[Warning] 檔案中包含未使用的多餘權重: {keys_unexpected[:5]}...")

            logger.info(f"[Model] 成功載入訓練後 {adapter_type} 模型權重: {state_dict_path}")
        else:
            logger.warning(f"[Warning] 找不到 {state_dict_path}，僅載入了基礎模型與隨機初始化的 MoE 參數")

        
        # 確保模型在正確的 device
        instance.model.to(device)
        
        return instance