# model/forward_modifier.py
import math
import torch
import torch.nn as nn
from transformers.models.t5.modeling_t5 import T5DenseActDense

# model/forward_modifier.py
from model.layers import LoRALayer, MoEBlock

def apply_lora_to_ffn(model, rank: int = 4):
    # 將LoRA應用到T5模型的前向傳播層
    # 如果層是T5DenseActDense
    # 則替換為LoRALayer
    def replace_ffn_with_lora(layer):
        if isinstance(layer, T5DenseActDense):
            return LoRALayer(layer, rank)
        return layer
    
    # 遍歷模型的編碼器和解碼器的每個層
    # 替換相應的前向傳播層
    for layer in model.encoder.block:
        layer.layer[1].DenseReluDense = replace_ffn_with_lora(layer.layer[1].DenseReluDense)
    for layer in model.decoder.block:
        layer.layer[2].DenseReluDense = replace_ffn_with_lora(layer.layer[2].DenseReluDense)
    return model

def apply_moe_to_ffn(model, num_experts=4, expert_rank=4):
    # 將MoE應用到T5模型的前向傳播層
    def replace_ffn_with_moe(layer):
        if isinstance(layer, T5DenseActDense):
            return MoEBlock(layer, num_experts=num_experts, expert_rank=expert_rank)
        return layer
    for layer in model.encoder.block:
        layer.layer[1].DenseReluDense = replace_ffn_with_moe(layer.layer[1].DenseReluDense)
    for layer in model.decoder.block:
        layer.layer[2].DenseReluDense = replace_ffn_with_moe(layer.layer[2].DenseReluDense)
    return model