# model/forward_modifier.py
import torch.nn as nn
from transformers.models.t5.modeling_t5 import T5Attention, T5DenseActDense, T5DenseGatedActDense
from model.layers import LoRALinear, LoRALayer, MoEBlock

def apply_lora_to_attention(model, dynamic_expansion=False, rank: int = 4, lora_alpha: int = 32):
    """
    將 T5 的 Attention 層 (q, v) 替換成 LoRALinear
    """
    for layer in model.modules():
        if isinstance(layer, T5Attention):
            # 依序替換 module 內的 Linear 層
            if hasattr(layer, 'q'):
                layer.q = LoRALinear(layer.q, dynamic_expansion, rank, lora_alpha)
            if hasattr(layer, 'v'):
                layer.v = LoRALinear(layer.v, dynamic_expansion, rank, lora_alpha)
                
    return model

def apply_lora_to_ffn(model, dynamic_expansion=False, rank: int = 4, lora_alpha: int = 32):
    """
    將 T5 的 FFN 層替換成單一 LoRALayer
    """
    # 定義替換邏輯
    def replace_ffn_with_lora(layer):
        # 如果這個層是 T5 的 FFN
        if isinstance(layer, (T5DenseActDense, T5DenseGatedActDense)):
            # 換成 LoRALayer
            return LoRALayer(layer, dynamic_expansion, rank, lora_alpha)
        return layer

    # Encoder 的結構：[0]:Attention -> [1]:FFN
    for layer in model.encoder.block:
        layer.layer[1].DenseReluDense = replace_ffn_with_lora(layer.layer[1].DenseReluDense)
    # Decoder 的結構：[0]:Self-Attn -> [1]:Cross-Attn -> [2]:FFN
    for layer in model.decoder.block:
        layer.layer[2].DenseReluDense = replace_ffn_with_lora(layer.layer[2].DenseReluDense)
    
    return model

def apply_moe_to_ffn(model, dynamic_expansion=False, num_experts: int = 4, expert_rank: int = 8, lora_alpha: int = 32, top_k: int = 2, task_embedding_dim=0):
    """
    將 T5 的 FFN 層替換成 MoEBlock
    """
    def replace_ffn_with_moe(layer):
        if isinstance(layer, (T5DenseActDense, T5DenseGatedActDense)):
            # 換成 MoEBlock
            return MoEBlock(layer, dynamic_expansion, num_experts, expert_rank, lora_alpha, top_k, task_embedding_dim)
        return layer
    
    for layer in model.encoder.block:
        layer.layer[1].DenseReluDense = replace_ffn_with_moe(layer.layer[1].DenseReluDense)
    for layer in model.decoder.block:
        layer.layer[2].DenseReluDense = replace_ffn_with_moe(layer.layer[2].DenseReluDense)
    
    return model