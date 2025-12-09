# model/forward_modifier.py
import torch.nn as nn
from transformers.models.t5.modeling_t5 import T5DenseActDense, T5DenseGatedActDense
from model.layers import LoRALayer, MoEBlock

def apply_lora_to_ffn(model, dynamic_expansion=False, rank: int = 4):
    """
    將 T5 的 FFN 層替換成單一 LoRALayer
    """
    # 定義替換邏輯
    def replace_ffn_with_lora(layer):
        # 如果這個層是 T5 的 FFN
        if isinstance(layer, (T5DenseActDense, T5DenseGatedActDense)):
            # 換成 LoRALayer
            return LoRALayer(layer, dynamic_expansion, rank)
        return layer

    # Encoder 的結構：[0]:Attention -> [1]:FFN
    for layer in model.encoder.block:
        layer.layer[1].DenseReluDense = replace_ffn_with_lora(layer.layer[1].DenseReluDense)
    # Decoder 的結構：[0]:Self-Attn -> [1]:Cross-Attn -> [2]:FFN
    for layer in model.decoder.block:
        layer.layer[2].DenseReluDense = replace_ffn_with_lora(layer.layer[2].DenseReluDense)
    
    return model

def apply_moe_to_ffn(model, dynamic_expansion=False, num_experts=4, expert_rank=8, top_k=2):
    """
    將 T5 的 FFN 層替換成 MoEBlock
    """
    def replace_ffn_with_moe(layer):
        if isinstance(layer, (T5DenseActDense, T5DenseGatedActDense)):
            # 換成 MoEBlock
            return MoEBlock(layer, dynamic_expansion, num_experts=num_experts, expert_rank=expert_rank, top_k=top_k)
        return layer
    
    for layer in model.encoder.block:
        layer.layer[1].DenseReluDense = replace_ffn_with_moe(layer.layer[1].DenseReluDense)
    for layer in model.decoder.block:
        layer.layer[2].DenseReluDense = replace_ffn_with_moe(layer.layer[2].DenseReluDense)
    
    return model