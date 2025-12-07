# helper/inference_efficiency.py
import torch
import time
from thop import profile
import types
from model.layers import MoEBlock
import torch.nn as nn
import subprocess

'''
計算推理時間
'''
def measure_inference_time(model, dataloader, device):
    model.eval()
    total_time = 0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            start_time = time.time()
            _ = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=10
            )
            end_time = time.time()
            total_time += (end_time - start_time)
            num_samples += len(batch["input_ids"])
    
    avg_time = total_time / num_samples
    print(f"Average inference time per sample: {avg_time:.6f} seconds")
    return avg_time

'''
計算 FLOPs
衡量模型計算成本
'''
def patched_forward(self, hidden_states, *args, **kwargs):
    # 強制設定 update_counts 為 False
    kwargs['update_counts'] = False
    # 呼叫原本保存下來的 forward 方法
    return self.original_forward(hidden_states, *args, **kwargs)

# 建立一個包裝器 Module
class T5Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # 這裡是原本的 T5ForConditionalGeneration

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        # 在這裡，你可以明確指定 T5ForConditionalGeneration.forward() 的引數
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )
def calculate_total_flops(model, input_size=(1, 256), device=None):
    # 遍歷模型中的所有 MoEBlock，並替換 forward 方法
    for module in model.modules():
        if isinstance(module, MoEBlock):
            # 如果尚未保存原本的 forward 方法，先保存
            if not hasattr(module, 'original_forward'):
                module.original_forward = module.forward
            # 使用 types.MethodType 將 patched_forward 綁定到該 module 上
            module.forward = types.MethodType(patched_forward, module)
    device = device or next(model.parameters()).device
    batch_size, seq_len = input_size

    # 取出正確 vocab size
    vocab_size = model.config.vocab_size

    # 建立 encoder 端的 dummy
    dummy_input_ids = torch.randint(
        low=0, high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long
    ).to(device)
    dummy_attention_mask = torch.ones_like(dummy_input_ids).to(device)

    # 建立 decoder 端的 dummy
    dec_seq_len = 16
    dummy_decoder_input_ids = torch.randint(
        low=0, high=vocab_size,
        size=(batch_size, dec_seq_len),
        dtype=torch.long
    ).to(device)
    dummy_decoder_attention_mask = torch.ones_like(dummy_decoder_input_ids).to(device)

    # 包裝器參考先前的寫法
    wrapped_model = T5Wrapper(model)
    flops, params = profile(
        wrapped_model,
        inputs=(
            dummy_input_ids,
            dummy_attention_mask,
            dummy_decoder_input_ids,
            dummy_decoder_attention_mask
        )
    )
    print(f"Total FLOPs: {flops:.2f}, Total Params: {params}")
    return flops, params

def calculate_moe_flops(model, input_size=(1, 256), device=None):
    from model.layers import MoEBlock
    # 遍歷模型中的所有 MoEBlock，並替換 forward 方法
    for module in model.modules():
        if isinstance(module, MoEBlock):
            # 如果尚未保存原本的 forward 方法，先保存
            if not hasattr(module, 'original_forward'):
                module.original_forward = module.forward
            # 使用 types.MethodType 將 patched_forward 綁定到該 module 上
            module.forward = types.MethodType(patched_forward, module)
    device = device or next(model.parameters()).device
    moe_flops = 0
    for module in model.modules():
        if isinstance(module, MoEBlock):
            # 取得輸入維度，例如從 gate 的 in_features
            input_dim = module.router.gate.in_features
            dummy_hidden = torch.randn(1, input_dim).to(device)  # 建立 float tensor
            # 使用已經 monkey-patched 過的 forward（更新 selection_counts 為 False）
            flops, _ = profile(module, inputs=(dummy_hidden,))
            moe_flops += flops
    print(f"Total MoE FLOPs: {moe_flops:.2f}")
    return moe_flops


'''
計算GPU使用量
'''
def get_vram_usage():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, text=True
    )
    used, total = map(int, result.stdout.strip().split(', '))
    usage_ratio = used / total
    return used, total, usage_ratio

# used, total, usage = get_vram_usage()
# print(f"GPU VRAM 使用量: {used} MB / {total} MB ({usage:.2%})")