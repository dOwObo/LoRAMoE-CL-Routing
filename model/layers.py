# model/layers.py
import torch
import torch.nn as nn
import math
from transformers.models.t5.modeling_t5 import T5DenseActDense

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank: int = 4):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        # 定義兩個低秩矩陣lora_A和lora_B
        # 分別用於降維和升維
        self.lora_A = nn.Parameter(torch.zeros((self.rank, original_layer.wi.weight.size(1)))) 
        self.lora_B = nn.Parameter(torch.zeros((original_layer.wi.weight.size(0), self.rank))) 
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) # 初始化lora_A使用Kaiming均勻初始化
        nn.init.zeros_(self.lora_B) # lora_B初始化為零
        self.dropout = nn.Dropout(0.1) # Dropout層以防止過擬合

    def forward(self, hidden_states):
        intermediate = self.original_layer.wi(hidden_states) # 通過原始層的權重進行前向傳播=>中間結果 'intermediate'
        lora_output = self.dropout(hidden_states @ self.lora_A.T) @ self.lora_B.T # 計算LoRA的輸出lora_output
        intermediate = intermediate + lora_output #  加到intermediate上
        intermediate = self.original_layer.act(intermediate)
        intermediate = self.original_layer.dropout(intermediate)
        output = self.original_layer.wo(intermediate)
        return output

class Router(nn.Module):
    # 根據輸入的隱藏狀態選擇專家
    def __init__(self, input_dim, num_experts):
        super().__init__() 
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        scores = torch.softmax(logits, dim=-1)
        top_k_experts = torch.argmax(scores, dim=-1)
        return top_k_experts, scores

# class MoEBlock(nn.Module):
#     def __init__(self, original_layer, num_experts=4, expert_rank=4):
#         super().__init__()
#         self.router = Router(original_layer.wi.weight.size(1), num_experts)
#         self.experts = nn.ModuleList([LoRALayer(original_layer, rank=expert_rank) for _ in range(num_experts)])
#         # self.selection_counts = torch.zeros(num_experts, dtype=torch.long)  # 專家選擇次數計數

#     def forward(self, hidden_states):
#         top_k_experts, _ = self.router(hidden_states)
#         # top_k_experts = top_k_experts.view(-1)  # 保證是 1 維
#         # self.selection_counts += torch.bincount(
#         #     top_k_experts, minlength=len(self.experts)
#         # )  # 更新計數
#         # self.selection_counts += torch.bincount(top_k_experts, minlength=len(self.experts))  # 更新計數
#         outputs = torch.zeros_like(hidden_states)
#         for i, expert in enumerate(self.experts):
#             mask = (top_k_experts == i).unsqueeze(-1).float()
#             expert_output = expert(hidden_states)
#             outputs += mask * expert_output
#         return outputs
class MoEBlock(nn.Module):
    # 將MoE結構應用到前向傳播層
    # 結合多個LoRA專家
    def __init__(self, original_layer, num_experts=4, expert_rank=4):
        super().__init__()
        self.router = Router(original_layer.wi.weight.size(1), num_experts)
        self.experts = nn.ModuleList([LoRALayer(original_layer, rank=expert_rank) for _ in range(num_experts)])
        self.selection_counts = torch.zeros(
            num_experts, dtype=torch.long, device=original_layer.wi.weight.device
        )  # 初始化到模型設備

    def forward(self, hidden_states):
        # 確保 hidden_states 在正確設備
        hidden_states = hidden_states.to(self.router.gate.weight.device)

        # 計算專家選擇
        top_k_experts, _ = self.router(hidden_states)

        # 確保 selection_counts 在正確設備
        self.selection_counts = self.selection_counts.to(hidden_states.device)

        # 更新專家選擇計數
        self.selection_counts += torch.bincount(
            top_k_experts.view(-1), minlength=len(self.experts)
        )

        outputs = torch.zeros_like(hidden_states)
        for i, expert in enumerate(self.experts):
            mask = (top_k_experts == i).unsqueeze(-1).float()
            expert_output = expert(hidden_states)
            outputs += mask * expert_output
        return outputs