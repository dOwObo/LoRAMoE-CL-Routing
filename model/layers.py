# model/layers.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import T5DenseActDense

from helper.logging import setup_logger

logger = setup_logger(__name__)

def orthogonal_projection(old_param, new_param):
    """
    將新參數投影到與舊參數正交的空間，減少對舊知識的干擾
    """
    if old_param is None or old_param.numel() == 0:
        return new_param
    
    Q, _ = torch.linalg.qr(old_param.T)
    new_proj = new_param - (new_param @ Q) @ Q.T

    return new_proj

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank: int = 4, dynamic_expansion: bool = False):
        """
        Args:
            dynamic_expansion (bool): 
                True  -> 列表儲存，任務增加時 append，Forward 加總所有參數
                False -> 單一參數，任務增加時更新同一組參數
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.dynamic_expansion = dynamic_expansion

        # 根據 CL 策略初始化
        if self.dynamic_expansion:
            # LoRA 參數池
            self.lora_As = nn.ParameterList()
            self.lora_Bs = nn.ParameterList()

            # 初始化第一組 LoRA 參數
            self.new_pair_parameters()
        else:
            # 定義 LoRA 兩個低秩矩陣，lora_A: (rank, in_dim) -> 負責降維，lora_B: (out_dim, rank) -> 負責升維
            self.lora_A = nn.Parameter(torch.zeros((self.rank, original_layer.wi.weight.size(1)))) 
            self.lora_B = nn.Parameter(torch.zeros((original_layer.wi.weight.size(0), self.rank)))

            self.reset_parameters()
        
        self.dropout = nn.Dropout(0.1)

    def reset_parameters(self):
        """
        初始化單一參數模式的權重
        """
        if not self.dynamic_expansion:
            # lora_A 使用 Kaiming 初始化，保留變異數
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # lora_B 初始化為 0，確保訓練初期 LoRA 輸出為 0，不影響原始模型表現
            nn.init.zeros_(self.lora_B)

    def new_pair_parameters(self):
        """
        新增一組 LoRA 參數
        """
        device = self.original_layer.wi.weight.device
        
        new_A = nn.Parameter(torch.zeros((self.rank, self.original_layer.wi.weight.size(1)), device=device))
        new_B = nn.Parameter(torch.zeros((self.original_layer.wi.weight.size(0), self.rank), device=device))

        nn.init.kaiming_uniform_(new_A, a=math.sqrt(5))

        # 正交投影 (僅在已有舊參數時執行)
        if len(self.lora_As) > 0:
            old_A = self.lora_As[-1].detach()
            new_A.data = orthogonal_projection(old_A, new_A.data)
        
        nn.init.zeros_(new_B)

        self.lora_As.append(new_A)
        self.lora_Bs.append(new_B)

    def add_new_task(self):
        """
        當新任務來時，動態新增 LoRA 參數
        """
        if self.dynamic_expansion:
            # 凍結舊參數
            for param in self.lora_As: param.requires_grad = False
            for param in self.lora_Bs: param.requires_grad = False
            # 建立新參數
            self.new_pair_parameters()
            logger.info(f"[Model] LoRALayer Expanded: Now has {len(self.lora_As)} adapters.")
        else:
            # 保留舊知識，不重置
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
            logger.info("[Model] LoRALayer Fixed: Ready for new task training (params reused).")

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (Batch, Seq_Len, Dim)
        """
        # 1. 原始路徑 (凍結的 T5 wi 層)
        intermediate = self.original_layer.wi(hidden_states)

        # 2. LoRA 路徑，先降維 (x @ A.T) -> Dropout -> 再升維 (@ B.T)
        lora_output = torch.zeros_like(intermediate)
        if self.dynamic_expansion:
            for A, B in zip(self.lora_As, self.lora_Bs):
                lora_output += (self.dropout(hidden_states) @ self.lora_A.T) @ self.lora_B.T
                """
                # [Debug] 觀察 LoRA 有沒有學到東西
                lora_norm = lora_output.norm().item()
                lora_total_norm += lora_norm
                """
        else:
            lora_output = (self.dropout(hidden_states) @ self.lora_A.T) @ self.lora_B.T
        
        # 3. 加總: T5 (原始知識) + LoRA (微調知識)
        intermediate = intermediate + lora_output

        # 4. 原始 T5 的激活函數 (ReLU 或 GELU) 與 Dropout
        intermediate = self.original_layer.act(intermediate)
        intermediate = self.original_layer.dropout(intermediate)
        
        # 5. 輸出，通過 T5 FFN 的第二層 Linear (wo)
        output = self.original_layer.wo(intermediate)
        
        return output
    
    def compute_orth_loss(self, lambda_orth: float) -> torch.Tensor:
        """
        計算正交 Loss ，只在 Dynamic Expansion CL 有效
        """
        if not self.dynamic_expansion or len(self.lora_As) < 2:
            return 0.0
        
        loss = 0
        # 計算最後一組(當前訓練中)與前面所有組的正交性
        current_A = self.lora_As[-1]
        current_B = self.lora_Bs[-1]
        for i in range(len(self.lora_As) - 1):
            loss += torch.sum((current_A @ self.lora_As[i].T)**2)
            loss += torch.sum((current_B.T @ self.lora_Bs[i])**2)
            
        return lambda_orth * loss

class Router(nn.Module):
    """
    決定每個 token 該去哪個專家，支援 Top-1 和 Top-K 兩種模式。
    """
    def __init__(self, input_dim, num_experts, top_k=2):
        super().__init__() 
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (Batch, Seq_Len, Dim)
        Returns:
            top_k_experts: indices (Top-1) or (indices, values) (Top-K)
            scores: full probability distribution (Batch, Seq_Len, Num_Experts)
        """
        # 計算每個專家的分數 (Logits)
        logits = self.gate(hidden_states)
        # 計算機率分佈 (Softmax)
        scores = F.softmax(logits, dim=-1)

        # Top-1
        if self.top_k == 1:
            top_k_experts = torch.argmax(scores, dim=-1)
        # Top-K
        else:
            top_k_values, top_k_indices = torch.topk(scores, k=self.top_k, dim=-1)
            top_k_experts = (top_k_indices, top_k_values)
        
        return top_k_experts, scores

class MoEBlock(nn.Module):
    """
    將 T5 原始 FFN 層替換成 Router 和多個 LoRALayer (Experts)。
    """
    def __init__(self, original_layer, dynamic_expansion=False, num_experts=4, expert_rank=4, top_k=2):
        super().__init__()

        # 初始化 Router
        self.router = Router(original_layer.wi.weight.size(1), num_experts, top_k=top_k)
        # 建立專家列表，每個專家都是一個 LoRALayer，共享原本的 T5 權重，但有獨立的 LoRA 參數
        self.experts = nn.ModuleList([LoRALayer(original_layer, dynamic_expansion, rank=expert_rank) for _ in range(num_experts)])
        # 統計數據，記錄每個專家被選擇次數，不存入 model_state_dict
        self.register_buffer("selection_counts", torch.zeros(num_experts, dtype=torch.long), persistent=False)
        # 儲存最後一次的分數 (用於計算 Load Balancing Loss)
        self.last_scores = None

    def add_new_task(self):
        """
        當新任務來時，新增 LoRA 參數或專家數量
        """
        for expert in self.experts:
            expert.add_new_task()

        """
        [新增專家]
        self.experts.append()
        """

        self.reset_stats()

    def reset_stats(self):
        """
        重置統計數據，在每個 Task 開始前呼叫
        """
        self.selection_counts.zero_()

    def forward(self, hidden_states):
        # Router 決定每個 token 要給哪個專家
        top_k_experts, scores = self.router(hidden_states)
        self.last_scores = scores

        # 初始化輸出 Tensor，形狀與 hidden_states 相同
        outputs = torch.zeros_like(hidden_states)

        for i, expert in enumerate(self.experts):

            # Top-K (Soft Routing)
            if isinstance(top_k_experts, tuple):
                top_k_indices, top_k_values = top_k_experts
                # Step 1: 判斷專家 i 是否在 Top-K 名單內 (True/False)
                is_in_top_k = (top_k_indices == i)
                # 略過整個 Batch 都沒有任何 token 選擇的專家 i
                if not is_in_top_k.any():
                    continue
                # Step 2: 計算加權權重
                weight = is_in_top_k.float() * top_k_values
                # Step 3: 加總 k 維度，(batch, seq, k) -> (batch, seq, 1)
                mask = weight.sum(dim=-1, keepdim=True)
            # Top-1 (Hard Routing)
            else:
                # Step 1: 判斷專家 i 是否被選中 (True/False)
                is_selected = (top_k_experts == i)
                # 略過沒有任何 token 選擇的專家 i
                if not is_selected.any():
                    continue
                # Step 2: 建立遮罩 (batch, seq) -> (batch, seq, 1)
                mask = is_selected.float().unsqueeze(-1)

            # 擴展維度以符合 hidden_states，(batch, seq, 1) -> (batch, seq, hidden_dim)
            mask = mask.expand_as(hidden_states)
            expert_output = expert(hidden_states)

            # 專家被選中時進行計算
            outputs += mask * expert_output

            """
            [累加 Mask 值]（已棄用）
            Code: self.selection_counts[i] += int(mask.sum().item())
            原因: 在 Top-k 模式下，mask 裡存放的是 0.0 到 1.0 之間的權重值。直接加總得到的是「該專家的總權重負載 (Total Load)」，而非「被選中的次數」。

            [統計實際大於 0 的 mask]（目前使用）
            將所有大於 0 的值視為 True，只要該專家參與該 Token 的計算，就計為 +1。
            """

            # 更新統計數據
            with torch.no_grad():
                selected_count = (mask[..., 0] > 0).sum()
                self.selection_counts[i] += selected_count
        
        return outputs