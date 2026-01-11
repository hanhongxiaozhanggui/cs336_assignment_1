import torch
import torch.nn as nn
import math
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 32768
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    bias: bool = True

class LayerNorm(nn.Module):
    def __init__(self, dims, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dims))
        self.bias = nn.Parameter(torch.zeros(dims))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # QKV 投影参数
        self.c_attn_weight = nn.Parameter(torch.empty(config.n_embd, 3 * config.n_embd))
        self.c_attn_bias = nn.Parameter(torch.zeros(3 * config.n_embd))
        # 输出投影参数
        self.c_proj_weight = nn.Parameter(torch.empty(config.n_embd, config.n_embd))
        self.c_proj_bias = nn.Parameter(torch.zeros(config.n_embd))
        
        nn.init.normal_(self.c_attn_weight, std=0.02)
        nn.init.normal_(self.c_proj_weight, std=0.02)

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = torch.matmul(x, self.c_attn_weight) + self.c_attn_bias
        q, k, v = qkv.split(self.n_embd, dim=2)

        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # 手动 Softmax
        att_max = torch.max(att, dim=-1, keepdim=True)[0]
        exp_att = torch.exp(att - att_max)
        att = exp_att / exp_att.sum(dim=-1, keepdim=True)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return torch.matmul(y, self.c_proj_weight) + self.c_proj_bias

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(config.n_embd, 4 * config.n_embd))
        self.b1 = nn.Parameter(torch.zeros(4 * config.n_embd))
        self.w2 = nn.Parameter(torch.empty(4 * config.n_embd, config.n_embd))
        self.b2 = nn.Parameter(torch.zeros(config.n_embd))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x):
        x = torch.matmul(x, self.w1) + self.b1
        # GeLU 激活函数
        x = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        return torch.matmul(x, self.w2) + self.b2

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 将 Parameter 移出 ModuleDict，直接作为属性
        self.wte = nn.Parameter(torch.empty(config.vocab_size, config.n_embd))
        self.wpe = nn.Parameter(torch.empty(config.block_size, config.n_embd))
        
        # 2. ModuleDict 只保留真正的 nn.Module
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)
        
        # 初始化权重
        nn.init.normal_(self.wte, std=0.02)
        nn.init.normal_(self.wpe, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        # 1. 前向传播
        x = self.wte[idx] + self.wpe[pos]
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        
        # 2. 计算 Logits
        logits = torch.matmul(x, self.wte.t()) # (B, T, vocab_size)

        # 3. 如果传入了 targets，则计算 loss
        loss = None
        if targets is not None:
            # 将 logits 展平为 (B*T, vocab_size)，targets 展平为 (B*T)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1
            )

        return logits, loss