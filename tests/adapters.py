from __future__ import annotations

from cs336_basics.tokenizer import train_bpe
from cs336_basics.tokenizer import get_tokenizer as get_bpe_tokenizer

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import numpy as np



from cs336_basics.optimizer import (
    AdamW,
    get_lr_cosine_schedule,
    save_checkpoint,
    load_checkpoint,
)
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


def get_adamw_cls():
    return AdamW


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    # 注意 shape: in_features[..., d_in] @ weights.T -> [..., d_out]
    return in_features @ weights.T



def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    return weights[token_ids]



def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    x1 = run_linear(d_model, d_ff, w1_weight, in_features)
    x3 = run_linear(d_model, d_ff, w3_weight, in_features)
    return run_linear(d_ff, d_model, w2_weight, run_silu(x1) * x3)




def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return attn @ V


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    B, L, _ = in_features.shape
    d_k = d_model // num_heads

    # 1. 线性投影 Q, K, V
    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    # 2. 分头 (B, L, H, d_k) -> (B, H, L, d_k)
    Q = Q.view(B, L, num_heads, d_k).transpose(1, 2)
    K = K.view(B, L, num_heads, d_k).transpose(1, 2)
    V = V.view(B, L, num_heads, d_k).transpose(1, 2)

    # 3. 创建因果掩码 (Causal Mask)
    # 创建一个 (L, L) 的下三角矩阵，1表示保留，0表示遮掩
    mask = torch.tril(torch.ones(L, L, device=in_features.device)).bool()
    
    # 4. 计算注意力 (建议直接调用你之前写好的函数)
    # 假设你的 scaled_dot_product_attention 支持 4D 且能处理 mask
    # 如果是手动写：
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    # 将 mask 应用到 scores 上 (未来的位置设为极小值)
    scores = scores.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    context = attn @ V

    # 5. 合并多头 (B, H, L, d_k) -> (B, L, H, d_k) -> (B, L, d_model)
    out = context.transpose(1, 2).contiguous().view(B, L, d_model)

    # 6. 输出投影
    return out @ o_proj_weight.T



def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    shape = in_query_or_key.shape
    L = shape[-2]
    device = in_query_or_key.device

    # 1. 计算频率 (frequencies)
    # inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
    # 注意：为了匹配测试，确保使用正确的 arange 步长
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
    
    # 2. 计算广播后的位置角
    # pos shape: (..., L) -> (..., L, 1)
    pos = token_positions.unsqueeze(-1).float()
    # angles shape: (..., L, d_k/2)
    angles = pos * inv_freq
    
    # 3. 计算 cos 和 sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    # 4. 成对旋转逻辑 [x0, x1, x2, x3] -> [x0*cos-x1*sin, x0*sin+x1*cos, ...]
    # 拆分偶数和奇数索引
    x0 = in_query_or_key[..., 0::2]
    x1 = in_query_or_key[..., 1::2]
    
    output = torch.empty_like(in_query_or_key)
    output[..., 0::2] = x0 * cos - x1 * sin
    output[..., 1::2] = x0 * sin + x1 * cos
    
    return output


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " batch sequence_length d_model"],
    token_positions: Int[Tensor, " batch sequence_length"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    B, L, _ = in_features.shape
    d_k = d_model // num_heads

    # 1. 线性投影 (Linear Projections)
    # PyTorch 的 nn.Linear(in, out) 权重维度通常是 (out, in)，所以这里用 @ weight.T
    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T

    # 2. 分头 (B, L, D) -> (B, L, H, d_k) -> (B, H, L, d_k)
    Q = Q.view(B, L, num_heads, d_k).transpose(1, 2)
    K = K.view(B, L, num_heads, d_k).transpose(1, 2)
    V = V.view(B, L, num_heads, d_k).transpose(1, 2)

    # 3. 应用 RoPE
    # 你的 run_rope 函数定义需要 5 个参数：d_k, theta, max_seq_len, in_query_or_key, token_positions
    if token_positions.ndim == 1:
        token_positions = token_positions.unsqueeze(0)
    
    # 修复调用：补全 d_k 和 max_seq_len 参数
    Q = run_rope(
        d_k=d_k, 
        theta=theta, 
        max_seq_len=max_seq_len, 
        in_query_or_key=Q, 
        token_positions=token_positions
    )
    K = run_rope(
        d_k=d_k, 
        theta=theta, 
        max_seq_len=max_seq_len, 
        in_query_or_key=K, 
        token_positions=token_positions
    )

    # 4. 计算注意力得分
    # scores: (B, H, L, L)
    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)

    # 5. 应用因果掩码 (Causal Mask)
    # 创建下三角矩阵：1 表示可见，0 表示掩盖
    mask = torch.tril(torch.ones((L, L), device=in_features.device)).bool()
    # 将 mask 应用于 scores，掩盖处填入负无穷
    scores = scores.masked_fill(~mask, float("-inf"))

    # 6. Softmax & 加权求和
    attn = torch.softmax(scores, dim=-1)
    context = attn @ V  # (B, H, L, d_k)

    # 7. 合并多头 (B, H, L, d_k) -> (B, L, H, d_k) -> (B, L, D)
    # 使用 .contiguous() 确保内存连续，否则 view 会报错
    out = context.transpose(1, 2).contiguous().view(B, L, d_model)

    # 8. 输出投影
    return out @ o_proj_weight.T


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    # RMSNorm 1
    x = run_rmsnorm(d_model, 1e-5, weights['ln1.weight'], in_features)

    # Multi-head self-attention with RoPE
    attn_out = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights['attn.q_proj.weight'],
        k_proj_weight=weights['attn.k_proj.weight'],
        v_proj_weight=weights['attn.v_proj.weight'],
        o_proj_weight=weights['attn.output_proj.weight'],
        in_features=x,
        token_positions=torch.arange(in_features.shape[1], device=in_features.device),
    )

    # 残差连接
    x = in_features + attn_out

    # RMSNorm 2
    y = run_rmsnorm(d_model, 1e-5, weights['ln2.weight'], x)

    # Feed-Forward Network (SwiGLU)
    ffn_out = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights['ffn.w1.weight'],
        w2_weight=weights['ffn.w2.weight'],
        w3_weight=weights['ffn.w3.weight'],
        in_features=y
    )

    # 残差连接
    out = x + ffn_out
    return out



def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    # Token embedding
    x = run_embedding(vocab_size, d_model, weights['token_embeddings.weight'], in_indices)

    # Transformer 层循环
    for i in range(num_layers):
        layer_weights = {k.split(f'layers.{i}.')[-1]: v for k, v in weights.items() if k.startswith(f'layers.{i}.')}
        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x
        )

    # 最终 RMSNorm
    x = run_rmsnorm(d_model, 1e-5, weights['ln_final.weight'], x)

    # LM Head
    logits = run_linear(d_model, vocab_size, weights['lm_head.weight'], x)

    return logits

def run_transformer_lm_truncated_input(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    # 直接调用原 lm
    return run_transformer_lm(
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights, in_indices
    )

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    # 计算均方根
    norm = torch.sqrt(in_features.pow(2).mean(dim=-1, keepdim=True) + eps)
    # 广播 weights 到最后一维
    return in_features / norm * weights.view(*([1]*(in_features.ndim-1)), -1)



def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return in_features * torch.sigmoid(in_features)



def run_get_batch(dataset, batch_size, context_length, device="cpu"):
    """
    从 dataset 中生成训练批次 x, y
    
    x: 当前上下文
    y: 下一步目标，即 x 向右偏移 1
    """
    dataset = np.array(dataset)  # 确保是 numpy array
    num_possible_starts = len(dataset) - context_length
    if num_possible_starts <= 0:
        raise ValueError("Dataset too small for the given context length")

    # 随机采样起始索引
    start_indices = np.random.randint(0, num_possible_starts, size=batch_size)
    
    # 构建 x 和 y
    x_batch = np.array([dataset[i:i+context_length] for i in start_indices])
    y_batch = np.array([dataset[i+1:i+context_length+1] for i in start_indices])
    
    # 转为 PyTorch tensor，并移动到指定 device
    x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_batch, dtype=torch.float32, device=device)
    
    return x_tensor, y_tensor


def run_softmax(in_features, dim):
    x = in_features - in_features.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)



def run_cross_entropy(inputs, targets):
    log_probs = torch.log_softmax(inputs, dim=-1)
    return -log_probs[torch.arange(targets.shape[0]), targets].mean()



def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # 过滤没有梯度的参数
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum((g**2).sum() for g in grads))
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)




def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    return get_lr_cosine_schedule(
        it=it,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )


def run_save_checkpoint(*args, **kwargs):
    return save_checkpoint(*args, **kwargs)


def run_load_checkpoint(*args, **kwargs):
    return load_checkpoint(*args, **kwargs)



def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    # 直接调用你写的工厂函数
    return get_bpe_tokenizer(vocab, merges, special_tokens)


def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    # 调用你刚才写的逻辑
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    return vocab, merges