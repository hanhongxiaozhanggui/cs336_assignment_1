import torch
import numpy as np
import os
from cs336_basics.model import GPT, GPTConfig  # 假设你的模型定义在这里
from cs336_basics.optimizer import AdamW      # 假设你的优化器定义在这里

# --- 1. 配置超参数 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32      # 根据显存调整
block_size = 256     # 上下文长度
max_iters = 5000     # 迭代次数
lr = 6e-4            # 学习率
eval_interval = 500  # 每隔多少步验证一次

# --- 2. 数据读取函数 ---
def get_batch(split):
    filename = f"data_bin/TinyStoriesV2-GPT4-{split}.bin"
    # 使用 memmap 避免将整个 3.5G 文件载入内存
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    # 随机选择起始索引
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    return x.to(device), y.to(device)

# --- 3. 初始化模型和优化器 ---
config = GPTConfig(vocab_size=32768, block_size=block_size)
model = GPT(config).to(device)

# 手动 CrossEntropy 实现 (作业要求)
def get_loss(logits, targets):
    # logits: (B, T, V), targets: (B, T)
    return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1)

# --- 4. 主训练循环 ---
print(f"正在 {device} 上启动训练...")
for iter in range(max_iters):
    # 获取数据
    xb, yb = get_batch('train')
    
    # 前向传播
    logits = model(xb)
    loss = get_loss(logits, yb)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪 (防止梯度爆炸)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    if iter % 100 == 0:
        print(f"Step {iter}: Loss = {loss.item():.4f}")

# 保存模型
torch.save(model.state_dict(), "checkpoint.pt")
print("训练完成，模型已保存。")