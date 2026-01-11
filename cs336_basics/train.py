import os
import time
import math
import torch
import numpy as np
from cs336_basics.model import GPT, GPTConfig

# --- 1. 超参数配置 ---
batch_size = 64
block_size = 256
max_iters = 10000  # 建议增加到 10000，4090 跑得很快
learning_rate = 6e-4
device = "cuda"
dtype = 'bfloat16'

# --- 2. 模型配置 ---
config = GPTConfig(
    vocab_size = 32769, # 必须是 32769 以匹配 EOT_ID 32768
    block_size = block_size,
    n_layer = 8,
    n_head = 8,
    n_embd = 512,
)

# --- 3. 数据加载优化 ---
data_dir = "data_bin"
def get_batch(split):
    # 修改：匹配刚才 prepare_data.py 生成的文件名
    s = "train" if split == "train" else "valid"
    filename = os.path.join(data_dir, f"TinyStoriesV2-GPT4-{s}.bin")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"找不到数据文件: {filename}，请检查 data_bin 目录")
        
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y

# --- 4. 初始化 ---
print(f"🚀 正在 {device} 上启动 8 层 GPT 训练...")
model = GPT(config).to(device)

if hasattr(torch, 'compile'):
    print("正在进行 torch.compile 静态编译 (这可能需要 1-2 分钟)...")
    model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-1)

def get_lr(it):
    if it < 200: return learning_rate * it / 200 # 稍微延长预热
    decay_ratio = (it - 200) / (max_iters - 200)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return 1e-5 + coeff * (learning_rate - 1e-5)

# --- 5. 训练循环 ---
start_time = time.time()

for it in range(max_iters):
    lr = get_lr(it)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    xb, yb = get_batch('train')

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if it % 100 == 0:
        t1 = time.time()
        # 增加一个简单的评估逻辑，看看验证集 Loss
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch('val')
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, v_loss = model(xv, yv)
        model.train()
        
        print(f"Step {it:4d}: Train Loss = {loss.item():.4f} | Val Loss = {v_loss.item():.4f} | LR = {lr:.2e} | Time = {t1-start_time:.2f}s")
        start_time = t1

# --- 6. 优雅保存 ---
# 移除 torch.compile 带来的前缀
raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
torch.save(raw_model.state_dict(), "checkpoint_optimized.pt")
print("\n✅ 训练完成！模型已清理前缀并保存。")