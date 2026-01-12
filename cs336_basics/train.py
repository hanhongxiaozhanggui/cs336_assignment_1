import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os

# 导入你的核心组件
from model import GPT, GPTConfig 
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from tests.adapters import run_gradient_clipping

# ==========================================
# 1. 超参数配置
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 注意：vocab_size 必须与你之前 BPE 训练的一致
config = GPTConfig(
    block_size=256,
    vocab_size=10000, 
    n_layer=4,
    n_head=4,
    n_embd=256
)

batch_size = 32
learning_rate = 5e-4
max_steps = 20000     # 训练总步数
warmup_steps = 500
weight_decay = 0.1

# ==========================================
# 2. 数据获取函数
# ==========================================
def get_batch(data_memmap, batch_size, block_size):
    # 随机选择起始位置
    ix = torch.randint(len(data_memmap) - block_size - 1, (batch_size,))
    # 必须转换为 int64，因为 Embedding 层不支持 uint16
    x = torch.stack([torch.from_numpy(data_memmap[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data_memmap[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# ==========================================
# 3. 训练主循环
# ==========================================
def train():
    # --- A. 数据映射 ---
    # 请确保 prepare_data.py 生成的文件在这个路径
    data_path = "data/train.bin"
    if not os.path.exists(data_path):
        print(f"❌ 错误: 找不到 {data_path}。请确认文件路径。")
        return
    
    # 使用 np.memmap 节省内存
    train_data = np.memmap(data_path, dtype=np.uint16, mode='r')
    print(f"✅ 数据加载成功，总 Token 数: {len(train_data)}")

    # --- B. 模型初始化 ---
    model = GPT(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🚀 GPT 初始化完成。参数量: {total_params / 1e6:.2f}M")

    # --- C. 优化器初始化 ---
    optimizer = AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )

    # --- D. 训练 ---
    model.train()
    pbar = tqdm(range(max_steps), desc="Training GPT")
    
    for step in pbar:
        # 1. 手动更新学习率 (适配你的 get_lr_cosine_schedule 签名)
        curr_lr = get_lr_cosine_schedule(
            it=step, 
            max_learning_rate=learning_rate, 
            min_learning_rate=learning_rate * 0.1, 
            warmup_iters=warmup_steps, 
            cosine_cycle_iters=max_steps
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr

        # 2. 获取数据
        x, y = get_batch(train_data, batch_size, config.block_size)
        
        # 3. 前向传播 (适配你的 GPT 类：同时返回 logits 和 loss)
        logits, loss = model(idx=x, targets=y) 
        
        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 5. 梯度裁剪 (适配你的函数签名: max_l2_norm)
        run_gradient_clipping(model.parameters(), max_l2_norm=1.0)
        
        # 6. 更新参数
        optimizer.step()
        
        # 7. 更新进度条状态
        if step % 10 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "lr": f"{curr_lr:.2e}"
            })

        # 8. 定期保存权重
        if step > 0 and step % 1000 == 0:
            torch.save(model.state_dict(), f"gpt_checkpoint_{step}.pt")

    # 最终保存
    torch.save(model.state_dict(), "gpt_final_2.pt")
    print("\n🎉 训练圆满完成！权重已保存至 gpt_final.pt")

if __name__ == "__main__":
    train()