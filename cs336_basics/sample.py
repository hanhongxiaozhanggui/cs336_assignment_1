import torch
import torch.nn.functional as F
from cs336_basics.model import GPT, GPTConfig
from cs336_basics.tokenizer import BPETokenizer

# --- 1. 配置 ---
device = "cuda"
checkpoint_path = "checkpoint_optimized.pt"
vocab_path = "data_bin/vocab.json"
merges_path = "data_bin/merges.txt"

# 保持与训练一致
config = GPTConfig(
    vocab_size=32769, 
    block_size=256,
    n_layer=8,
    n_head=8,
    n_embd=512,
)

# --- 2. 加载 ---
# 确保加载时处理了 special_tokens，防止 ID 偏移
tokenizer = BPETokenizer.load(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
model = GPT(config).to(device)

# 加载权重并清理前缀
state_dict = torch.load(checkpoint_path, map_location=device)
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

print(f"模型与词表加载成功！准备开始创作...")

# --- 3. 核心生成函数 ---
def generate_long_story(prompt, min_tokens=150, max_tokens=450, temperature=0.9):
    """
    min_tokens: 强迫模型至少写这么长
    max_tokens: 封顶长度
    temperature: 0.9 增加一点文学色彩
    """
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # 记录已经生成的 ID，用于去重惩罚
    prompt_len = len(input_ids)
    EOT_ID = 32768

    for i in range(max_tokens):
        # 裁剪窗口
        x_cond = x if x.size(1) <= config.block_size else x[:, -config.block_size:]
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, _ = model(x_cond)
        
        # 1. 提取最后一个 token 的分布
        logits = logits[0, -1, :] / temperature
        
        # 2. 强力重复惩罚 (解决 "The bird... The bird..." 问题)
        # 对已经在序列中出现的 token 施加指数级概率衰减
        tokens_seen = set(x[0].tolist())
        for tid in tokens_seen:
            if logits[tid] > 0:
                logits[tid] /= 1.3  # 惩罚系数 1.3
            else:
                logits[tid] *= 1.3

        # 3. Top-K 过滤，保留前 50 个最可能的词
        v, _ = torch.topk(logits, min(50, logits.size(-1)))
        logits[logits < v[-1]] = -float('Inf')
        
        # 4. 采样
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        
        # 5. 拼接
        x = torch.cat((x, next_id.view(1, 1)), dim=1)
        
        # 6. 智能停止逻辑
        if next_id.item() == EOT_ID:
            # 如果还没写够 min_tokens，就忽略这个结束符，继续编
            if i < min_tokens:
                continue 
            else:
                break
                
    # 7. 解码与后处理
    full_text = tokenizer.decode(x[0].tolist())
    
    # 清理：只取到第一个结束符（如果有）
    if "<|endoftext|>" in full_text[len(prompt):]:
        # 找到 prompt 之后的第一个 EOT
        main_story = full_text.split("<|endoftext|>")[0]
    else:
        main_story = full_text
        
    return main_story.strip()

# --- 4. 运行 ---
if __name__ == "__main__":
    test_prompt = "Once upon a time, there was a little bird who"
    
    print("\n" + "="*50)
    print(f"设定 Prompt: {test_prompt}")
    print("正在生成长篇故事...\n")
    
    story = generate_long_story(test_prompt)
    
    print(story)
    print("\n" + "="*50)