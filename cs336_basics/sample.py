import torch
import torch.nn.functional as F
from cs336_basics.model import GPT, GPTConfig
from cs336_basics.tokenizer import BPETokenizer
import json
import os

# --- 1. 配置与加载 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoint.pt"  # 确保你 train.py 里保存了这个文件
vocab_path = "data_bin/vocab.json"
merges_path = "data_bin/merges.txt"

def load_tokenizer():
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)
        # JSON 的 key 是字符串，需要转回 int
        vocab = {int(k): bytes(v) for k, v in vocab_raw.items()}
    
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p0, p1 = map(int, line.split())
                merges.append((p0, p1))
    
    # 注意：这里的 special_tokens 要和 prepare_data.py 保持一致
    return BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

# --- 2. 核心采样逻辑 ---
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=50):
    """
    temperature: 越高越随机(有创意)，越低越死板(保险)
    top_k: 只从概率最高的前 K 个词里选，防止出现胡言乱语
    """
    model.eval()
    for _ in range(max_new_tokens):
        # 裁剪上下文长度
        idx_cond = idx if idx.size(1) <= 256 else idx[:, -256:]
        
        # 前向传播
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        # Top-K 过滤
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        # 计算概率并采样
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 拼接新词
        idx = torch.cat((idx, idx_next), dim=1)
        
        # 如果生成了停止符，提前结束 (可选)
        # if idx_next.item() == tokenizer.special_token_id: break
        
    return idx

# --- 3. 运行生成 ---
if __name__ == "__main__":
    tokenizer = load_tokenizer()
    
    # 初始化模型架构并加载权重
    config = GPTConfig(vocab_size=32768, block_size=256)
    model = GPT(config).to(device)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"成功加载权重: {checkpoint_path}")
    else:
        print("未找到权重文件，将使用随机初始化的模型进行测试...")

    # 设置你的开头
    prompt = "Once upon a time, there was a little bird who"
    print(f"\n输入提示: {prompt}")
    
    # 编码输入
    start_ids = tokenizer.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # 生成输出
    y = generate(model, x, max_new_tokens=100, temperature=0.8)
    
    # 解码并打印
    output_text = tokenizer.decode(y[0].tolist())
    print("-" * 30)
    print(f"生成的文本:\n{output_text}")
    print("-" * 30)