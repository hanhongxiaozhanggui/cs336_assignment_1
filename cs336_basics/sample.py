import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from tokenizer import BPETokenizer
import pickle

# ==========================================
# 1. 配置与加载
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "gpt_final_2.pt"
tokenizer_pkl = "tokenizer_data.pkl" # 你之前保存 tokenizer 的文件

# 加载 Tokenizer
with open(tokenizer_pkl, "rb") as f:
    tokenizer_data = pickle.load(f)
tokenizer = BPETokenizer(
    tokenizer_data["vocab"], 
    tokenizer_data["merges"], 
    tokenizer_data["special_tokens"]
)

# 初始化模型并加载权重
config = GPTConfig(
    block_size=256,
    vocab_size=10000, 
    n_layer=4,
    n_head=4,
    n_embd=256
)
model = GPT(config).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ==========================================
# 2. 生成逻辑 (Top-k Sampling)
# ==========================================
def generate(prompt, max_new_tokens=100, temperature=0.8, top_k=10):
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        # 如果长度超过 block_size，需要裁剪
        idx_cond = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size:]
        
        # 推理
        logits, _ = model(idx_cond)
        # 只取最后一个 time step 的 logits，并应用温度
        logits = logits[:, -1, :] / temperature
        
        # Top-k 过滤
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        # 采样
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 拼接到序列中
        idx = torch.cat((idx, idx_next), dim=1)
        
        # 如果抽到了停止符，提前结束 (假设 <|endoftext|> 的 ID 是某个值)
        # if idx_next.item() == tokenizer.byte_to_id.get(b'<|endoftext|>'): break

    return tokenizer.decode(idx[0].tolist())

# ==========================================
# 3. 测试生成
# ==========================================
prompt = "Once upon a time, there was a little dog named"
print(f"Prompt: {prompt}\n" + "-"*30)
generated_text = generate(prompt, max_new_tokens=150)
print(generated_text)