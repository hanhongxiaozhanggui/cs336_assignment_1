import numpy as np
import pickle
from tokenizer import train_bpe, BPETokenizer

# 1. 训练 BPE (这一步可能需要几分钟)
print("正在训练 BPE...")
special_tokens = ["<|endoftext|>"]
vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", vocab_size=10000, special_tokens=special_tokens)

# 2. 保存 Tokenizer 结果，防止下次还得重练
with open("tokenizer_data.pkl", "wb") as f:
    pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": special_tokens}, f)

# 3. 初始化 Tokenizer 并预处理数据
tokenizer = BPETokenizer(vocab, merges, special_tokens)
print("正在编码文本...")
with open("data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
    text = f.read()

ids = tokenizer.encode(text)
ids_array = np.array(ids, dtype=np.uint16)
ids_array.tofile("data/train.bin")
print(f"完成！数据已保存到 data/train.bin，共 {len(ids_array)} 个 token。")