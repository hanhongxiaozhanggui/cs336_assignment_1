# prepare_data.py
import os
import numpy as np
import json
from cs336_basics.tokenizer import train_bpe, BPETokenizer

# --- 配置 ---
DATA_DIR = "data"
OUT_DIR = "data_bin"
VOCAB_SIZE = 32768
SPECIAL_TOKENS = ["<|endoftext|>"]

def prepare():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. 训练词表
    # 注意：我们只用 TinyStories 的前 100MB 训练，否则太慢了
    print("正在抽取样本训练词表...")
    sample_text_path = os.path.join(DATA_DIR, "train_sample.txt")
    with open(os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt"), "r", encoding="utf-8") as f:
        sample_content = f.read(5 * 1024 * 1024) # 读取 100MB,先改成5
    with open(sample_text_path, "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    print("开始训练 BPE...")
    vocab, merges = train_bpe(sample_text_path, VOCAB_SIZE, SPECIAL_TOKENS)
    tokenizer = BPETokenizer(vocab, merges, SPECIAL_TOKENS)
    tokenizer.save(os.path.join(OUT_DIR, "vocab.json"), os.path.join(OUT_DIR, "merges.txt"))
    
    # 2. 转换所有 txt 文件为 bin
    files = [
        "TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt",
        "owt_train.txt", "owt_valid.txt"
    ]
    
    for filename in files:
        in_path = os.path.join(DATA_DIR, filename)
        out_path = os.path.join(OUT_DIR, filename.replace(".txt", ".bin"))
        
        if not os.path.exists(in_path):
            continue
            
        total_size = os.path.getsize(in_path)
        processed_bytes = 0
        
        print(f"\n正在转换 {filename} (约 {total_size / 1e6:.2f} MB)...")
        
        with open(in_path, "r", encoding="utf-8") as f, open(out_path, "wb") as f_bin:
            # 优化：逐行读取或分小块读取
            for line in f:
                if not line.strip(): continue # 跳过空行
                
                # 编码
                ids = tokenizer.encode(line)
                if not ids: continue
                
                # 写入
                ids_array = np.array(ids, dtype=np.uint16)
                f_bin.write(ids_array.tobytes())
                
                # 更新进度 (按粗略字符数算，快很多)
                processed_bytes += len(line.encode("utf-8"))
                
                # 频率控制：不要每行都打印，否则打印本身就变成了性能瓶颈
                # 我们可以每处理 1MB 打印一次
                if processed_bytes % (1024 * 1024) < 1000: 
                    progress = (processed_bytes / total_size) * 100
                    print(f"\r进度: {progress:6.2f}% | 已处理: {processed_bytes/1e6:.1f}MB", end="")

        print(f"\n✅ 成功保存至 {out_path}")

if __name__ == "__main__":
    prepare()