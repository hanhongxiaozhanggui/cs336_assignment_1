import os
import numpy as np
import json
from cs336_basics.tokenizer import train_bpe, BPETokenizer

DATA_DIR = "data"
OUT_DIR = "data_bin"
VOCAB_SIZE = 32768 
EOT_ID = 32768 

def prepare():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. 训练词表 - 保持 100MB 样本
    train_txt = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
    sample_text_path = os.path.join(DATA_DIR, "train_sample.txt")
    
    if not os.path.exists(sample_text_path):
        print("正在抽取 100MB 样本...")
        with open(train_txt, "r", encoding="utf-8") as f:
            sample_content = f.read(100 * 1024 * 1024)
        with open(sample_text_path, "w", encoding="utf-8") as f:
            f.write(sample_content)
    
    print("开始训练 BPE...")
    vocab, merges = train_bpe(sample_text_path, VOCAB_SIZE, ["<|endoftext|>"])
    tokenizer = BPETokenizer(vocab, merges, ["<|endoftext|>"])
    tokenizer.save(os.path.join(OUT_DIR, "vocab.json"), os.path.join(OUT_DIR, "merges.txt"))
    
    # 2. 修正后的 bin 转换逻辑
    for filename in ["TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"]:
        in_path = os.path.join(DATA_DIR, filename)
        out_path = os.path.join(OUT_DIR, filename.replace(".txt", ".bin"))
        if not os.path.exists(in_path): continue
            
        print(f"\n🚀 正在逻辑转换 {filename}...")
        all_ids = []
        MAX_MB = 200 if "train" in filename else 20 # 验证集不需要太大
        processed_bytes = 0

        with open(in_path, "r", encoding="utf-8") as f:
            # --- 关键修改：按行读取，严禁暴力切分单词 ---
            for line in f:
                if processed_bytes / 1024 / 1024 >= MAX_MB:
                    break
                
                text = line.strip()
                if not text: continue # 跳过空行
                
                # 只有完整的行编码，BPE 才能匹配到 "Once upon a time"
                ids = tokenizer.encode(text)
                ids.append(EOT_ID) 
                all_ids.extend(ids)
                
                processed_bytes += len(line.encode("utf-8"))
                if len(all_ids) % 5000 == 0:
                    print(f"\r进度: {processed_bytes / 1024 / 1024:.1f} / {MAX_MB} MB", end="")

        print(f"\n保存 {out_path}，总 Token 数: {len(all_ids)}")
        np.array(all_ids, dtype=np.uint16).tofile(out_path)

if __name__ == "__main__":
    prepare()