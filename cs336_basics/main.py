# main.py
from cs336_basics.tokenizer import train_bpe, get_tokenizer, save_vocab_merges, load_vocab_merges

def main():
    # --------------------------
    # 1️⃣ 配置参数
    # --------------------------
    corpus_file = "corpus.txt"      # 你的训练语料
    vocab_path = "vocab.json"
    merges_path = "merges.json"
    special_tokens = ["<PAD>", "<EOS>"]
    vocab_size = 1000               # 可根据作业要求调整

    # --------------------------
    # 2️⃣ 训练 BPE
    # --------------------------
    print("Training BPE tokenizer...")
    vocab, merges = train_bpe(corpus_file, vocab_size, special_tokens)
    print(f"Vocab size: {len(vocab)}, Merges: {len(merges)}")

    # --------------------------
    # 3️⃣ 保存 vocab & merges
    # --------------------------
    save_vocab_merges(vocab, merges, vocab_path, merges_path)
    print(f"Saved vocab to {vocab_path} and merges to {merges_path}")

    # --------------------------
    # 4️⃣ 加载 vocab & merges
    # --------------------------
    vocab, merges = load_vocab_merges(vocab_path, merges_path)
    tokenizer = get_tokenizer(vocab, merges, special_tokens)
    print("Tokenizer loaded!")

    # --------------------------
    # 5️⃣ 测试单条文本编码/解码
    # --------------------------
    text = "Hello world!"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(f"Original: {text}")
    print(f"Encoded IDs: {ids}")
    print(f"Decoded: {decoded}")

    # --------------------------
    # 6️⃣ 测试可迭代文本编码
    # --------------------------
    texts = ["This is the first line.", "And this is the second line!"]
    all_ids = list(tokenizer.encode_iterable(texts))
    print(f"Iterable encoding IDs: {all_ids}")

if __name__ == "__main__":
    main()
