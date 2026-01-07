import regex as re
from collections import Counter

# 严格按照作业要求的 PAT
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path, vocab_size, special_tokens):
    # 1. 初始化 Vocab
    vocab = {i: bytes([i]) for i in range(256)}
    for i, st in enumerate(special_tokens):
        vocab[256 + i] = st.encode("utf-8")
    
    # 2. 读取文本并进行精准预分词
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 将特殊字符视为不可分割的整体，但在训练 BPE 时排除它们
    # 这里我们只对非特殊字符部分进行 BPE 训练
    special_pattern = "|".join(re.escape(st) for st in special_tokens)
    
    # 真正的 BPE 训练只针对普通文本块
    # 改进：先按特殊字符分割，再对普通文本段应用 PAT
    parts = re.split(f"({special_pattern})", text)
    
    word_counts = Counter()
    for part in parts:
        if part in special_tokens or not part:
            continue
        # 对普通文本段进行 PAT 分词
        for word in re.findall(PAT, part):
            # 将每个单词转化为 bytes 的元组
            word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
            word_counts[word_bytes] += 1

    merges = []
    num_merges = vocab_size - 256 - len(special_tokens)
    
    # 性能优化：预先计算所有单词内部的 pair
    # 这样就不需要每一轮都重新扫描整个 word_counts
    for _ in range(num_merges):
        pair_counts = Counter()
        for word, count in word_counts.items():
            for i in range(len(word) - 1):
                pair_counts[word[i:i+2]] += count
        
        if not pair_counts:
            break
            
        # 严格执行：最高频 -> 字典序最大
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)
        
        new_token = best_pair[0] + best_pair[1]
        vocab[256 + len(special_tokens) + len(merges) - 1] = new_token
        
        # 极致更新逻辑：只处理包含 best_pair 的单词
        new_word_counts = Counter()
        for word, count in word_counts.items():
            if best_pair not in zip(word, word[1:]):
                new_word_counts[word] = count
                continue
            
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_counts[tuple(new_word)] = count
        word_counts = new_word_counts
        
    return vocab, merges