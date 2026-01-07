import regex as re
from collections import Counter
import regex as re  # 建议使用 regex 库以支持 \p{L}

PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens_list = special_tokens or []
        self.byte_to_id = {v: k for k, v in vocab.items()}
        
        # 构造一个专门匹配特殊 Token 的正则，长者优先
        if self.special_tokens_list:
            sorted_specials = sorted(self.special_tokens_list, key=len, reverse=True)
            special_re = "|".join(re.escape(st) for st in sorted_specials)
            # 使用括号捕获，这样 re.split 会保留分隔符
            self.split_pat = re.compile(f"({special_re})")
        else:
            self.split_pat = None
            
        # 预编译普通文本的 PAT
        self.norm_pat = re.compile(PAT)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
            
        ids = []
        
        # 第一步：按特殊 Token 切分文本
        if self.split_pat:
            # re.split 会保留被括号捕获的特殊字符
            parts = self.split_pat.split(text)
        else:
            parts = [text]
            
        for part in parts:
            if not part:
                continue
                
            # 第二步：检查当前部分是否是特殊 Token
            if part in self.special_tokens_list:
                p_bytes = part.encode("utf-8")
                if p_bytes in self.byte_to_id:
                    ids.append(self.byte_to_id[p_bytes])
                continue
            
            # 第三步：普通文本部分，按照 PAT 进一步切分
            for sub_text in self.norm_pat.findall(part):
                # 将子段转为字节序列进行 BPE 合并
                word = [bytes([b]) for b in sub_text.encode("utf-8")]
                
                # BPE 合并逻辑
                for p0, p1 in self.merges:
                    new_word = []
                    i = 0
                    while i < len(word):
                        if i < len(word) - 1 and word[i] == p0 and word[i+1] == p1:
                            new_word.append(p0 + p1)
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    word = new_word
                
                # 查表转为 ID
                for token_bytes in word:
                    if token_bytes in self.byte_to_id:
                        ids.append(self.byte_to_id[token_bytes])
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens_bytes = [self.vocab[idx] for idx in ids]
        return b"".join(tokens_bytes).decode("utf-8", errors="replace")

    def encode_iterable(self, text_iterable):
        for text in text_iterable:
            for idx in self.encode(text):
                yield idx

    
def get_tokenizer(vocab, merges, special_tokens=None):
    return BPETokenizer(vocab, merges, special_tokens)