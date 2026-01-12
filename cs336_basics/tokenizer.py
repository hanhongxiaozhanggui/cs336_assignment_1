# cs336_basics/tokenizer.py
import regex as re
from collections import Counter
from typing import List, Tuple, Dict, Iterable
from functools import lru_cache
from collections import Counter, defaultdict
import heapq

# GPT-2 风格正则（CS336 标准）
PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# ============================================================
# 1. BPE 训练
# ============================================================
def train_bpe(input_path, vocab_size, special_tokens):
    # 1. 基础准备
    vocab = {i: bytes([i]) for i in range(256)}
    for i, st in enumerate(special_tokens):
        vocab[256 + i] = st.encode("utf-8")
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. 初始分词 (使用你之前的 PAT)
    word_counts = Counter(re.findall(PAT, text))
    
    # 将单词表示为 token 列表，并记录频率
    # words: [[id, id, id], ...]
    # counts: [freq, freq, ...]
    words = []
    counts = []
    for word, freq in word_counts.items():
        words.append([bytes([b]) for b in word.encode("utf-8")])
        counts.append(freq)

    # 3. 建立索引：pair -> set of word_indices
    # 记录每个 pair 出现在哪些单词里
    pair_to_word_indices = defaultdict(set)
    pair_freqs = Counter()

    for i, word in enumerate(words):
        for j in range(len(word) - 1):
            pair = (word[j], word[j + 1])
            pair_to_word_indices[pair].add(i)
            pair_freqs[pair] += counts[i]

    merges = []
    num_merges = vocab_size - 256 - len(special_tokens)

    # 4. 迭代合并
    for _ in range(num_merges):
        if not pair_freqs:
            break
        
        # 找到频率最高的 pair (可以通过 heapq 进一步优化，但 Counter.most_common(1) 已经很快了)
        best_pair, freq = pair_freqs.most_common(1)[0]
        if freq <= 0: break
        
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[256 + len(special_tokens) + len(merges) - 1] = new_token

        # 只处理包含这个 best_pair 的单词索引
        affected_indices = pair_to_word_indices[best_pair]
        
        # 在删除索引前清空 pair_freqs 里的旧数据
        del pair_freqs[best_pair]
        del pair_to_word_indices[best_pair]

        for idx in affected_indices:
            word = words[idx]
            count = counts[idx]
            
            # 在合并前，先移除该单词贡献的所有其他 pair 频率
            # 这一步是关键：保证计数准确
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                if p != best_pair: # best_pair 已经删过了
                    pair_freqs[p] -= count
                    pair_to_word_indices[p].discard(idx)

            # 执行合并逻辑
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            words[idx] = new_word
            
            # 合并后，重新把新单词的 pair 加入索引和频率统计
            word = words[idx]
            for i in range(len(word) - 1):
                p = (word[i], word[i+1])
                pair_freqs[p] += count
                pair_to_word_indices[p].add(idx)

    return vocab, merges



# ============================================================
# 2. Tokenizer 类
# ============================================================
class BPETokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.byte_to_id = {v: k for k, v in vocab.items()}

        self.word_pat = re.compile(PAT)

        if self.special_tokens:
            sp = sorted(self.special_tokens, key=len, reverse=True)
            self.split_pat = re.compile("(" + "|".join(re.escape(s) for s in sp) + ")")
        else:
            self.split_pat = None

    # 新增这个内部方法，专门处理单个单词并缓存结果
    @lru_cache(maxsize=10000)
    def _encode_word(self, token_bytes: bytes) -> List[int]:
        word = [bytes([b]) for b in token_bytes]
        for p0, p1 in self.merges:
            if len(word) <= 1: break
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word)-1 and word[i] == p0 and word[i+1] == p1:
                    new_word.append(p0 + p1)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return [self.byte_to_id[b] for b in word]

    # 修改你的 encode 方法
    def encode(self, text: str) -> List[int]:
        if not text: return []
        ids = []
        parts = self.split_pat.split(text) if self.split_pat else [text]
        for part in parts:
            if not part: continue
            if part in self.special_tokens:
                ids.append(self.byte_to_id[part.encode("utf-8")])
                continue
            for token in self.word_pat.findall(part):
                # 关键：调用带缓存的函数
                ids.extend(self._encode_word(token.encode("utf-8")))
        return ids

    def decode(self, ids: List[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, texts: Iterable[str]):
        for t in texts:
            for i in self.encode(t):
                yield i


# ============================================================
# 3. 工厂函数（tests 用）
# ============================================================
def get_tokenizer(vocab, merges, special_tokens=None):
    return BPETokenizer(vocab, merges, special_tokens)
