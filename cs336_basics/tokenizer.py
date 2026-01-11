# cs336_basics/tokenizer.py
import regex as re
from collections import Counter
from typing import List, Tuple, Dict, Iterable

# GPT-2 风格正则（CS336 标准）
PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# ============================================================
# 1. BPE 训练
# ============================================================
def train_bpe(input_path, vocab_size, special_tokens):
    """
    Efficient BPE training that passes CS336 speed test.
    """
    import regex as re
    from collections import Counter, defaultdict

    # --- init vocab ---
    vocab = {i: bytes([i]) for i in range(256)}
    for i, st in enumerate(special_tokens):
        vocab[256 + i] = st.encode("utf-8")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # --- split on special tokens ---
    special_pattern = "|".join(re.escape(st) for st in special_tokens)
    parts = re.split(f"({special_pattern})", text)

    # --- build initial word counts ---
    word_counts = Counter()
    for part in parts:
        if not part or part in special_tokens:
            continue
        for word in re.findall(PAT, part):
            word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
            word_counts[word_bytes] += 1

    # --- build initial pair counts ---
    pair_counts = Counter()
    word_to_pairs = defaultdict(set)

    for word, freq in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            word_to_pairs[pair].add(word)

    merges = []
    num_merges = vocab_size - 256 - len(special_tokens)

    for _ in range(num_merges):
        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)

        new_token = best_pair[0] + best_pair[1]
        vocab[256 + len(special_tokens) + len(merges) - 1] = new_token

        affected_words = word_to_pairs[best_pair]
        word_to_pairs.pop(best_pair, None)
        pair_counts.pop(best_pair, None)

        new_word_counts = Counter()

        for word in affected_words:
            freq = word_counts[word]
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)

            new_word_counts[new_word] += freq
            word_counts.pop(word)

            # remove old pairs
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                word_to_pairs[pair].discard(word)

        # add new words and new pairs
        for word, freq in new_word_counts.items():
            word_counts[word] += freq
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq
                word_to_pairs[pair].add(word)

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

    def encode(self, text: str) -> List[int]:
        if not text:
            return []

        ids = []
        parts = self.split_pat.split(text) if self.split_pat else [text]

        for part in parts:
            if not part:
                continue

            if part in self.special_tokens:
                b = part.encode("utf-8")
                ids.append(self.byte_to_id[b])
                continue

            for token in self.word_pat.findall(part):
                word = [bytes([b]) for b in token.encode("utf-8")]

                for p0, p1 in self.merges:
                    i = 0
                    new_word = []
                    while i < len(word):
                        if i < len(word) - 1 and word[i] == p0 and word[i + 1] == p1:
                            new_word.append(p0 + p1)
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    word = new_word

                for b in word:
                    ids.append(self.byte_to_id[b])

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
