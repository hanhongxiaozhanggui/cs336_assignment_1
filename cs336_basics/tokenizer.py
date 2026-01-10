# tokrnizer.py
import regex as re
import regex as re  # 建议使用 regex 库以支持 \p{L}
import json
from collections import Counter, defaultdict

PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path, vocab_size, special_tokens):
    # 1. 读取数据并进行初步分词
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 使用 GPT-2 的正则表达式
    PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    words = PAT.findall(text)
    
    # 统计每个单词出现的频次 (单词以 bytes 的 tuple 形式存储)
    word_counts = Counter(tuple(w.encode("utf-8")) for w in words)
    
    # 2. 初始化词表 (0-255 字节) 和特殊 token
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")
        
    num_merges = vocab_size - len(vocab)
    merges = []

    # 3. 建立反向索引：Pair -> {包含了这个 pair 的单词们}
    # 建立计数器：Pair -> 出现的总次数
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            p = (word[i], word[i+1])
            pair_counts[p] += count
            pair_to_words[p].add(word)

    # 4. 核心训练循环
    print(f"开始执行 {num_merges} 次合并...")
    for i in range(num_merges):
        if not pair_counts:
            break
            
        # 寻找频率最高的对 (如果有多个，按字典序排序保证确定性)
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        
        # 记录合并规则
        merges.append(best_pair)
        new_token_id = 256 + len(special_tokens) + i
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        # --- 修复 KeyError 的核心逻辑 ---
        # 我们必须先将受影响的单词取出来，并复制一份 list，避免在迭代中修改 set
        affected_words = list(pair_to_words[best_pair])
        
        for word in affected_words:
            if word not in word_counts:
                continue
                
            count = word_counts[word]
            
            # (A) 从索引中彻底移除该单词旧的所有 pair 计数
            # 使用 .discard() 替代 .remove() 以防万一
            for j in range(len(word) - 1):
                p = (word[j], word[j+1])
                pair_counts[p] -= count
                pair_to_words[p].discard(word)
                if pair_counts[p] <= 0:
                    del pair_counts[p]

            # (B) 执行合并生成新词
            new_word = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and (word[j], word[j+1]) == best_pair:
                    new_word.append(new_token_id)
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            new_word = tuple(new_word)

            # (C) 更新全局统计
            del word_counts[word]
            word_counts[new_word] = count

            # (D) 将合并后的新词及其产生的 pair 重新加入索引
            for j in range(len(new_word) - 1):
                p = (new_word[j], new_word[j+1])
                pair_counts[p] += count
                pair_to_words[p].add(new_word)

        if (i + 1) % 500 == 0:
            print(f"进度: {i+1}/{num_merges} 合并已完成")

    return vocab, merges


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab  # id -> bytes
        self.special_tokens_list = special_tokens or []
        self.byte_to_id = {v: k for k, v in vocab.items()}
        
        # 核心：merges 存储 (id, id)，方便 encode 时快速查找
        self.ranks = {}
        for i, pair in enumerate(merges):
            # 兼容处理：确保存入 ranks 的是 (int, int)
            p0 = pair[0][0] if isinstance(pair[0], bytes) else pair[0]
            p1 = pair[1][0] if isinstance(pair[1], bytes) else pair[1]
            self.ranks[(p0, p1)] = i
            
        self.special_re = None
        if self.special_tokens_list:
            re_str = "|".join(re.escape(st) for st in self.special_tokens_list)
            self.special_re = re.compile(f"({re_str})")
        self.norm_pat = re.compile(PAT)

    def _encode_chunk(self, text):
        # 初始字节转 ID
        word = list(text.encode("utf-8"))
        while len(word) >= 2:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            # 找到最先合并的那一对
            best_pair = min(pairs, key=lambda p: self.ranks.get(p, float('inf')))
            if best_pair not in self.ranks:
                break
            
            new_id = 256 + len(self.special_tokens_list) + self.ranks[best_pair]
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == best_pair:
                    new_word.append(new_id); i += 2
                else:
                    new_word.append(word[i]); i += 1
            word = new_word
        return word

    def encode(self, text):
        if not text: return []
        if not self.special_re:
            parts = [text]
        else:
            parts = [p for p in self.special_re.split(text) if p]

        ids = []
        for part in parts:
            if part in self.special_tokens_list:
                ids.append(self.byte_to_id[part.encode("utf-8")])
            else:
                for match in self.norm_pat.finditer(part):
                    ids.extend(self._encode_chunk(match.group()))
        return ids

    def decode(self, ids: list[int]) -> str:
        # 将所有的 ID 转换回对应的 bytes
        tokens_bytes = []
        for idx in ids:
            if idx in self.vocab:
                tokens_bytes.append(self.vocab[idx])
            else:
                # 如果遇到了词表中没有的 ID（虽然正常流程不应该发生）
                continue
        
        # 将字节流拼接并解码为字符串
        # errors="replace" 可以防止因为字节切分不完整导致的解码崩溃
        return b"".join(tokens_bytes).decode("utf-8", errors="replace")

    def save(self, vocab_path, merges_path):
        # 转换 bytes 为 list 以便存 JSON
        serial_vocab = {k: list(v) for k, v in self.vocab.items()}
        with open(vocab_path, "w") as f:
            json.dump(serial_vocab, f, indent=4)
        
        # 存 merges 时直接存数字对，简单可靠
        with open(merges_path, "w") as f:
            # 我们直接根据存好的 ranks 还原顺序写入
            sorted_merges = sorted(self.ranks.items(), key=lambda x: x[1])
            for (p0, p1), _ in sorted_merges:
                f.write(f"{p0} {p1}\n")

    @classmethod
    def load(cls, vocab_path, merges_path, special_tokens=None):
        with open(vocab_path, "r") as f:
            data = json.load(f)
            vocab = {int(k): bytes(v) for k, v in data.items()}
        
        merges = []
        with open(merges_path, "r") as f:
            for line in f:
                if line.strip():
                    p0, p1 = map(int, line.split())
                    merges.append((p0, p1))
        return cls(vocab, merges, special_tokens)


    
def get_tokenizer(vocab, merges, special_tokens=None):
    return BPETokenizer(vocab, merges, special_tokens)
