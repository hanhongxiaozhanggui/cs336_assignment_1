import numpy as np

# 加载刚才生成的大文件
path = "data_bin/TinyStoriesV2-GPT4-train.bin"
data = np.memmap(path, dtype=np.uint16, mode='r')

print(f"总 Token 数: {len(data):,}")
print(f"前 20 个 Token ID: {data[:20]}")

# 检查是否有超出 vocab_size (32768) 的无效 ID
max_id = data.max()
print(f"最大 Token ID: {max_id}")
if max_id >= 32768:
    print("警告：发现超出词表范围的 ID！")
else:
    print("数据合法性检查通过！")
    