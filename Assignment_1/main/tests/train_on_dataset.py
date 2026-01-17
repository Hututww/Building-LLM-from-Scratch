from BPE_trainer import train_bpe
import pickle

INPUT_PATH = "tests/TinyStories/TinyStories-train.txt"
VOCAB_SIZE = 10000               
SPECIAL_TOKENS = ["<|endoftext|>"] 

print("开始训练 BPE 分词器...")

vocab, merges = train_bpe(
    input_path=INPUT_PATH,
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS
)

print("nice 训练完成")

# 保存结果
with open("vocab_tinystories_10k.pkl", "wb") as f:
    pickle.dump(vocab, f)
print("词汇表已保存至: vocab_tinystories_10k.pkl")

with open("merges_tinystories_10k.txt", "w", encoding="utf-8") as f:
    for b1, b2 in merges:
        f.write(b1.hex() + " " + b2.hex() + "\n")
print("合并序列已保存至: merges_tinystories_10k.txt")

# 计算统计指标
longest_tok = max(vocab.values(), key=len)
shortest_tok = min(vocab.values(), key=len)
all_lengths = [len(v) for v in vocab.values()]
avg_length = sum(all_lengths) / len(vocab)

# 打印指标 (采用常规对齐打印风格)
print("\n" + "=" * 50)
print("BPE 训练统计报告")
print("-" * 50)
print("词汇表总数:    ", len(vocab))
print("合并步数:      ", len(merges))
print("最大字节长度:  ", len(longest_tok))
print("最长 Token:    ", longest_tok)
print("最小字节长度:  ", len(shortest_tok))
print("平均字节长度:  ", round(avg_length, 2))
print("=" * 50)