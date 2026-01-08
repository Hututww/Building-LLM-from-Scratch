import regex as re
from typing import Iterable
from BPE_tokenizer import Tokenizer, PAT

def train_bpe(input_path: str, 
              vocab_size: int, 
              special_tokens: list[str]
              ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练BPE分词器
    参数：
        input_path: 训练数据的文本文件路径
        vocab_size: 最终词汇表的最大规模(含初始字节、合并结果、特殊token)
        special_tokens: 需加入词汇表的特殊token(不参与训练)
    返回：
        vocab: 分词器词汇表(token ID → token字节)
        merges: BPE合并操作序列(按合并顺序排列)
    """
    tokenizer = Tokenizer(vocab = {}, merges = [], special_tokens = special_tokens)

    vocab = {}
    for token_byte, token_id in tokenizer.special_byte_to_id.items():
        vocab[token_id] = token_byte
    initial_place = max(vocab.keys(), default = -1) + 1
    for byte in range(256):
        vocab[initial_place + byte] = bytes([byte]) # 内层括号是防止生成依托空字节的
    
    initial_vocab_size = len(vocab)
    if initial_vocab_size >= vocab_size:
        return vocab, []
    
    max_merge_times = vocab_size - initial_vocab_size
    merges = [] 

    def _file_loader() -> Iterable[str]:
        """
        文件读取大王来也
        """
        with open(input_path, "r", encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                if not line :
                    continue
                yield line
    
    byte_seq = []
    for text in _file_loader():
        pre_tokens = tokenizer._pre_tokenize(text)
        for elem in pre_tokens:
            if elem in tokenizer.special_byte_to_id:
                continue
            single_byte_seq = [bytes([i]) for i in elem]
            byte_seq.append(single_byte_seq)

    if not byte_seq : # 注意not和is None的区别！
        return vocab, []
    
    # 合并时间
    for elem in range(max_merge_times):
        freq = {}
        for seq in byte_seq:
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                if pair in freq:
                    freq[pair] += 1
                else:
                    freq[pair] = 1
        if not freq: break

        best_pair = max(freq, key = freq.get)
        merges.append(best_pair)

        merged_byte = b"".join(best_pair)
        vocab[len(vocab)] = merged_byte

        new_seq = []
        for seq in byte_seq:
            _new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i+1]) == best_pair:
                    _new_seq.append(merged_byte)
                    i += 2
                else:
                    _new_seq.append(seq[i])
                    i += 1
            new_seq.append(_new_seq)
        byte_seq = new_seq

    return vocab, merges