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
                if not line :
                    continue
                yield line
    
    byte_seq = []
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    pre_tokens = tokenizer._pre_tokenize(text)
    for elem in pre_tokens:
        if elem in tokenizer.special_byte_to_id:
            continue
        single_byte_seq = [bytes([i]) for i in elem]
        byte_seq.append(single_byte_seq)

    if not byte_seq : # 注意not和is None的区别！
        return vocab, []
    
    record = {}
    for index, seq in enumerate(byte_seq):
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            record.setdefault(pair, set()).add((index, i))

    for elem in range(max_merge_times):
        if not record:
            break

        best_pair = None
        best_cnt = 0
        candidates = []
        for pair, pos_set in record.items():
            cnt = len(pos_set)
            if cnt == 0:
                continue
            if cnt > best_cnt:
                best_cnt = cnt
                candidates = [pair]
            elif cnt == best_cnt:
                candidates.append(pair)

        if best_cnt == 0:
            break

        best_pair = max(candidates)

        merges.append(best_pair)
        merged_byte = b"".join(best_pair)
        vocab[len(vocab)] = merged_byte

        affected = {pos[0] for pos in record[best_pair]}        
        for index in affected:
            old_seq = byte_seq[index]
            
            for i in range(len(old_seq) - 1):
                old_pair = (old_seq[i], old_seq[i+1])
                if old_pair in record:
                    record[old_pair].discard((index, i))
                    if not record[old_pair]:
                        del record[old_pair]

            new_seq = []
            i = 0
            while i < len(old_seq):
                if i < len(old_seq) - 1 and (old_seq[i], old_seq[i+1]) == best_pair:
                    new_seq.append(merged_byte)
                    i += 2
                else:
                    new_seq.append(old_seq[i])
                    i += 1
            byte_seq[index] = new_seq

            for i in range(len(new_seq) - 1):
                new_pair = (new_seq[i], new_seq[i+1])
                record.setdefault(new_pair, set()).add((index, i))

    return vocab, merges