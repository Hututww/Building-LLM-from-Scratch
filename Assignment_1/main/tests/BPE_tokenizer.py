import regex as re
import pickle # 神秘小模块，据说读取文件会方便
from typing import Iterable, Iterator

# 标准的pre-tokenize正则表达式 文件里给了的
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        构建分词器
        参数：
            vocab: dict[int, bytes]：词汇表
            merges: list[tuple[bytes, bytes]]：合并操作序列
            special_tokens: list[str] | None = None：特殊tokens列表（可选）
        """
        self.vocab = vocab
        self.merges = merges
        if special_tokens is not None:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else: self.special_tokens = []

        # 不想遍历键值对找 利用字典哈希特性做一个反映射利于编码查找
        self.byte_to_id = {i : j for j , i in vocab.items()}
        self.special_byte_to_id = {token.encode("utf-8"): self.byte_to_id.get(token.encode("utf-8")) for token in self.special_tokens}

        # 对于special tokens要有添加到vocab分配新id的逻辑
        new_id = max(self.vocab.keys(), default = -1) + 1
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.byte_to_id:
                self.vocab[new_id] = token_bytes
                self.byte_to_id[token_bytes] = new_id
                self.special_byte_to_id[token_bytes] = new_id
                new_id += 1

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        从序列化文件构建分词器
        参数：
            vocab_filepath: str：词汇表文件路径
            merges_filepath: str：合并操作序列文件路径
            special_tokens: list[str] | None = None：特殊tokens列表（可选）
        返回：
            Tokenizer：分词器实例
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f) # 序列化
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f: 
            for line in f:
                if not line:
                    continue
                line = line.strip() # 去掉换行符、tab和多的每行开头结尾空格
                parts = line.split()
                hex1 = parts[0]
                hex2 = parts[1]
                b1 = bytes.fromhex(hex1)
                b2 = bytes.fromhex(hex2)
                merges.append((b1, b2))
        return cls(vocab, merges, special_tokens)

    # 为了在encode里面方便写东西，得加俩函数，_pre_tokenize()做预分词，_apply_merge()做合并
    def _pre_tokenize(self, text: str) -> list[bytes]:
        """
        encode()内部使用的预分词函数
        参数：
            text: str: 文本here
        返回：
            pre_tokens: 分好词的字节序列列表
        """
        if not self.special_tokens:
            return [match.group(0).encode("utf-8") for match in re.finditer(PAT, text)] # 按照正则的方式把text里面的句子切成子字串并且转换成了字节

        special_part = "|".join(re.escape(token) for token in self.special_tokens)
        split_them_all = re.split(f"({special_part})", text)
        pre_tokens = []

        for elem in split_them_all:
            if not elem:
                continue
            if elem in self.special_tokens:
                pre_tokens.append(elem.encode("utf-8"))
            else:
                pre_tokens.extend([pair.group(0).encode("utf-8") for pair in re.finditer(PAT, elem)])
        return pre_tokens

    def _apply_merges(self, byte_seq: list[bytes]) -> list[bytes]:
        """
        依旧encode()内部的合并函数
        参数：
            byte_seq: preTokenized之后逐字拆开的的单字节的序列列表
        返回：
            merged_group: 合并过后的单+多字节序列列表
        """
        merged_group = byte_seq.copy()
        for elem in self.merges:
            if len(merged_group) < 2: break
            i = 0
            while i < len(merged_group) - 1:
                pair = (merged_group[i], merged_group[i+1])
                if pair == elem:
                    merged = b"".join(pair) 
                    merged_group = merged_group[:i] + [merged] + merged_group[i+2:]
                else: 
                    i += 1
        return merged_group

    def encode(self, text: str) -> list[int]:
        """
        将输入文本编码为token ID序列
        参数：
            text: str：输入文本
        返回：
            token_ids: list[int]：token ID序列
        """
        # step one: pretokenize=)
        pre_tokens = self._pre_tokenize(text)

        token_ids = []
        for elem in pre_tokens:
            # step two: special token
            if elem in self.special_byte_to_id:
                token_ids.append(self.special_byte_to_id[elem])
                continue

            # step three: normal token
            byte_seq = [bytes([i]) for i in elem]

            # step four：apply merges
            merged_seq = self._apply_merges(byte_seq)

            # step five: 反向map
            for j in merged_seq: token_ids.append(self.byte_to_id[j])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对可迭代字符串（如Python文件句柄）进行惰性编码，生成token ID
        适用于无法加载到内存的大型文件
        参数：
            iterable: Iterable[str]：可迭代字符串
        返回：
            Iterator[int]：token ID迭代器
        """
        for text in iterable: yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        将token ID序列解码为文本
        参数：
            ids: list[int]：token ID序列
        返回：
            str：解码后的文本
        """
        # 先把id变字节
        byte_seq =  []
        for token_id in ids:
            byte_seq.append(self.vocab[token_id])
        
        decoded_bytes = b"".join(byte_seq)
        decoded_text = decoded_bytes.decode("utf-8", errors="replace")
        return decoded_text
    

    """
    补充: self.vocab大概长这个样子:
        self.vocab = {
            0: b'<|endoftext|>',
            1: b'Hachimi',
            2: b'!',
            3: b' ',
            4: b'哦啊啊'
            }

    """