# 260102 搞定了前两部分
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
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        if special_tokens is not None:
            self.special_tokens = special_tokens.copy()
        else: self.special_tokens = []

        # 不想遍历键值对找 利用字典哈希特性做一个反映射利于编码查找
        self.byte_to_id = {i : j for j , i in self.vocab.items()}
        self.special_byte_to_id = {}

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

    def encode(self, text: str) -> list[int]:
        """
        将输入文本编码为token ID序列
        参数：
            text: str：输入文本
        返回：
            list[int]：token ID序列
        """
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对可迭代字符串（如Python文件句柄）进行惰性编码，生成token ID
        适用于无法加载到内存的大型文件
        参数：
            iterable: Iterable[str]：可迭代字符串
        返回：
            Iterator[int]：token ID迭代器
        """
        pass

    def decode(self, ids: list[int]) -> str:
        """
        将token ID序列解码为文本
        参数：
            ids: list[int]：token ID序列
        返回：
            str：解码后的文本
        """
        pass