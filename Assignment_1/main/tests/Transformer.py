import torch
import torch.nn as nn
import math

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        负责给执行线性变换的Linear类进行初始化
        input:
            in_features: 输入的最终维度
            out_features: 输出的最终维度
            device: 储存参数的设备
            dtype: 参数的数据类型
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        container_tensor = torch.empty((in_features, out_features), device=device, dtype=dtype)
        self.weight = nn.Parameter(container_tensor)

        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight, # 将生成的随机数放入weight矩阵
            mean = 0.0, # 中心值
            std = sigma,
            a = -3.0 * sigma,
            b = 3.0 * sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)
    
class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        实现tokenId到特征向量这一转化(的初始化)
        input:
            num_embeddings: 词汇表大小
            embedding_dim: 嵌入向量的维度 即d_model
            device: 存储参数的设备
            dtype: 参数的数据类型
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=1.0, 
            a=-3.0, 
            b=3.0
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        均方层归一化的初始化
        """
        return