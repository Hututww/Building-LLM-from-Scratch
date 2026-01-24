import torch
import torch.nn as nn
import math

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_val)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)

    return exp_x / sum_exp

class Linear(nn.Module):
    "负责维度线性变换"
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
    "tokenId变特征向量的编码过程"
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
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        nn.init.ones_(self.weight) # 把所有东西变1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_type = x.dtype
        x_32 = x.to(torch.float32)

        mean_sqr = x_32.pow(2).mean(dim = -1, keepdim=True) # 对特征向量（横的那个）求均方根
        """
        t1 = [2.0, 4.0, 6.0] 
        t2 = [1.0, 1.0, 1.0]

        mean_sqr = [[[18.67], [1.0]]]
        """
        x_final = x_32 * torch.rsqrt(self.eps + mean_sqr) # eps作用来了

        return (x_final * self.weight).to(original_type)

class SwiGLU(nn.Module):
    "SwiGLU前馈网络"
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()

        if d_ff is None:
            d_ff = int(8.0 * d_model / 3)
            d_ff = (d_ff + 63) // 64 * 64

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor):
        # w2(SiLU(w1x) * w3x)
        gate = self.w1(x)
        gate_activated = torch.nn.functional.silu(gate)

        info = self.w3(x)

        return self.w2(gate_activated * info)
    
class RotaryPositionalEmbedding(nn.Module):
    "旋转位置编码"
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        theta: RoPE的θ
        d_k: Query/Key 的维度
        max_seq_len: 输入的最大序列长度
        """
        super().__init__()
        self.d_k = d_k

        indices = torch.arange(0, d_k, 2).float().to(device)
        freq = 1.0 / (theta ** (indices / d_k))

        position = torch.arange(max_seq_len).to(device)
        angles = torch.outer(position, freq)

        cos = angles.cos().repeat_interleave(2, dim=-1) # dim = -1代表每一行单独操作
        sin = angles.sin().repeat_interleave(2, dim=-1)

        self.register_buffer("cos_buffer", cos, persistent=False)
        self.register_buffer("sin_buffer", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        cos = self.cos_buffer[token_positions]
        sin = self.sin_buffer[token_positions]

        x_rotated = torch.empty_like(x)
        # 这里的 0::2 表示：从索引 0 开始，每隔 2 个取一个（即取所有偶数位 x0, x2...）
        # 把奇数位的值取负，填到偶数位上
        x_rotated[..., 0::2] = -x[..., 1::2] # x_rotated = [-x1, x1, -x3, x3]
        x_rotated[..., 1::2] = x[..., 0::2]  # x_rotated = [-x1, x0, -x3, x2]

        return x * cos + x_rotated * sin

        