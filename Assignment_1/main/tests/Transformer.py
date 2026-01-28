import torch
import torch.nn as nn
import math

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    把一堆原始分布变成和为1的概率分布
    """
    max_val = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_val)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)

    return exp_x / sum_exp


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None):
    # 缩放点积注意力
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) # 最后两维是序列长度seq_len 特征维度d_k
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    
    weights = softmax(scores, dim=-1)
    return torch.matmul(weights, value)

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

class MultiheadSelfAttention(nn.Module):
    # 多头注意力
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(theta=10000.0, d_k=self.d_k, max_seq_len=2048, device=device)
    
    def forward(self, x: torch.Tensor, positions=None):
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x) # (Batch, Seq_Len, num_heads, d_k)
        k = self.w_k(x)
        v = self.w_v(x)

        # (batch, seq, num_heads, d_k) -> (batch, num_heads, seq, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        q = self.rope(q, positions)
        k = self.rope(k, positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        attention_output = scaled_dot_product_attention(q, k, v, mask=mask)
        out = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.w_o(out)

class TransformerBlock(nn.Module):
    # 单个的Transformer 块
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # 第一个子层：多头注意力
        self.attn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(d_model, num_heads, device=device, dtype=dtype)

        # 第二个子层：前馈网络SwiGLU
        self.ffn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None):
        """
        x的形状: [Batch_Size, Sequence_Length, d_model]。

            Batch_Size: 一批同时训练的句子数量。
            Sequence_Length: 一句话里有多少个词 Token数。
            d_model: 每个词被表示成了一个多长的向量。
        """
        # 多头注意力子层
        x = x + self.attn(self.attn_norm(x), positions=positions)
        
        # 前馈网络子层
        x = x + self.ffn(self.ffn_norm(x))
        
        return x
    
class TransformerLM(nn.Module):
    # 属于我的flow1！
    def __init__(self, vocab_size: int, context_length: int, d_ff: int, d_model: int, num_heads: int, num_layers: int, device=None, dtype=None):
        super().__init__()
        # 词嵌入层
        self.token_embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # 层堆叠
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype) for _ in range(num_layers)])

        # 输出前的归一化
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        
        # 输出线性层：将特征映射回词词汇表大小
        self.output_layer = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor):
        x = self.token_embedding(token_ids)
        
        # 2. 为 RoPE 准备位置索引
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # 逐层通过 TransformerBlock
        for layer in self.layers:
            x = layer(x, positions=positions)
            
        x = self.final_norm(x)
        logits = self.output_layer(x)
        
        return logits
