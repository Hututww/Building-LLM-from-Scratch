from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


from .BPE_tokenizer import Tokenizer, PAT
from .Transformer import Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding, MultiheadSelfAttention, TransformerBlock, TransformerLM, softmax, scaled_dot_product_attention
from .Training_Utils_for_Trans import AdamW, calc_cross_entropy_loss, learning_rate_schedule

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    给定线性层的权重，计算批处理输入的变换结果

    Args:
        in_dim (int)：输入维度的尺寸
        out_dim (int)：输出维度的尺寸
        weights (Tensor)：使用的线性层权重
        in_features (Tensor)：应用该函数的输入张量

    Returns:
        Tensor: 线性模块变换后的输出结果
    """
    layer = Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    layer.load_state_dict({"weight": weights.T})

    return layer(in_features)

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    给定嵌入层的权重，获取一批令牌(token)ID 对应的嵌入向量

    Args:
        vocab_size (int)：词汇表中的嵌入数量
        d_model (int)：嵌入向量的维度大小
        weights (Tensor)：从中提取向量的嵌入矩阵
        token_ids (Tensor)：要从嵌入层提取的令牌 ID 集合

    Returns:
        Float[Tensor, "... d_model"]: 由你的嵌入层返回的批处理嵌入向量.
    """
    emb = Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    emb.load_state_dict({"weight": weights})
        
    return emb(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 SwiGLU 网络的权重，返回使用这些权重运行实现的输出结果

    参数：
        d_model (int): 前馈网络输入和输出的维度尺寸
        d_ff (int): SwiGLU 内部升维变换的维度尺寸
        w1_weight (Float[Tensor, "d_ff d_model"]): 为 W1 存储的权重
        w2_weight (Float[Tensor, "d_model d_ff"]): 为 W2 存储的权重
        w3_weight (Float[Tensor, "d_ff d_model"]): 为 W3 存储的权重
        in_features (Float[Tensor, "... d_model"]): 输入到前馈层的嵌入特征

    返回值：
        Float[Tensor, "... d_model"]: 与输入嵌入特征形状相同的输出嵌入结果
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu = SwiGLU(d_model, d_ff=d_ff, device=w1_weight.device, dtype=w1_weight.dtype) # 设置属性 模块
    swiglu.load_state_dict({
        "w1.weight": w1_weight.T,
        "w2.weight": w2_weight.T,
        "w3.weight": w3_weight.T,
    })

    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    给定键 (K)、查询 (Q) 和值 (V) 张量，返回你的缩放点积注意力实现的输出结果

    参数：
        Q (Float[Tensor, " ... queries d_k"]): 查询 (Query) 张量
        K (Float[Tensor, " ... keys d_k"]): 键 (Key) 张量
        V (Float[Tensor, " ... values d_v"]): 值 (Value) 张量
        mask (Bool[Tensor, " ... queries keys"] | None): 掩码 (Mask) 张量

    返回值：
        Float[Tensor, " ... queries d_v"]: 缩放点积注意力 (SDPA) 的输出结果
    """
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    给定一个朴素的、非批处理实现的多头注意力的键（K）、查询（Q）和值（V）投影权重，
    返回一个优化的批处理实现的输出结果。该实现应能在单次矩阵乘法中完成所有头的键、
    查询和值投影计算。

    此函数不应使用 RoPE（旋转位置编码）。
    参见 Vaswani 等人 (2017) 论文的第 3.2.2 节。

    参数：
        d_model (int): 前馈输入和输出的特征维度尺寸。
        num_heads (int): 多头注意力中使用的头数。
        max_seq_len (int): 如果你的实现包含预缓存逻辑，则为支持的最大序列长度。
        q_proj_weight (Float[Tensor, "d_k d_in"]): 查询 (Q) 投影的权重。
        k_proj_weight (Float[Tensor, "d_k d_in"]): 键 (K) 投影的权重。
        v_proj_weight (Float[Tensor, "d_k d_in"]): 值 (V) 投影的权重。
        o_proj_weight (Float[Tensor, "d_model d_v"]): 输出投影的权重。
        in_features (Float[Tensor, "... sequence_length d_in"]): 运行该实现的输入张量。

    返回值：
        Float[Tensor, " ... sequence_length d_out"]: 使用给定的 QKV 投影权重和输入特征，
        运行优化的批处理多头注意力实现后得到的输出张量。
    """
    mha = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, device=in_features.device)

    mha.load_state_dict({
        "w_q.weight": q_proj_weight.T,
        "w_k.weight": k_proj_weight.T,
        "w_v.weight": v_proj_weight.T,
        "w_o.weight": o_proj_weight.T,
    })

    return mha(in_features)

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    给定一个朴素的、非批处理实现的多头注意力的键（K）、查询（Q）和值（V）投影权重，
    返回一个优化的批处理实现的输出结果。该实现应能在单次矩阵乘法中完成所有头的键、
    查询和值投影计算。

    此版本的 MHA 应包含旋转位置编码（RoPE）。
    在这种情况下，RoPE 的嵌入维度必须是头嵌入维度（d_model // num_heads）。
    参见 Vaswani 等人 (2017) 论文的第 3.2.2 节。

    参数：
        d_model (int): 前馈输入和输出的特征维度。
        num_heads (int): 多头注意力中使用的头数。
        max_seq_len (int): 如果你的实现包含预缓存逻辑，则为支持的最大序列长度。
        theta (float): RoPE 参数（即公式中的 Theta）。
        q_proj_weight (Float[Tensor, "d_k d_in"]): 查询（Q）投影的权重。
        k_proj_weight (Float[Tensor, "d_k d_in"]): 键（K）投影的权重。
        v_proj_weight (Float[Tensor, "d_k d_in"]): 值（V）投影的权重。
        o_proj_weight (Float[Tensor, "d_model d_v"]): 输出投影的权重。
        in_features (Float[Tensor, "... sequence_length d_in"]): 运行该实现的输入张量。
        token_positions (Int[Tensor, " ... sequence_length"] | None): 可选张量，包含令牌（token）的位置信息。

    返回值：
        Float[Tensor, " ... sequence_length d_out"]: 使用给定的 QKV 投影权重和输入特征，
        运行优化的批处理多头注意力实现后得到的输出张量。
    """
    mha = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, device=in_features.device)
    
    mha.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_model//num_heads, max_seq_len=max_seq_len, device=in_features.device)
    
    mha.load_state_dict({
        "w_q.weight": q_proj_weight.T,
        "w_k.weight": k_proj_weight.T,
        "w_v.weight": v_proj_weight.T,
        "w_o.weight": o_proj_weight.T,
    })

    return mha(in_features, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    针对给定的输入张量运行 RoPE（旋转位置编码）

    参数：
        d_k (int): Query 或 Key 张量的嵌入维度大小。
        theta (float): RoPE 的参数（即公式中的 Theta）
        max_seq_len (int): 预计算缓存支持的最大序列长度（如果你的实现包含预缓存逻辑）
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): 需要执行 RoPE 的输入张量
        token_positions (Int[Tensor, "... sequence_length"]): 包含每个 token 位置信息的张量，
            通常形状为 (batch_size, sequence_length)

    返回值：
        Float[Tensor, " ... sequence_length d_k"]: 应用了 RoPE 旋转变换后的张量
    """
    rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_k,
        max_seq_len=max_seq_len,
        device=in_query_or_key.device
    )
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定前置归一化（pre-norm）Transformer 块的权重和输入特征，
    返回在该输入特征上执行 Transformer 块计算后的输出结果。

    此函数应当使用 RoPE（旋转位置编码）。
    根据你的具体实现，你可能只需要将相关参数传递给 TransformerBlock 的构造函数，
    或者你可能需要初始化自己的 RoPE 类并将其作为参数传入。

    参数：
        d_model (int): Transformer 块输入的维度。
        num_heads (int): 多头自注意力中使用的头数。`d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 逐位置前馈网络（feed-forward）内部层的维度。
        max_seq_len (int): 如果你的实现包含预缓存逻辑，则为支持的最大序列长度。
        theta (float): RoPE 参数（即公式中的 Theta）。
        weights (dict[str, Tensor]):
            参考实现的状态字典（state dict）。
            该字典的键（key）包括：
            - `attn.q_proj.weight`
                所有 `num_heads` 个注意力头的查询（Query）投影权重。
                形状为 (d_model, d_model)。
                行数据按 (num_heads, d_k) 的矩阵顺序排列，
                即 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `attn.k_proj.weight`
                所有 `num_heads` 个注意力头的键（Key）投影权重。
                形状为 (d_model, d_model)。
                行数据按 (num_heads, d_k) 的矩阵顺序排列，
                即 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `attn.v_proj.weight`
                所有 `num_heads` 个注意力头的值（Value）投影权重。
                形状为 (d_model, d_model)。
                行数据按 (num_heads, d_v) 的矩阵顺序排列，
                即 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `attn.output_proj.weight`
                多头自注意力输出投影（output projection）的权重。
                形状为 (d_model, d_model)。
            - `ln1.weight`
                Transformer 块中应用于第一个 RMSNorm 的仿射变换（affine transform）权重。
                形状为 (d_model,)。
            - `ffn.w1.weight`
                前馈网络（FFN）中第一个线性变换的权重。
                形状为 (d_model, d_ff)。
            - `ffn.w2.weight`
                前馈网络（FFN）中第二个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `ffn.w3.weight`
                前馈网络（FFN）中第三个线性变换的权重。
                形状为 (d_model, d_ff)。
            - `ln2.weight`
                Transformer 块中应用于第二个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            用于运行该实现的输入张量。

    返回值：
        Float[Tensor, "batch sequence_length d_model"]: 在使用 RoPE 的情况下，
        对输入特征运行 Transformer 块后的输出张量。
    """
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, device=in_features.device, dtype=in_features.dtype)

    d_k = d_model // num_heads
    block.attn.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_features.device)

    block.load_state_dict({
        "attn_norm.weight": weights["ln1.weight"],
        "attn.w_q.weight": weights["attn.q_proj.weight"].T,
        "attn.w_k.weight": weights["attn.k_proj.weight"].T,
        "attn.w_v.weight": weights["attn.v_proj.weight"].T,
        "attn.w_o.weight": weights["attn.output_proj.weight"].T,
        "ffn_norm.weight": weights["ln2.weight"],
        "ffn.w1.weight": weights["ffn.w1.weight"].T,
        "ffn.w2.weight": weights["ffn.w2.weight"].T,
        "ffn.w3.weight": weights["ffn.w3.weight"].T,
    })

    batch_size, seq_len, _ = in_features.shape
    positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0).expand(batch_size, -1)

    return block(in_features, positions=positions)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """
    给定前置归一化（pre-norm）Transformer 块的权重和输入特征，返回在输入特征上运行该 
    Transformer 块的输出结果。

    此函数应使用 RoPE（旋转位置编码）。
    取决于具体的实现，你可能只需要将相关参数传递给 TransformerBlock 的构造函数，
    或者你可能需要初始化自己的 RoPE 类并将其作为参数传入。

    参数：
        d_model (int): Transformer 块输入的维度。
        num_heads (int): 多头自注意力（multi-head self-attention）中使用的头数。
            `d_model` 必须能被 `num_heads` 整除。
        d_ff (int): 前馈网络内部层的维度。
        max_seq_len (int): 预缓存支持的最大序列长度（如果你的实现包含预缓存逻辑）。
        theta (float): RoPE 参数（即公式中的 Theta）。
        weights (dict[str, Tensor]):
            参考实现的状态字典（State dict）。
            此字典的键包括：
            - `attn.q_proj.weight`
                所有 `num_heads` 个注意力头的查询（Query）投影权重。
                形状为 (d_model, d_model)。
                行数据按形状为 (num_heads, d_k) 的矩阵排列，
                即 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `attn.k_proj.weight`
                所有 `num_heads` 个注意力头的键（Key）投影权重。
                形状为 (d_model, d_model)。
                行数据按形状为 (num_heads, d_k) 的矩阵排列，
                即 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `attn.v_proj.weight`
                所有 `num_heads` 个注意力头的值（Value）投影权重。
                形状为 (d_model, d_model)。
                行数据按形状为 (num_heads, d_v) 的矩阵排列，
                即 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `attn.output_proj.weight`
                多头自注意力输出投影的权重。
                形状为 (d_model, d_model)。
            - `ln1.weight`
                应用于 Transformer 块中第一个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
            - `ffn.w1.weight`
                前馈网络（FFN）中第一个线性变换的权重。
                形状为 (d_model, d_ff)。
            - `ffn.w2.weight`
                前馈网络（FFN）中第二个线性变换的权重。
                形状为 (d_ff, d_model)。
            - `ffn.w3.weight`
                前馈网络（FFN）中第三个线性变换的权重。
                形状为 (d_model, d_ff)。
            - `ln2.weight`
                应用于 Transformer 块中第二个 RMSNorm 的仿射变换权重。
                形状为 (d_model,)。
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            用于运行该实现的输入张量。

    返回值：
        Float[Tensor, "batch sequence_length d_model"]: 在使用 RoPE 的情况下，
        输入特征运行 Transformer 块后的输出张量。
    """
    model = TransformerLM(vocab_size=vocab_size, context_length=context_length, d_ff=d_ff, d_model=d_model, num_heads=num_heads, num_layers=num_layers, device=in_indices.device)
    d_k = d_model // num_heads

    new_state_dict = {}
    new_state_dict["token_embedding.weight"] = weights["token_embeddings.weight"]
    new_state_dict["final_norm.weight"] = weights["ln_final.weight"]
    new_state_dict["output_layer.weight"] = weights["lm_head.weight"].T

    for i in range(num_layers):
        layer_name = f"layers.{i}."
    
        model.layers[i].attn.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_k,max_seq_len=context_length, device=in_indices.device)
        # attn
        new_state_dict[f"{layer_name}attn_norm.weight"] = weights[f"{layer_name}ln1.weight"]
        new_state_dict[f"{layer_name}attn.w_q.weight"] = weights[f"{layer_name}attn.q_proj.weight"].T
        new_state_dict[f"{layer_name}attn.w_k.weight"] = weights[f"{layer_name}attn.k_proj.weight"].T
        new_state_dict[f"{layer_name}attn.w_v.weight"] = weights[f"{layer_name}attn.v_proj.weight"].T
        new_state_dict[f"{layer_name}attn.w_o.weight"] = weights[f"{layer_name}attn.output_proj.weight"].T

        # ffn
        new_state_dict[f"{layer_name}ffn_norm.weight"] = weights[f"{layer_name}ln2.weight"]
        new_state_dict[f"{layer_name}ffn.w1.weight"] = weights[f"{layer_name}ffn.w1.weight"].T
        new_state_dict[f"{layer_name}ffn.w2.weight"] = weights[f"{layer_name}ffn.w2.weight"].T
        new_state_dict[f"{layer_name}ffn.w3.weight"] = weights[f"{layer_name}ffn.w3.weight"].T

    model.load_state_dict(new_state_dict)
    return model(in_indices)
    
def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 RMSNorm 仿射变换的权重，返回在输入特征上运行 RMSNorm 的输出结果。

    参数：
        d_model (int): RMSNorm 输入的维度尺寸
        eps (float): 为了数值稳定性而添加到分母中的微小值
        weights (Float[Tensor, "d_model"]): RMSNorm 的权重（缩放参数）
        in_features (Float[Tensor, "... d_model"]): 待执行 RMSNorm 的输入特征
            可以具有任意数量的前导维度（如 batch 或 sequence 维度）

    返回值：
        Float[Tensor, "... d_model"]: 形状与 `in_features` 相同的张量，
        包含对 `in_features` 运行 RMSNorm 后的输出计算结果。
    """
    layer = RMSNorm(d_model, eps, device=weights.device, dtype=weights.dtype)
    layer.load_state_dict({"weight": weights})

    return layer(in_features) 


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    给定一个输入张量，返回对该输入在给定维度 `dim` 上执行 softmax 操作后的结果

    参数：
        in_features (Float[Tensor, "..."]): 待执行 softmax 的输入特征。形状是任意的
        dim (int): 对 `in_features` 应用 softmax 的维度

    返回值：
        Float[Tensor, "..."]: 与 `in_features` 形状相同的张量，包含在指定维度进行 softmax 归一化后的输出
    """
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """
    给定输入和目标的张量，计算所有样本的平均交叉熵损失。

    参数：
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] 是第 i 个样本中第 j 类的未归一化对数概率（logit）。
        targets (Int[Tensor, "batch_size"]): 形状为 (batch_size,) 的张量，包含正确类别的索引。
            每个值必须在 0 到 `num_classes - 1` 之间。

    返回值：
        Float[Tensor, ""]: 所有样本的平均交叉熵损失。
    """
    return calc_cross_entropy_loss(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    给定带有线性热身的余弦学习率衰减调度的参数以及迭代轮数，返回该迭代轮数在指定调度下的学习率。

    参数：
        it (int): 获取学习率的当前迭代轮数。
        max_learning_rate (float): alpha_max，余弦学习率调度中的最大学习率。
        min_learning_rate (float): alpha_min，余弦学习率调度中的最小/最终学习率。
        warmup_iters (int): T_w，学习率线性热身的迭代轮数。
        cosine_cycle_iters (int): T_c，余弦退火迭代的总轮数。

    返回值：
        在指定调度下，当前迭代轮数对应的学习率。
    """
    return learning_rate_schedule(alpha_max=max_learning_rate, alpha_min=min_learning_rate, t=it, t_w=warmup_iters, t_c=cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
            input_path: str | os.PathLike,
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