from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


from .BPE_tokenizer import Tokenizer, PAT
from .Transformer import Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding, MultiheadSelfAttention, softmax, scaled_dot_product_attention

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
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
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
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
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
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


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
    r"""Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


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
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


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