import torch
import triton
import triton.language as tl

@triton.jit
def _flash_fwd_kernel(Q, K, V, softmax_scale, l, output,
                      stride_qb, stride_qh, stride_qm, stride_qk,
                      stride_kb, stride_kh, stride_kn, stride_kk,
                      stride_vb, stride_vh, stride_vn, stride_vk,
                      stride_ob, stride_oh, stride_om, stride_ok,
                      batch_size, num_heads, seq_len, head_dim,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    stride_qb	Stride of Q's Batch	    要从第一个 Batch 跳到第二个 Batch，需要跨过多少个数字。
    stride_qh	Stride of Q's Head	    在同一个 Batch 里，从第一个“头”跳到第二个“头”需要走多少步。
    stride_qm	Stride of Q's M-axis	在同一个“头”里，从第一行 Token 跳到第二行 Token 需要走多少步。
    stride_qk	Stride of Q's K-dim	    在同一行里，从第一个特征值跳到第二个特征值需要走多少步。
    """
    program_id = tl.program_id(0)

    offset_headbatch = tl.program_id(1)
    off_b = offset_headbatch // num_heads
    off_h = offset_headbatch % num_heads

    q_ptr = Q + off_b * stride_qb + off_h * stride_qh
    k_ptr = K + off_b * stride_kb + off_h * stride_kh
    v_ptr = V + off_b * stride_vb + off_h * stride_vh
    o_ptr = output + off_b * stride_ob + off_h * stride_oh
    l_ptr = l + offset_headbatch * seq_len

    row_m = tl.arange(0, BLOCK_M) + program_id * BLOCK_M
    row_n = tl.arange(0, BLOCK_N)
    row_k = tl.arange(0, head_dim)

    q = tl.load(q_ptr + row_m[:, None] * stride_qm + row_k[None, :] * stride_qk)

    a_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    exp_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    total = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):

    





class flash_forward_pass_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        纯PyTorch实现FlashAttention-2前向
        输入形状：[batch_size, num_heads, seq_len, head_dim]
        输出：output, log_sum_exp
        """
        device = Q.device
        dtype = Q.dtype
        batch_size, num_heads, seq_len, head_dim = Q.shape
        output = torch.zeros_like(Q)
        log_sum_exp = torch.full((batch_size, num_heads, seq_len), -float('inf'), device=device, dtype=torch.float32)

        block_row_size = 16
        block_col_size = 16

        num_row = (seq_len + block_row_size - 1) // block_row_size
        num_col = (seq_len + block_col_size - 1) // block_col_size

        for i in range(num_row):
            max_score = torch.full((batch_size, num_heads, min((i + 1) * block_row_size, seq_len) - i * block_row_size), -float("inf"), device=device, dtype=torch.float32)
            sum_exp = torch.zeros((batch_size, num_heads, min((i + 1) * block_row_size, seq_len) - i * block_row_size), device=device, dtype=torch.float32)
            accumulated_output = torch.zeros((batch_size, num_heads, min((i + 1) * block_row_size, seq_len) - i * block_row_size, head_dim), device=device, dtype=dtype)
            
            query = Q[:, :, i * block_row_size:min((i + 1) * block_row_size, seq_len), :]

            for j in range(num_col):
                key = K[:, :, j * block_col_size:min((j + 1) * block_col_size, seq_len), :]
                value = V[:, :, j * block_col_size:min((j + 1) * block_col_size, seq_len), :]

                scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
                scores_max = torch.max(scores, dim=-1).values
                real_max = torch.max(max_score, scores_max)

                exp_scale = torch.exp(max_score - real_max)
                max_exp = torch.exp(scores - real_max.unsqueeze(-1))

                sum_exp = sum_exp * exp_scale + torch.sum(max_exp, dim=-1)
                accumulated_output = accumulated_output * exp_scale.unsqueeze(-1) + torch.matmul(max_exp, value)
                max_score = real_max

            output[:, :, i * block_row_size:min((i + 1) * block_row_size, seq_len), :] = accumulated_output / sum_exp.unsqueeze(-1)
            log_sum_exp[:, :, i * block_row_size:min((i + 1) * block_row_size, seq_len)] = max_score + torch.log(sum_exp)

        ctx.save_for_backward(Q, K, V, output, log_sum_exp)
        ctx.is_causal = is_causal
        return output, log_sum_exp
    
    @staticmethod
    def backward(ctx, output, log_sum_exp):
        raise NotImplementedError("not implemented")
    

class flash_forward_pass_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_casual=False): 
        device = Q.device
        dtype = Q.dtype
        batch_size, num_heads, seq_len, head_dim = Q.shape
        softmax_scale = 1.0 / torch.sqrt(torch.tensor(head_dim))
        BLOCK_M = 16
        BLOCK_N = 16
        
        output = torch.zeros_like(Q)
        log_sum_exp = torch.full((batch_size, num_heads, seq_len), -float('inf'), device=device, dtype=torch.float32)
 
        grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)

        _flash_fwd_kernel[grid](
            Q, K, V, softmax_scale, log_sum_exp, output,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            batch_size, num_heads, seq_len, head_dim, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )

        ctx.save_for_backward(Q, K, V, output, log_sum_exp)
        return output, log_sum_exp
    
    @staticmethod
    def backward(ctx, output, log_sum_exp):
        raise NotImplementedError("Not Implemented")