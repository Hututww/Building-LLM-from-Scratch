import torch

def forward(ctx, Q, K, V, is_causal=False):
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
        return output

