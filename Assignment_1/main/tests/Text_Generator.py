import torch
import torch.nn.functional as F

def top_p_sampling(logits, p):
    """
    核采样，就是保留累积概率达到p的最小词集
    """
    if p >= 1.0:
        return logits # 全量概率

    # 先排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    # 算累积概率
    cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probabilities > p # 遍历cumu里面的每一个值 当前值大于p就在这个位置改成True
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone() # 把刚超过p的那个词保留下来 其他照删不误
    sorted_indices_to_remove[..., 0] = False # 第一个必须留下 不然删光了就好玩了
    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')

    return logits

def generate(model, prompt_tokens, max_new_tokens, temperature=1.0, top_p=1.0, end_token=None):
    """
    文本生成主函数
    """
    model.eval()
    # prompt_tokens的形状(batch, seq_len)
    generated = prompt_tokens

    for i in range(max_new_tokens):
        # forward部分 只取序列最后一个位置的预测结果
        if generated.size(1) <= 128:
            context_window = generated
        else:
            context_window = generated[:, -128:]

        logits = model(context_window)
        last_logits = logits[:, -1, :] # 形状(batch, vocab_size)

        # 温度缩放部分
        last_logits = last_logits / max(temperature, 1e-5)

        # 核采样Top-p
        last_logits = top_p_sampling(last_logits, top_p)

        probabilities = F.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1) # 形状是(batch, 1)
        generated = torch.cat((generated, next_token), dim=1)
        
        if end_token is not None and (next_token == end_token).all():
            break # 查停机符的

    return generated