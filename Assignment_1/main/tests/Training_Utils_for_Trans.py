import torch
from torch import Tensor

def calc_cross_entropy_loss(inputs: Tensor, targets: Tensor):
    max_logits = torch.max(inputs, dim=-1, keepdim=True).values
    shifted_logits = inputs - max_logits

    log_accumulate_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True)) + max_logits

    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1))
    loss = log_accumulate_exp - target_logits

    return torch.mean(loss)