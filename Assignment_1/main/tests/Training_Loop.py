import torch
import numpy as np

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device):
    max_begin_idx = len(x) - context_length - 1

    random_idx_seed = np.random.randint(0, max_begin_idx + 1, size=(batch_size,))

    inputs_list = [x[i : i + context_length] for i in random_idx_seed]
    targets_list = [x[i + 1 : i + context_length + 1] for i in random_idx_seed]

    inputs = torch.tensor(np.array(inputs_list), dtype=torch.long, device=device)
    targets = torch.tensor(np.array(targets_list), dtype=torch.long, device=device)

    return inputs, targets