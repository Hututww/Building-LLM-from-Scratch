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

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "iteration": iteration}
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, map_location='cpu', weights_only=True)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint["iteration"]