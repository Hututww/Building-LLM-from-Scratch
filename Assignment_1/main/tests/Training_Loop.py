import os
import torch
import numpy as np

from .Training_Utils_for_Trans import calc_cross_entropy_loss, gradient_clipping

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

def train(model, 
          optimizer, 
          train_data_path, 
          max_iters=1000, 
          batch_size=3, 
          context_length=128,
          device="cpu",
          checkpoint_path="checkpoints"
          ):
    # 暂时还没加验证的逻辑
    os.makedirs(checkpoint_path, exist_ok=True)
    train_data = np.memmap(train_data_path, dtype=np.uint16, mode="r")

    model.to(device)
    model.train()

    for i in range(max_iters):
        inputs, targets = get_batch(train_data, batch_size, context_length, device)

        logits = model(inputs)

        loss = calc_cross_entropy_loss(logits, targets)

        optimizer.zero_grad()
        loss.backward()

        gradient_clipping(model.parameters(), l2_max=1.0)

        optimizer.step()

        if i % 10 == 0:
            print(f"迭代 {i}/{max_iters}, 损失: {loss.item():.4f}")

    print("训练搞定！")