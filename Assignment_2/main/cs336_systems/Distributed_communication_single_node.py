import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def benchmark_all_reduce(rank, world_size, backend, tensor_size):
    setup(rank, world_size, backend)

    device = 'cuda' if backend == 'nccl' else 'cpu'
    data = torch.randn(tensor_size, device=device)
    torch.cuda.synchronize()

    start_time = time.time()
    dist.all_reduce(data, async_op=False)
    elapsed_time = time.time() - start_time

    print(f"Rank {rank} finished all-reduce in {elapsed_time:.6f} seconds")

    dist.destroy_process_group()

def run_benchmark():
    backends = ["gloo", "nccl"]
    tensor_sizes = [1 * 1024**2, 10 *1024**2, 100 * 1024**2, 1024**3]
    process_counts = [2, 4, 6]
    results = []

    for backend in backends:
        for tensor_size in tensor_sizes:
            for world_size in process_counts:
                print(f"Testing backend={backend}, tensor_size={tensor_size}, world_size={world_size}")
                mp.spawn(benchmark_all_reduce, args=(world_size, backend, tensor_size), nprocs=world_size)

if __name__ == "__main__":
    run_benchmark()                