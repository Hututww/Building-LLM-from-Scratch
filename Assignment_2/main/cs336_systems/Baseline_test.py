import os
import sys
import torch
import timeit
import argparse
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../cs336-basics"))

from cs336_basics.model import BasicsTransformerLM

MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, choices=MODEL_CONFIGS.keys(), default="small")
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["fwd", "fwd_bwd"], default="fwd")
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MODEL_CONFIGS[args.model_size]
    
# init
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=args.context_length,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        rope_theta=10000.0,
    ).to(device)

# generate rand input
    input_ids = torch.randint(0, 10000, (args.batch_size, args.context_length), device=device)

# warmup and benchmking flow
    def _run_step():
        if args.mode == "fwd":
            with torch.no_grad():
                model(input_ids)
        else:
            loss = model(input_ids).sum()
            loss.backward()
            model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

    print(f"预热 ({args.warmup_steps} 步)...")
    for _ in range(args.warmup_steps):
        _run_step()

    print(f"测时 ({args.num_steps} 步)...")
    start = timeit.default_timer()
    for _ in range(args.num_steps):
        _run_step()
    end = timeit.default_timer()

# calc time
    avg_time = (end - start) / args.num_steps
    result = {
        "模型规模": args.model_size,
        "上下文长度": args.context_length,
        "批次大小": args.batch_size,
        "模式": args.mode,
        "平均耗时(秒)": avg_time,
        "吞吐量(tokens/秒)": (args.batch_size * args.context_length) / avg_time,
        "显存占用(MB)": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    }
    
    print(pd.DataFrame([result]).to_markdown(index=False))

if __name__ == "__main__":
    benchmark()
