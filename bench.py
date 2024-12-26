#!/usr/bin/env python3

"""
Transformer Operation Benchmarking Script for Multiple GPUs and DTypes

This script benchmarks various transformer operations on different GPU types including:
- NVIDIA RTX 3090 (24GB GDDR6X)
- NVIDIA A10 (24GB GDDR6)
- NVIDIA H100 (80GB HBM3)

Operations benchmarked:
1. Dense matrix multiplication
2. Query-Key attention (initialization phase)
3. Query-Key attention (auto-regressive phase)

Available dtypes:
- Float16 (fp16)
- Float32 (fp32)
- BFloat16 (bf16)
- Int8 (int8)
- Float8 (fp8) - H100 only
"""

import os
import time
import itertools
import numpy as np
import pandas as pd
import pickle
import gzip
from datetime import datetime
from tqdm.auto import tqdm
import torch
import argparse

# Disable gradient computation for benchmarking
torch.set_grad_enabled(False)

# GPU configurations
GPU_CONFIGS = {
    "3090": {
        "memory": 24e9,  # 24GB GDDR6X
        "name": "RTX 3090",
        "supported_dtypes": ["fp16", "fp32", "bf16", "int8"]
    },
    "a10": {
        "memory": 24e9,  # 24GB GDDR6
        "name": "A10",
        "supported_dtypes": ["fp16", "fp32", "bf16", "int8"]
    },
    "h100": {
        "memory": 80e9,  # 80GB HBM3
        "name": "H100",
        "supported_dtypes": ["fp16", "fp32", "bf16", "int8", "fp8"]
    }
}
DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "int8": torch.int8,
    "fp8": torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else None
}
DTYPE_SIZES = {
    "fp16": 2,  # 2 bytes
    "fp32": 4,  # 4 bytes
    "bf16": 2,  # 2 bytes
    "int8": 1,  # 1 byte
    "fp8": 1    # 1 byte
}

# Benchmark configurations
ND_LIST = list(itertools.chain(itertools.product([12, 16, 32], [64]), itertools.product([32, 40, 56, 72, 96], [128])))
SEQLEN_LIST = [10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 5000]
BS_LIST = list(itertools.chain(range(1, 8), range(8, 16, 2), range(16, 32, 4), range(32, 64, 8), range(64, 128, 16), [128]))

def get_available_gpu():
    """
    Find and return the first available CUDA GPU.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPUs available")
    return torch.device(f"cuda:0")

def benchmark(f, *, f_setup=None, min_repeat: int, min_secs: float, tqdm_kwargs: dict | None=None) -> np.ndarray:
    """
    Benchmark a function by running it multiple times and measuring latency.
    """
    latency = []
    
    # First run, ignore min_secs
    if f_setup is not None:
        f_setup()
    st = time.perf_counter_ns()
    f()
    ed = time.perf_counter_ns()
    latency.append((ed-st)/1e9)
    
    # Subsequent runs, until reaching both min_repeat and min_secs
    min_nanos = int(min_secs * 1e9)
    start_nanos = time.perf_counter_ns()
    while True:
        now_nanos = time.perf_counter_ns()
        if len(latency) > min_repeat and now_nanos - start_nanos > min_nanos:
            break
        if f_setup is not None:
            f_setup()
        st = time.perf_counter_ns()
        f()
        ed = time.perf_counter_ns()
        latency.append((ed-st)/1e9)
    return np.array(latency)

def tail_mean(xs, skip=0.2):
    """Calculate mean of array after skipping initial portion."""
    return xs[int(len(xs) * skip):].mean()

def benchmark_dense(out, nd_list, seqlen_list, bs_list, gpu_config, dtype_name):
    """
    Benchmark dense matrix multiplication operations using specified precision.
    """
    device = get_available_gpu()
    memory_limit = gpu_config["memory"]
    dtype = DTYPE_MAP[dtype_name]
    bytes_per_elem = DTYPE_SIZES[dtype_name]
    seqlen_list = [1] + seqlen_list
    total = len(list(itertools.product(nd_list, seqlen_list, bs_list)))
    pbar = tqdm(total=total)
    
    for (n, d), seqlen in reversed(list(itertools.product(nd_list, seqlen_list))):
        h = n * d
        try:
            maxbs = max(b for b in bs_list if (b*seqlen*h + h*h + b*seqlen*h) * bytes_per_elem < memory_limit)
        except ValueError:
            pbar.update(len(bs_list))
            continue
            
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)
        X = torch.rand((maxbs, seqlen, h), dtype=dtype, device=device)
        W = torch.rand((h, h), dtype=dtype, device=device)
            
        torch.cuda.synchronize()
        for bs in reversed(bs_list):
            if bs > maxbs:
                pbar.update()
                continue
                
            pbar.set_postfix(n=n, h=h, d=d, seqlen=seqlen, bs=bs, dtype=dtype_name)
            def run():
                torch.matmul(X[:bs], W)
                torch.cuda.synchronize()
                
            def clear_cache():
                cache.zero_()
                torch.cuda.synchronize()
                
            latency = benchmark(run, f_setup=clear_cache, min_repeat=20, min_secs=2)
            l = tail_mean(latency)
            out.append({
                "n": n,
                "d": d,
                "seqlen": seqlen,
                "bs": bs,
                "dtype": dtype_name,
                "latency": l
            })
            pbar.update()
        del cache, X, W
        torch.cuda.empty_cache()
    pbar.close()

def benchmark_qk_init(out, nd_list, seqlen_list, bs_list, gpu_config, dtype_name):
    """
    Benchmark Query-Key attention initialization.
    """
    device = get_available_gpu()
    memory_limit = gpu_config["memory"]
    dtype = DTYPE_MAP[dtype_name]
    bytes_per_elem = DTYPE_SIZES[dtype_name]
    total = len(list(itertools.product(nd_list, seqlen_list, bs_list)))
    pbar = tqdm(total=total)
    
    for (n, d), seqlen in reversed(list(itertools.product(nd_list, seqlen_list))):
        h = n * d
        try:
            # maxbs = max(b for b in bs_list if b*n*seqlen*d*2*2+b*n*seqlen**2*2 < memory_limit)
            maxbs = max(b for b in bs_list if (b*n*seqlen*d*2 + b*n*seqlen**2) * bytes_per_elem < memory_limit)
        except ValueError:
            pbar.update(len(bs_list))
            continue
            
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)
        Qmax = torch.rand((maxbs, n, seqlen, d), dtype=dtype, device=device)
        Kmax = torch.rand((maxbs, n, seqlen, d), dtype=dtype, device=device)
            
        torch.cuda.synchronize()
        for bs in reversed(bs_list):
            pbar.set_postfix(n=n, h=h, d=d, seqlen=seqlen, bs=bs, dtype=dtype_name)
            if bs > maxbs:
                pbar.update()
                continue
            Q = Qmax[:bs]
            K = Kmax[:bs]
            def run():
                torch.bmm(Q.view(bs * n, seqlen, d), 
                         K.view(bs * n, seqlen, d).transpose(1, 2))
                torch.cuda.synchronize()
            def clear_cache():
                cache.zero_()
                torch.cuda.synchronize()
            latency = benchmark(run, f_setup=clear_cache, min_repeat=20, min_secs=2)
            l = tail_mean(latency)
            out.append({
                "n": n,
                "d": d,
                "seqlen": seqlen,
                "bs": bs,
                "dtype": dtype_name,
                "latency": l
            })
            pbar.update()
        del cache, Q, K, Qmax, Kmax
        torch.cuda.empty_cache()
    pbar.close()

def benchmark_qk_ar(out, nd_list, seqlen_list, bs_list, gpu_config, dtype_name):
    """
    Benchmark Query-Key attention in auto-regressive mode.
    """
    device = get_available_gpu()
    memory_limit = gpu_config["memory"]
    dtype = DTYPE_MAP[dtype_name]
    bytes_per_elem = DTYPE_SIZES[dtype_name]
    total = len(list(itertools.product(nd_list, seqlen_list, bs_list)))
    pbar = tqdm(total=total)
    
    for (n, d), seqlen in reversed(list(itertools.product(nd_list, seqlen_list))):
        h = n * d
        try:
            # maxbs = max(b for b in bs_list if b*n*(1+seqlen)*d*2+b*n*seqlen*2 < memory_limit)
            maxbs = max(b for b in bs_list if (b*n*(1+seqlen)*d + b*n*seqlen) * bytes_per_elem < memory_limit)
        except ValueError:
            pbar.update(len(bs_list))
            continue
            
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)
        Qmax = torch.rand((maxbs, n, 1, d), dtype=dtype, device=device)
        Kmax = torch.rand((maxbs, n, seqlen, d), dtype=dtype, device=device)
            
        torch.cuda.synchronize()
        
        for bs in reversed(bs_list):
            pbar.set_postfix(n=n, h=h, d=d, seqlen=seqlen, bs=bs, dtype=dtype_name)
            if bs > maxbs:
                pbar.update()
                continue
                
            Q = Qmax[:bs]
            K = Kmax[:bs]
            
            def run():
                torch.bmm(Q.view(bs * n, 1, d), 
                         K.view(bs * n, seqlen, d).transpose(1, 2))
                torch.cuda.synchronize()
                
            def clear_cache():
                cache.zero_()
                torch.cuda.synchronize()
                
            latency = benchmark(run, f_setup=clear_cache, min_repeat=20, min_secs=2)
            l = tail_mean(latency)
            out.append({
                "n": n,
                "d": d,
                "seqlen": seqlen,
                "bs": bs,
                "dtype": dtype_name,
                "latency": l
            })
            pbar.update()
        del cache, Q, K, Qmax, Kmax
        torch.cuda.empty_cache()
    pbar.close()

def process_results(data, gpu_config):
    """Process benchmark results and save as CSV."""
    results = []
    for dtype_name in data:
        # Get bytes per element for current dtype
        bytes_per_elem = DTYPE_SIZES[dtype_name]
        
        df_dense = (
            pd.DataFrame.from_dict(data[dtype_name]["dense"])
            .assign(h=lambda x: x["n"] * x["d"])
            .assign(flop=lambda x: (x["bs"] * x["seqlen"] * x["h"]**2) * 2)
            .assign(io=lambda x: (x["bs"]*x["seqlen"]*x["h"] + x["h"]**2 + x["bs"]*x["seqlen"]*x["h"]) * bytes_per_elem)
            .assign(intensity=lambda x: x["flop"] / x["io"])
            .assign(throughput=lambda x: x["bs"]*x["seqlen"] / x["latency"])
            .assign(series="dense")
            .assign(dtype=dtype_name)
        )
        
        df_qk_init = (
            pd.DataFrame.from_dict(data[dtype_name]["qk_init"])
            .assign(h=lambda x: x["n"] * x["d"])
            .assign(flop=lambda x: (x["bs"]*x["n"]*x["d"]*x["seqlen"]**2) * 2)
            .assign(io=lambda x: (x["bs"]*x["n"]*(x["seqlen"]*x["d"]*2 + x["seqlen"]**2)) * bytes_per_elem)
            .assign(intensity=lambda x: x["flop"] / x["io"])
            .assign(throughput=lambda x: x["bs"]*x["seqlen"] / x["latency"])
            .assign(series="qk_init")
            .assign(dtype=dtype_name)
        )
        
        df_qk_ar = (
            pd.DataFrame.from_dict(data[dtype_name]["qk_ar"])
            .assign(h=lambda x: x["n"] * x["d"])
            .assign(flop=lambda x: (x["bs"]*x["n"]*x["d"]*x["seqlen"]) * 2)
            .assign(io=lambda x: (x["bs"]*x["n"]*(x["d"] + x["seqlen"]*x["d"] + x["seqlen"])) * bytes_per_elem)
            .assign(intensity=lambda x: x["flop"] / x["io"])
            .assign(throughput=lambda x: x["bs"] / x["latency"])
            .assign(series="qk_ar")
            .assign(dtype=dtype_name)
        )
        
        results.extend([df_dense, df_qk_init, df_qk_ar])

    # Combine and save all results
    timestamp = datetime.now().strftime("%Y%m%d")
    gpu_name = gpu_config["name"].lower().replace(" ", "-")
    pd.concat(results).to_csv(
        f"data/transformer-batching-microbenchmarks-{gpu_name}-multi-dtype-{timestamp}.csv", 
        index=False
    )

def main(gpu_type, dtypes=None):
    gpu_config = GPU_CONFIGS[gpu_type.lower()]
    
    # Use all supported dtypes if none specified
    if dtypes is None:
        dtypes = gpu_config["supported_dtypes"]
    
    # Run benchmarks
    data = {dtype: {} for dtype in dtypes}
    
    print(f"\nRunning benchmarks on {gpu_config['name']}...")
    print(f"Memory limit: {gpu_config['memory']/1e9:.1f}GB")
    print(f"Testing dtypes: {', '.join(dtypes)}")
    
    for dtype in dtypes:
        print(f"\nRunning benchmarks for {dtype}...")
        
        print("\nRunning Query-Key initialization benchmarks...")
        db = []
        benchmark_qk_init(db, ND_LIST, SEQLEN_LIST, BS_LIST, gpu_config, dtype)
        data[dtype]["qk_init"] = db

        print("\nRunning Query-Key auto-regressive benchmarks...")
        db = []
        benchmark_qk_ar(db, ND_LIST, SEQLEN_LIST, BS_LIST, gpu_config, dtype)
        data[dtype]["qk_ar"] = db

        print("\nRunning dense operation benchmarks...")
        db = []
        benchmark_dense(db, ND_LIST, SEQLEN_LIST, BS_LIST, gpu_config, dtype)
        data[dtype]["dense"] = db

        # Save benchmark results
        timestamp = datetime.now().strftime("%Y%m%d")
        gpu_name = gpu_config["name"].lower().replace(" ", "-")
        with gzip.open(f"data/{timestamp}-transformer-batching-{gpu_name}-{dtype}.pkl.gz", "wb") as f:
            pickle.dump(data[dtype], f)

    # Process and save results as CSV
    process_results(data, gpu_config)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Transformer operation benchmarking across multiple GPU types')
    parser.add_argument('--gpu', type=str, choices=['3090', 'a10', 'h100'], 
                      help='GPU type to benchmark (3090, a10, or h100)')
    parser.add_argument('--dtype', type=str, nargs='+',
                      choices=['fp16', 'fp32', 'bf16', 'int8', 'fp8'],
                      help='Data types to benchmark')
    parser.add_argument('--all', action='store_true', 
                      help='Run benchmarks on all available GPU types')
    parser.add_argument('--custom-memory', type=float,
                      help='Override default GPU memory limit in GB (optional)')
    args = parser.parse_args()

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Print available GPU information
    if torch.cuda.is_available():
        print(f"Available GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        raise RuntimeError("No CUDA GPU available for benchmarking")

    # Handle custom memory limit
    if args.custom_memory:
        for gpu in GPU_CONFIGS:
            GPU_CONFIGS[gpu]["memory"] = args.custom_memory * 1e9
        print(f"\nUsing custom memory limit: {args.custom_memory}GB")

    # Print configuration
    print("\nBenchmarking configurations:")
    print("- Model configurations:", ND_LIST)
    print("- Sequence lengths:", SEQLEN_LIST)
    print("- Batch sizes:", BS_LIST)

    if args.all:
        # Run benchmarks for all GPU types
        for gpu_type in GPU_CONFIGS:
            try:
                print(f"\n{'='*50}")
                print(f"Starting benchmarks for {GPU_CONFIGS[gpu_type]['name']}")
                print(f"{'='*50}")
                main(gpu_type, args.dtype)
            except Exception as e:
                print(f"Error running benchmarks for {gpu_type}: {str(e)}")
                continue
    elif args.gpu:
        # Run benchmarks for specified GPU type
        if args.gpu.lower() not in GPU_CONFIGS:
            print(f"Error: Unsupported GPU type '{args.gpu}'")
            print(f"Supported GPUs: {list(GPU_CONFIGS.keys())}")
            exit(1)
        main(args.gpu, args.dtype)
    else:
        parser.print_help()
        exit(1)
