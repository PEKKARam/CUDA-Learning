"""Simple script for benchmarking CUDA kernels."""

import time
import argparse
import torch

# Note: We need to import the CUDA kernels *after* importing torch
import my_cuda_kernels


def benchmark(f, iters, warmup, *args):
    """
    Note: This is a simple benchmarking function that does not take into account
    the overhead of the Python interpreter and CUDA device initialization. For
    more accurate results, consider using the PyTorch profiler or Nvidia ncu.
    """
    for _ in range(warmup):
        f(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        out = f(*args)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    print(f"Avg. time: {(t1-t0)/iters/1000:.2f} us ({iters} iterations)")

    """
    Uncomment the following block to run the PyTorch profiler, which provides
    more detailed performance metrics.
    """
    # print("\nRunning PyTorch profiler...")
    # with torch.profiler.profile() as prof:
    #     for i in range(iters):
    #         out = f(*args)
    #         torch.cuda.synchronize()
    # print(prof.key_averages().table())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark CUDA kernels.')
    parser.add_argument('-i', '--iters', type=int, default=1000, help='Measured iterations')
    parser.add_argument('-w', '--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--op', choices=('matmul', 'square'), default='matmul')
    parser.add_argument('--size', type=int, default=1000, help='Matrix/vector size')
    args = parser.parse_args()

    if args.iters < 1 or args.warmup < 0 or args.size < 1:
        parser.error('--iters/--size must be positive and --warmup must be non-negative')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is not available; run this script on a CUDA host.')
    n = args.size

    torch.manual_seed(1)

    if args.op == 'square':
        inp = torch.randn(n, device='cuda')
        print(f"\nBenchmarking square kernel on input size ({n},)")
        benchmark(my_cuda_kernels.square, args.iters, args.warmup, inp)
    else:
        m1 = torch.randn(n, n, device="cuda")
        m2 = torch.randn(n, n, device="cuda")
        print(f"\nBenchmarking matmul kernel on input size ({n}, {n})")
        benchmark(my_cuda_kernels.matmul, args.iters, args.warmup, m1, m2)
