"""Simple script for benchmarking CUDA kernels."""

import time
import argparse
import torch

# Note: We need to import the CUDA kernels *after* importing torch
import my_cuda_kernels


REDUCE_OPS = (
    "reduce_no_bankconflict",
    "reduce_add_during_load",
    "reduce_add_during_load_v2",
    "reduce_unroll_last_warp",
    "reduce_completely_unroll",
    "reduce_multi_add",
    "reduce_shuffle",
)

SGEMM_OPS = ("matmul", "sgemm_naive", "sgemm_baseline")


def benchmark(f, iters, warmup, profile_range, *args):
    """
    Note: This is a simple benchmarking function that does not take into account
    the overhead of the Python interpreter and CUDA device initialization. For
    more accurate results, consider using the PyTorch profiler or Nvidia ncu.
    """
    torch.cuda.nvtx.range_push("warmup")
    try:
        for _ in range(warmup):
            f(*args)
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    if profile_range:
        torch.cuda.cudart().cudaProfilerStart()

    torch.cuda.nvtx.range_push("benchmark")
    t0 = time.perf_counter_ns()
    try:
        for _ in range(iters):
            out = f(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
    finally:
        torch.cuda.nvtx.range_pop()
        if profile_range:
            torch.cuda.cudart().cudaProfilerStop()

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
    parser.add_argument(
        '--op', choices=('square', *SGEMM_OPS, *REDUCE_OPS), default='matmul'
    )
    parser.add_argument('--size', type=int, default=1000, help='Matrix/vector size')
    parser.add_argument(
        '--profile-range',
        action='store_true',
        help='Bracket measured iterations with cudaProfilerStart/Stop',
    )
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
        benchmark(my_cuda_kernels.square, args.iters, args.warmup, args.profile_range, inp)
    elif args.op in SGEMM_OPS:
        m1 = torch.randn(n, n, device="cuda")
        m2 = torch.randn(n, n, device="cuda")
        print(f"\nBenchmarking {args.op} kernel on input size ({n}, {n})")
        benchmark(
            getattr(my_cuda_kernels, args.op),
            args.iters,
            args.warmup,
            args.profile_range,
            m1,
            m2,
        )
    else:
        inp = torch.randn(n, device='cuda')
        print(f"\nBenchmarking {args.op} on input size ({n},)")
        benchmark(
            getattr(my_cuda_kernels, args.op),
            args.iters,
            args.warmup,
            args.profile_range,
            inp,
        )
