# CUDA + PyTorch template

A clean and simple template for developing CUDA C++ kernels and testing them in Python/PyTorch 🚀🚀.

Tested on Ubuntu 24.04.

## Structure

```
.
├── README.md
├── benchmark.py   // use this script to benchmark your kernels
├── csrc           // C/C++ CUDA files
│   ├── api.cpp    // define the Python interface here
│   ├── matmul.cu  // a sample CUDA kernel
│   └── square.cu  // another sample CUDA kernel
├── requirements.txt
├── setup.py       // your code is compiled through this script
└── tests          // test the correctness of your kernels here
    ├── test_matmul.py
    └── test_square.py
```

## 💻 Installation with uv

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and the
virtual environment. The local development environment currently uses:

```text
uv 0.12.3
Python 3.14.7
PyTorch 2.13.0+cu130
CUDA Toolkit 13.3 (/usr/local/cuda)
```

An Nvidia GPU and a working driver are required to run the kernels. Building
the extension also requires a CUDA Toolkit containing `nvcc`; Python packages
installed by uv do not replace the system compiler selected by `CUDA_HOME`.

Create the same local environment from the repository root:

```shell
uv python install 3.14
uv venv --python 3.14
source .venv/bin/activate

uv pip install "torch==2.13.0"
uv pip install -r requirements.txt
TORCH_CUDA_ARCH_LIST=12.0 uv pip install -e . --no-build-isolation
```

`--no-build-isolation` is required because `setup.py` imports PyTorch while
building the CUDA extension. The uv environment intentionally does not need a
standalone `pip` installation; use `uv pip` to manage its packages.

Check that uv, PyTorch and the CUDA compiler resolve to the expected versions:

```shell
uv --version
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
echo "${CUDA_HOME:-/usr/local/cuda}"
"${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version
uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

## 🔥 How to run

This repo contains two sample CUDA kernels that you can use as a starting point: `csrc/square.cu` and `csrc/matmul.cu`.

The first step is to compile the kernels in editable mode:

```shell
TORCH_CUDA_ARCH_LIST=12.0 uv run python setup.py build_ext --inplace --force
```

This forces all CUDA sources to be rebuilt for the local RTX 5060 Ti (`sm_120`),
so an older binary compiled for another GPU is not reused. Re-run it after
changing CUDA sources. On another GPU, replace `12.0` with the value printed by
`torch.cuda.get_device_capability()` (for example, use `8.0` for an A100).

Once you've compiled, you can use the provided scripts to:

1. Test your kernels:

```shell
uv run python -m pytest -v -s
```

test a single kernel:

```shell
uv run python -m pytest tests/test_matmul.py -v -s
```

test a single function specifically:

```shell
uv run python -m pytest tests/test_square.py::test_square_kernel -v -s
```

部分算子使用了参数化测试。如果只想运行某个尺寸，可以先查看完整的 pytest case ID：

```shell
uv run python -m pytest tests/test_matmul.py --collect-only -q
```

```shell
tests/test_matmul.py::test_matmul_kernel[2-2-20]
tests/test_matmul.py::test_matmul_kernel[20-2-2]
tests/test_matmul.py::test_matmul_kernel[3-7-11]
tests/test_matmul.py::test_matmul_kernel[1024-1024-1024]
tests/test_matmul.py::test_matmul_kernel[1000-10-10]
tests/test_matmul.py::test_matmul_kernel[10-10-1000]
tests/test_matmul.py::test_matmul_kernel[999-999-999]
```

然后单独运行一个 case。因为方括号在 shell 中有特殊含义，建议把完整 node ID 用引号包起来：

```shell
uv run python -m pytest \
"tests/test_matmul.py::test_matmul_kernel[1024-1024-1024]" \
-v -s
```

2. Benchmark your kernels:

```shell
uv run python benchmark.py --op matmul --size 1000 --warmup 10 --iters 1000
```

These two commands should work out of the box for the two kernels mentioned above.

That's it! Now, you can start hacking away and create your own CUDA kernels.

## Detailed profiling

Once you start writing more serious kernels, you probably want to do more precise benchmarking.  The `benchmark.py` script is a simple script for timing your kernels, but it is not as precise as using a profiler. If you want to get detailed information about the performance bottlenecks of your kernels, consider using the `ncu` profiler. For example:

```shell
sudo /usr/local/cuda/bin/ncu \
    --kernel-name square_kernel \
    --launch-skip 10 \
    --launch-count 1 \
    "$PWD/.venv/bin/python" benchmark.py \
    --op square --size 1000000 --warmup 10 --iters 1
```

For timeline-level analysis:

```shell
/usr/local/cuda/bin/nsys profile \
    --trace=cuda,nvtx,osrt \
    --force-overwrite=true \
    --output=profile \
    "$PWD/.venv/bin/python" benchmark.py \
    --op matmul --size 1000 --warmup 10 --iters 100
```

`--kernel-name` filters the report to the selected kernel; `--launch-skip` avoids
collecting the warmup launches. `--force-overwrite=true` allows Nsight Systems
to replace an existing `profile.nsys-rep` instead of failing because the output
file already exists.

Nsight Compute needs access to hardware performance counters. On this machine,
running it without `sudo` produces `ERR_NVGPUCTRPERM`; Nsight Systems does not
need performance-counter permission for the CUDA timeline above.
    

## Running ncu profiler on a cloud GPU instance

The Nsight profiler (`ncu`) is a very useful tool to profile CUDA kernels.  However, it will not run out of the box on cloud GPUs. If you run `ncu`, you might get an output like this:

```shell
$ ncu ./benchmark
==PROF== Connected to process 2258 (/mnt/tobias/benchmark)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```
 

To fix this, run `ncu` with `sudo`. The command above uses absolute paths for
both `ncu` and the uv-created Python interpreter, so it does not depend on the
restricted `sudo` PATH. If an administrator has enabled performance counters
for normal users, the `sudo` prefix can be removed.

```bash
/usr/local/cuda/bin/ncu --version
/usr/local/cuda/bin/nsys --version
```

## TODO

- [ ] Docker
