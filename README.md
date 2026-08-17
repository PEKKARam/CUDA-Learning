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

This repo contains square, matmul and several reduction CUDA kernels that can be
used as starting points.

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

test a particular version of the kernel:

```shell
uv run python -m pytest tests/test_reduce.py -k reduce_unroll_last_warp -v -s
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

1. Benchmark your kernels:

```shell
uv run python benchmark.py --op matmul --size 1000 --warmup 10 --iters 1000
```

These two commands should work out of the box for the two kernels mentioned above.

That's it! Now, you can start hacking away and create your own CUDA kernels.

## CUDA 调试与性能分析

`ncu` 和 `nsys` 严格来说是性能分析器，不是源码断点调试器。建议按问题选择工具：

| 目标 | 工具 | 本项目应使用的构建模式 |
| --- | --- | --- |
| 看 CPU/CUDA 时间线、kernel launch、同步和空洞 | Nsight Systems (`nsys`) | `profile` |
| 看单个 kernel 的 occupancy、访存、吞吐和 stall | Nsight Compute (`ncu`) | `profile` |
| 在 `.cu` 中下断点、查看线程和变量 | `cuda-gdb` / VS Code CUDA Debugger | `debug` |
| 查越界、race、未初始化访问和同步错误 | `compute-sanitizer` | `debug` 或 `profile` |

先确认 CUDA Toolkit 中包含这些工具：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/ncu" --version
"${CUDA_HOME:-/usr/local/cuda}/bin/nsys" --version
"${CUDA_HOME:-/usr/local/cuda}/bin/cuda-gdb" --version
"${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer" --version
```

### 三种编译模式

`setup.py` 通过 `CUDA_BUILD_MODE` 提供三种模式：

| 模式 | 主要编译参数 | 用途 |
| --- | --- | --- |
| `release`（默认） | `-O2` | 日常运行和最终性能复测 |
| `profile` | `-O2 --generate-line-info` | `ncu/nsys`，保留优化并提供源码行关联 |
| `debug` | host `-O0 -g`，device `-O0 -g -G` | `cuda-gdb` 和正确性调试 |

切换模式时必须强制重编译，避免继续加载旧的 `.so`：

```shell
# ncu / nsys
CUDA_BUILD_MODE=profile TORCH_CUDA_ARCH_LIST=12.0 \
uv run python setup.py build_ext --inplace --force

# cuda-gdb；不要用这个构建结果评价性能
CUDA_BUILD_MODE=debug TORCH_CUDA_ARCH_LIST=12.0 \
uv run python setup.py build_ext --inplace --force
```

其他 GPU 应把 `12.0` 替换为实际 compute capability。`-G` 会禁用或改变大量
device 优化，并影响寄存器、occupancy 和耗时，因此 `debug` 模式下的 profiler
数据不能作为优化依据。

### Benchmark 捕获区间

`benchmark.py` 会写入 `warmup` 和 `benchmark` 两个 NVTX range。加上
`--profile-range` 后，还会用 `cudaProfilerStart/Stop` 只包围正式测量循环，避免
CUDA context 初始化、张量创建和 warmup 污染报告。当前支持：

```shell
uv run python benchmark.py --op square --size 1000000 --warmup 10 --iters 100
uv run python benchmark.py --op matmul --size 1024 --warmup 10 --iters 20
uv run python benchmark.py --op reduce_shuffle --size 1048576 --warmup 10 --iters 100
```

### 使用 Nsight Systems

先用 `nsys` 定位时间花在哪里，再决定要用 `ncu` 深挖哪个 kernel：

```shell
mkdir -p reports
"${CUDA_HOME:-/usr/local/cuda}/bin/nsys" profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output=reports/matmul \
    "$PWD/.venv/bin/python" benchmark.py \
    --op matmul --size 1024 --warmup 10 --iters 20 --profile-range
```

输出为 `reports/matmul.nsys-rep`。可以用 GUI 打开，或先在终端查看摘要：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/nsys" stats reports/matmul.nsys-rep
```

重点检查 CUDA API 调用、kernel 持续时间、launch 间空洞、意外的
`cudaDeviceSynchronize`/内存拷贝，以及 NVTX `benchmark` 区间。

### 使用 Nsight Compute

用 `ncu` 采集单个正式迭代。`--set full` 会多次重放 kernel，开销很大，不要一开始
就对数百次迭代使用：

```shell
mkdir -p reports
"${CUDA_HOME:-/usr/local/cuda}/bin/ncu" \
    --profile-from-start=off \
    --target-processes=all \
    --set=full \
    --kernel-name-base=demangled \
    --kernel-name='regex:square_kernel' \
    --export=reports/square \
    "$PWD/.venv/bin/python" benchmark.py \
    --op square --size 1000000 --warmup 10 --iters 1 --profile-range
```

输出为 `reports/square.ncu-rep`，可在 Nsight Compute GUI 中查看，也可导入 CLI：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/ncu" --import reports/square.ncu-rep
```

分析顺序建议为：先看 `SpeedOfLight` 判断计算或带宽利用率，再看 occupancy、memory
workload 和 warp stall。模板 kernel（例如 `reduce_shuffle<4, 256>`）使用 demangled
名称；若过滤不到，先去掉 `--kernel-name` 生成一次报告，从输出中复制实际名称。

若出现 `ERR_NVGPUCTRPERM`，表示普通用户无权访问 GPU performance counters。
本机临时运行可在上面的绝对路径命令前加 `sudo`；服务器或云实例上应由管理员按
NVIDIA 驱动安全策略启用 counters。`sudo` 不是 `nsys` CUDA timeline 的常规要求。

### 使用 cuda-gdb 或 VS Code 断点

先用 `debug` 模式重编译，然后启动：

```shell
CUDA_LAUNCH_BLOCKING=1 \
"${CUDA_HOME:-/usr/local/cuda}/bin/cuda-gdb" --args \
"$PWD/.venv/bin/python" benchmark.py \
--op square --size 1024 --warmup 0 --iters 1
```

进入 `cuda-gdb` 后可以执行：

```text
set breakpoint pending on
break square_kernel
run
info cuda kernels
info cuda threads
print i
print inp[i]
```

仓库的 `.vscode/launch.json` 已配置为用 `.venv/bin/python` 启动同一个 square case。
安装 NVIDIA Nsight Visual Studio Code Edition 后，先用 `debug` 模式构建，在
`csrc/square.cu` 中下断点，再选择 `CUDA: Debug square kernel`。调试其他算子时
修改配置中的 `--op`、尺寸和断点即可。

### 使用 compute-sanitizer

断点前建议先排除常见内存和同步错误：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer" --tool memcheck \
"$PWD/.venv/bin/python" benchmark.py \
--op square --size 1024 --warmup 0 --iters 1

"${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer" --tool racecheck \
"$PWD/.venv/bin/python" benchmark.py \
--op reduce_no_bankconflict --size 65536 --warmup 0 --iters 1
```

遇到异步报错时，也可先用 `CUDA_LAUNCH_BLOCKING=1` 运行对应 pytest case，让错误更
靠近真实 launch 位置。它会改变执行时序和性能，只用于定位错误。

## 新增算子检查清单

新增 CUDA 算子时同步维护以下位置：

1. 在 `csrc/<op>.cu` 实现 kernel 和 launcher。`setup.py` 会递归发现 `csrc` 下的
   `.cu/.cpp`，通常不需要手动增加 sources。
2. Launcher 使用 `c10::cuda::getCurrentCUDAStream()`，launch 后调用
   `C10_CUDA_KERNEL_LAUNCH_CHECK()`，避免破坏 PyTorch stream 语义并尽早报告错误。
3. 在 `csrc/api.cpp` 声明 launcher、实现 Tensor wrapper，检查 device、dtype、shape、
   contiguous 和同设备约束，并在 `PYBIND11_MODULE` 中注册 Python 名称。
4. 在 `tests/test_<op>.py` 添加与 PyTorch reference 的正确性比较，覆盖小尺寸、非整除
   block 的尺寸、空张量（若支持）、边界 shape 和错误输入。
5. 在 `benchmark.py` 的 `--op` choices 中加入名字，并添加有代表性的输入构造。若有
   多个实现版本，可像 reduce 一样统一命名，方便横向 profile。
6. 用 `--force` 重编译，依次运行单元测试、`compute-sanitizer`、`nsys` 和 `ncu`。
   保存报告时把算子、shape、版本写进文件名，例如
   `reports/reduce_shuffle-n1048576-v2.ncu-rep`。
7. kernel 名称尽量稳定且有辨识度，方便 `ncu --kernel-name` 过滤；修改 launch 参数或
   算法后，同时更新 benchmark 默认 case、测试边界和 README 中的示例（如适用）。

一套推荐流程是：`release` 下先通过测试，`compute-sanitizer` 查正确性，`nsys` 找时间线
热点，`profile` 模式用 `ncu` 分析热点 kernel，优化后回到 `release` 模式做最终复测。

## TODO

- [ ] Docker
