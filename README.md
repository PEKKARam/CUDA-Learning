# CUDA Learning

这是一个面向 CUDA kernel 学习、源码调试和性能优化的工程。项目采用“共享 CUDA
核心 + standalone CMake 工具 + 可选 PyTorch binding”的结构：kernel 和 launcher
只实现一次，既能由纯 CUDA 可执行文件直接测试和 profile，也能作为 PyTorch 自定义
算子调用。

当前包含：

- square kernel；
- 多个 sum-reduction 优化版本；
- row-major naive SGEMM；
- cuBLAS SGEMM 正确性与性能 baseline；
- CTest、Python/pytest、CUDA Event benchmark、NVTX 和 CUDA Profiler API；
- `cuda-gdb`、Nsight Systems、Nsight Compute、Compute Sanitizer 使用入口。

## 架构

```text
.
├── CMakeLists.txt
├── include/cuda_learning/
│   ├── kernels.h              # 与框架无关的 kernel/launcher 公共接口
│   ├── cuda_check.h           # standalone CUDA 错误检查
│   └── standalone_utils.h     # 显存、随机输入、校验、CUDA Event/NVTX 计时
├── src/
│   ├── square/square.cu
│   ├── reduce/reduce.cu
│   └── sgemm/sgemm.cu         # SGEMM kernel 与实现 registry
├── tests/cpp/                 # 不依赖 PyTorch 的 CTest 正确性测试
├── benchmarks/                # standalone benchmark/profile 入口
│   ├── benchmark_square.cu
│   ├── benchmark_reduce.cu
│   └── benchmark_sgemm.cu     # 自定义 SGEMM 与 cuBLAS 对比
├── bindings/pytorch/api.cpp   # 可选的 PyTorch 薄适配层
├── tests/                     # PyTorch/pytest 正确性测试
├── setup.py                   # 构建 PyTorch extension
└── benchmark.py               # 从 Python 调用 kernel 的辅助 benchmark
```

核心依赖方向为：

```text
src CUDA kernels
      |
      +--> CMake CTest
      +--> standalone benchmarks --> cuBLAS baseline
      +--> PyTorch binding --> pytest / Python benchmark
```

`src/` 不包含 Torch/c10 头文件。launcher 接受显式 `cudaStream_t`，因此 standalone
程序和 PyTorch 当前 stream 可以安全复用同一实现。

## 环境

当前本机验证环境：

```text
Ubuntu 24.04
Python 3.14.7
PyTorch 2.13.0+cu130
CUDA Toolkit 13.3 (/usr/local/cuda)
RTX 5060 Ti, compute capability 12.0
CMake 3.28.3
```

创建 Python 环境并安装 Ninja/pytest：

```shell
uv python install 3.14
uv venv --python 3.14
source .venv/bin/activate
uv pip install "torch==2.13.0"
uv pip install -r requirements.txt
```

检查工具链和 GPU：

```shell
cmake --version
ninja --version
"${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version
uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

其他 GPU 应把后续命令中的 `120`/`12.0` 替换为实际 compute capability，例如 A100
使用 CMake architecture `80`，PyTorch 使用 `TORCH_CUDA_ARCH_LIST=8.0`。

## CMake Standalone 工作流

这是学习 kernel、跑 cuBLAS 对比和使用 NVIDIA 调试工具时的首选入口。它不启动
Python，也不加载 PyTorch，因此时间线更干净，`cuda-gdb` 断点也更直接。

### 构建模式

| CMake 模式 | CUDA 参数 | 用途 |
| --- | --- | --- |
| `Release` | `-O3` | 最终性能复测 |
| `Profile` | `-O3 -lineinfo` | `nsys` / `ncu`，保留优化与源码行信息 |
| `Debug` | `-O0 -G` | `cuda-gdb` 源码断点和变量检查 |

Profile 构建：

```shell
source .venv/bin/activate
cmake -S . -B build/cmake-profile -G Ninja \
    -DCMAKE_BUILD_TYPE=Profile \
    -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/cmake-profile -j
```

Debug 和 Release 使用独立目录，避免错误复用带 `-G` 的目标文件：

```shell
cmake -S . -B build/cmake-debug -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/cmake-debug -j

cmake -S . -B build/cmake-release -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/cmake-release -j
```

`Debug/-G` 会改变 device 优化、寄存器分配、occupancy 和耗时，不能用它评价性能。

### C++ 正确性测试

```shell
ctest --test-dir build/cmake-profile --output-on-failure

# 也可以直接运行单个程序
build/cmake-profile/test_square
build/cmake-profile/test_reduce
build/cmake-profile/test_sgemm
```

SGEMM C++ 测试会自动遍历 registry 中的全部自定义实现，并与 CPU reference 比较。
新增 SGEMM 版本后不需要为它复制一套测试主程序。

### Standalone benchmark

```shell
build/cmake-profile/benchmark_square \
    --size 1048576 --warmup 10 --iters 100

build/cmake-profile/benchmark_reduce \
    --impl all --size 1048576 --warmup 10 --iters 100

build/cmake-profile/benchmark_reduce \
    --impl shuffle --size 1048576 --warmup 10 --iters 100

build/cmake-profile/benchmark_sgemm \
    --impl all --m 1024 --n 1024 --k 1024 --warmup 10 --iters 100
```

SGEMM 支持：

```shell
--impl naive     # 当前手写 baseline
--impl cublas    # cuBLAS FP32 baseline
--impl all       # 打印所有已注册手写版本和 cuBLAS
```

输出同时包含平均时间和 GFLOP/s。程序在计时前会先比较 naive 与 cuBLAS 的结果；完整
的全部实现/多 shape 正确性由 `test_sgemm` 负责。

cuBLAS 默认使用 column-major。本项目通过计算 `C^T = B^T * A^T` 实现与自定义 kernel
相同的 row-major `C[M,N] = A[M,K] * B[K,N]`，并使用
`CUBLAS_PEDANTIC_MATH` 作为严格 FP32 baseline。后续研究 TF32/Tensor Core 时，应新增
明确命名的 cuBLAS 配置，不要悄悄改变 baseline 的 math mode。

## NVIDIA 调试与性能分析

| 问题 | 工具 | 构建目录 |
| --- | --- | --- |
| CPU/CUDA 时间线、launch 间空洞、同步 | Nsight Systems | `cmake-profile` |
| 单个 kernel 的吞吐、访存、occupancy、stall | Nsight Compute | `cmake-profile` |
| `.cu` 断点、CUDA thread、局部变量 | `cuda-gdb` | `cmake-debug` |
| 越界、race、未初始化访问、同步错误 | Compute Sanitizer | `cmake-debug` 或 `cmake-profile` |

benchmark 的 `--profile` 会用 `cudaProfilerStart/Stop` 只包住正式计时循环，并写入
NVTX range。使用 `--profile` 时必须选择一个具体 `--impl`，避免报告混入多个实现。

### Nsight Systems

```shell
mkdir -p reports/sgemm
"${CUDA_HOME:-/usr/local/cuda}/bin/nsys" profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output=reports/sgemm/naive-1024 \
    build/cmake-profile/benchmark_sgemm \
    --impl naive --m 1024 --n 1024 --k 1024 \
    --warmup 10 --iters 20 --profile

"${CUDA_HOME:-/usr/local/cuda}/bin/nsys" stats \
    reports/sgemm/naive-1024.nsys-rep
```

### Nsight Compute

`--set full` 会多次重放 kernel，通常只采集一次正式迭代：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/ncu" \
    --profile-from-start=off \
    --set=full \
    --kernel-name-base=demangled \
    --kernel-name='regex:sgemm_naive_kernel' \
    --export=reports/sgemm/naive-1024 \
    build/cmake-profile/benchmark_sgemm \
    --impl naive --m 1024 --n 1024 --k 1024 \
    --warmup 10 --iters 1 --profile
```

如果出现 `ERR_NVGPUCTRPERM`，说明普通用户无权访问 GPU performance counters。
可以临时在绝对路径的 `ncu` 命令前使用 `sudo`，或由管理员按 NVIDIA 驱动安全策略
开放 counters。Nsight Systems 的普通 CUDA timeline 通常不需要此权限。

### cuda-gdb 与 VS Code

```shell
CUDA_LAUNCH_BLOCKING=1 \
"${CUDA_HOME:-/usr/local/cuda}/bin/cuda-gdb" --args \
build/cmake-debug/benchmark_sgemm \
--impl naive --m 128 --n 128 --k 128 --warmup 0 --iters 1
```

进入调试器后：

```text
set breakpoint pending on
break sgemm_naive_kernel
run
info cuda kernels
info cuda threads
print row
print col
print sum
```

`.vscode/launch.json` 提供：

- `CUDA: Debug standalone SGEMM`：推荐，直接调试 CMake 可执行文件；
- `CUDA: Debug PyTorch SGEMM`：用于检查 binding、Tensor/stream 集成问题。

需要安装 NVIDIA Nsight Visual Studio Code Edition，并先完成对应 Debug 构建。

### Compute Sanitizer

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer" --tool memcheck \
    build/cmake-debug/test_sgemm

"${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer" --tool racecheck \
    build/cmake-debug/benchmark_reduce \
    --impl no_bankconflict --size 65536 --warmup 0 --iters 1
```

## 可选 PyTorch 自定义算子

PyTorch binding 只负责 Tensor 校验、输出分配、取得当前 CUDA stream 和 Python 注册，
不会复制 kernel。当前 Python API：

```text
square
matmul             # sgemm_naive 的兼容别名
sgemm_naive
sgemm_baseline     # sgemm_naive 的兼容别名
reduce_*
```

构建 profile extension：

```shell
CUDA_BUILD_MODE=profile TORCH_CUDA_ARCH_LIST=12.0 \
uv run python setup.py build_ext --inplace --force
```

`CUDA_BUILD_MODE` 支持：

| 模式 | CUDA 参数 | 用途 |
| --- | --- | --- |
| `release` | `-O2` | Python 最终复测 |
| `profile` | `-O2 --generate-line-info` | 从 Python 入口使用 ncu/nsys |
| `debug` | `-O0 -g -G` | 调试 PyTorch binding 与 kernel |

运行 Python 测试和辅助 benchmark：

```shell
uv run python -m pytest -v -s
uv run python -m pytest tests/test_sgemm.py -v -s

uv run python benchmark.py \
    --op sgemm_naive --size 1024 --warmup 10 --iters 100
```

纯 kernel 性能结论应优先使用 standalone benchmark。Python 入口主要用于验证
PyTorch Tensor、dtype、shape、device、stream 和框架集成语义。

## 新增 SGEMM 优化版本

建议按学习阶段使用明确名称，例如：

```text
naive
coalesced
shared_tiled
block_tiled_1d
block_tiled_2d
vectorized
double_buffered
wmma
```

新增 `shared_tiled` 的最小流程：

1. 新建 `src/sgemm/sgemm_shared_tiled.cu`，实现 kernel 和符合 `SgemmLauncher`
   签名的 host launcher。CMake 与 `setup.py` 都会自动发现新的 `.cu`。
2. 在 `src/sgemm/sgemm.cu` 中声明 launcher，并向 `sgemm_implementations()` registry
   增加一项 `{"shared_tiled", launch_sgemm_shared_tiled}`。
3. 重新构建后，`test_sgemm` 会自动对新版本运行全部 C++ shape；
   `benchmark_sgemm --impl all` 和 `--impl shared_tiled` 也会自动可用。
4. 只有确实需要从 Torch 调用该版本时，才在 `bindings/pytorch/api.cpp` 添加一个薄
   wrapper/注册项，并在 `tests/test_sgemm.py` 增加该 Python 名称。

因此，一个只用于 CUDA 学习和性能对比的新 SGEMM 版本通常只修改两个位置：新的
`.cu` 实现文件和一行 registry；不需要复制 `main()`、计时、cuBLAS、参数解析或测试
框架。

每个版本至少检查：

- 非 tile 整除的 M/N/K；
- 很小矩阵和非方阵；
- shared-memory 边界和同步；
- 向量化 load/store 的对齐与尾部；
- 与 cuBLAS/CPU reference 的误差；
- warmup、固定输入、相同 stream/math mode 下的公平计时；
- `compute-sanitizer`、`nsys`、`ncu` 后再做最终 Release 复测。

## 新增其他算子

1. 在 `src/<op>/` 实现 framework-agnostic kernel/launcher，并把公共接口放到
   `include/cuda_learning/kernels.h` 或独立 op header。
2. 添加 `tests/cpp/test_<op>.cu`；若新增独立 target，在 `CMakeLists.txt` 的 test 列表
   加入 op 名称。
3. 添加 `benchmarks/benchmark_<op>.cu` 和对应 benchmark target，复用
   `standalone_utils.h`，不要在每个 kernel 文件中复制 `main()`。
4. 选择合适 reference：SGEMM 用 cuBLAS，reduce 可用 CPU/CUB，深度学习算子可用
   cuDNN 或 PyTorch。不是所有算子都适合用 cuBLAS。
5. 需要 Torch 集成时再添加 binding 和 pytest；纯 CUDA 学习版本不必同步暴露 Python。

推荐顺序：CTest 正确性 -> Compute Sanitizer -> Nsight Systems -> Nsight Compute ->
Release benchmark -> 可选 PyTorch 集成。

## 构建提示

- `build/`、`.so`、profiler 原始报告和 `reports/` 默认不会提交到 Git。
- 当前 CUDA Toolkit 13.3 与 PyTorch cu130 存在 minor-version mismatch 警告；本机
  CMake、extension、CTest、pytest 和 CUDA 运行均已验证通过。
- 切换 CMake build type 使用不同 build 目录；切换 `CUDA_BUILD_MODE` 时为 extension
  加 `--force`，避免加载旧目标文件。
