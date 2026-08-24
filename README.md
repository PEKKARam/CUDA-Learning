# CUDA Learning

这个项目用于学习 CUDA kernel、SGEMM 优化、cuBLAS 对比、CUDA 调试以及可选的
PyTorch 自定义算子。CUDA kernel 只实现一次，由 CMake standalone 程序和 PyTorch
binding 共同复用。

```text
src/                         framework-agnostic CUDA kernel/launcher
include/cuda_learning/       公共接口、CUDA 检查、测试/计时工具
tests/cpp/                   CMake/CTest 正确性测试
benchmarks/                  standalone benchmark、NVTX、cuBLAS 对比
bindings/pytorch/api.cpp     可选 PyTorch 薄封装
tests/                       pytest 测试
setup.py                     构建 PyTorch extension
benchmark.py                 Python benchmark
```

当前 SGEMM 实现由 `src/sgemm/sgemm.cu` 的 registry 管理，例如：

```text
naive
shared_memory
multi_workload
```

## 环境和构建

需要 CUDA Toolkit、可用 NVIDIA GPU、CMake、Ninja，以及可选的 uv/PyTorch 环境。
当前机器使用 CUDA 13.3、PyTorch 2.13.0+cu130、RTX 5060 Ti (`sm_120`)。

```shell
uv venv --python 3.14
source .venv/bin/activate
uv pip install "torch==2.13.0"
uv pip install -r requirements.txt

cmake --version
"${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version
uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

其他 GPU 将下面的 `120` 替换成对应的 CUDA architecture，例如 A100 使用 `80`。

项目提供三种 CMake 构建模式：

| 模式      | 主要参数             | 用途                        |
| --------- | -------------------- | --------------------------- |
| `Debug`   | CUDA `-O0 -G`        | `cuda-gdb`/VS Code 单步调试 |
| `Profile` | CUDA `-O3 -lineinfo` | `nsys`/`ncu` 性能分析       |
| `Release` | CUDA `-O3`           | 最终性能复测                |

建议为不同模式使用不同 build 目录：

```shell
# 性能分析
cmake -S . -B build/cmake-profile -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" \
    -DCMAKE_BUILD_TYPE=Profile \
    -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/cmake-profile --clean-first -j

# 源码调试
cmake -S . -B build/cmake-debug -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/cmake-debug --clean-first -j
```

`Debug/-G` 会改变寄存器、occupancy 和执行时间，不能用来做性能结论。

## 正确性测试

### CMake/CTest

```shell
ctest --test-dir build/cmake-profile --output-on-failure

# 查看实时测试过程、shape 和实现名称
ctest --test-dir build/cmake-profile \
    -R '^sgemm$' -V

# 直接运行
build/cmake-profile/test_square
build/cmake-profile/test_reduce
build/cmake-profile/test_sgemm
```

`test_sgemm` 会打印每个 shape 和 registry 实现，例如：

```text
SGEMM test plan: 10 shapes, 3 registered implementations
[shape 8/10] M=32, N=32, K=32
  [impl 1/3] naive ... PASS
  [impl 2/3] shared_memory ... PASS
  [impl 3/3] multi_workload ... PASS
```

当前测试覆盖小矩阵、非方阵、tile 整除和非整除的 M/N/K。新增实现注册后会自动参与
所有 SGEMM shape 测试。

### PyTorch/pytest

```shell
CUDA_BUILD_MODE=profile TORCH_CUDA_ARCH_LIST=12.0 \
uv run python setup.py build_ext --inplace --force

uv run python -m pytest -v -s
uv run python -m pytest tests/test_sgemm.py -v -s
```

## Standalone benchmark

```shell
build/cmake-profile/benchmark_square \
    --size 1048576 --warmup 10 --iters 100

build/cmake-profile/benchmark_reduce \
    --impl shuffle --size 1048576 --warmup 10 --iters 100

build/cmake-profile/benchmark_sgemm \
    --impl all --m 1024 --n 1024 --k 1024 \
    --warmup 10 --iters 100
```

SGEMM 支持：

```shell
--impl naive
--impl shared_memory
--impl multi_workload
--impl cublas
--impl all
```

benchmark 会在正式计时前验证每个被选中的实现；错误实现不会输出性能结果。计时使用
CUDA Event，正式区间不包含输入分配、warmup 和 correctness copy。cuBLAS 使用 row-major
适配和 `CUBLAS_PEDANTIC_MATH` 作为 FP32 baseline。

## CUDA 工具

### Compute Sanitizer

先查越界和 race，再单步调试：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer" \
    --tool memcheck \
    build/cmake-debug/benchmark_sgemm \
    --impl multi_workload --m 31 --n 29 --k 17 \
    --warmup 0 --iters 1

"${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer" \
    --tool racecheck \
    build/cmake-debug/benchmark_sgemm \
    --impl multi_workload --m 32 --n 32 --k 32 \
    --warmup 0 --iters 1
```

### `cuda-gdb`/VS Code

使用 Debug build，并将输入缩小到一次 launch：

```shell
CUDA_LAUNCH_BLOCKING=1 \
"${CUDA_HOME:-/usr/local/cuda}/bin/cuda-gdb" --args \
build/cmake-debug/benchmark_sgemm \
--impl multi_workload --m 32 --n 32 --k 32 \
--warmup 0 --iters 1
```

进入 `cuda-gdb` 后：

```text
set breakpoint pending on
break sgemm_multi_workload
run
info cuda kernels
info cuda threads
print threadIdx.x
print threadIdx.y
print row
print col
print tile
print sum[0][0]
```

`.vscode/launch.json` 提供 `CUDA: Debug standalone SGEMM` 和 PyTorch 调试配置。前者
适合调试 kernel；后者只在需要检查 Tensor/stream/binding 时使用。

### Nsight Systems

`benchmark_sgemm --profile` 使用 `cudaProfilerStart/Stop` 和 NVTX，只捕获正式循环：

```shell
mkdir -p reports/sgemm
"${CUDA_HOME:-/usr/local/cuda}/bin/nsys" profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output=reports/sgemm/multi-workload \
    build/cmake-profile/benchmark_sgemm \
    --impl multi_workload --m 1024 --n 1024 --k 1024 \
    --warmup 10 --iters 20 --profile

"${CUDA_HOME:-/usr/local/cuda}/bin/nsys" stats \
    reports/sgemm/multi-workload.nsys-rep
```

### Nsight Compute

只 profile 一个实现和一次正式迭代：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/ncu" \
    --profile-from-start=off \
    --set=full \
    --kernel-name-base=demangled \
    --kernel-name='regex:sgemm_multi_workload' \
    --export=reports/sgemm/multi-workload \
    build/cmake-profile/benchmark_sgemm \
    --impl multi_workload --m 1024 --n 1024 --k 1024 \
    --warmup 10 --iters 1 --profile
```

重点看 occupancy、register/shared-memory 使用、global/shared memory throughput、bank
conflict 和 warp stall。若出现 `ERR_NVGPUCTRPERM`，需要 `sudo` 或管理员开放 GPU
performance counters；Nsight Systems CUDA timeline 通常不需要该权限。

## Nsight 学习路线（先照做，不要一次看完所有面板）

Nsight Systems（`nsys`）和 Nsight Compute（`ncu`）解决的是两类不同问题：

```text
nsys  -> 这一段程序的时间花在哪里？kernel 是否串行？是否被 memcpy/CPU 空转拖慢？
ncu   -> 某一个 kernel 为什么慢？线程、内存、occupancy、stall 的瓶颈是什么？
```

建议固定使用下面的顺序：

```text
正确性测试 -> Compute Sanitizer -> nsys 时间线 -> ncu 单 kernel 指标 -> Release 复测
```

不要用 `Debug/-G`、`CUDA_LAUNCH_BLOCKING=1`、Compute Sanitizer 或 cuda-gdb 的时间
作为性能结论；这些工具会改变 kernel 的执行速度。

### 第一步：准备 Profile 构建

`Profile` 模式保留优化，同时加入源码行号，适合 nsys/ncu；`Debug` 模式带 `-G`，只
适合定位错误和 cuda-gdb 单步。每次切换模式使用不同 build 目录，避免旧 object 混用：

```shell
cmake -S . -B build/cmake-profile -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" \
    -DCMAKE_BUILD_TYPE=Profile \
    -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/cmake-profile --clean-first -j
```

确认目标 GPU 架构：

```shell
uv run python -c \
    "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

RTX 5060 Ti 使用 `120`；其他 GPU 将 `120` 换成对应的 compute capability。

### 第二步：先用 Compute Sanitizer 排错

只使用一个实现、一个小输入和一次迭代，错误信息最容易对应到源码：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer" \
    --tool memcheck \
    build/cmake-debug/benchmark_reduce \
    --impl shuffle --size 1003 --warmup 0 --iters 1
```

SGEMM：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer" \
    --tool memcheck \
    build/cmake-debug/benchmark_sgemm \
    --impl multi_workload --m 31 --n 29 --k 17 \
    --warmup 0 --iters 1
```

`memcheck` 主要查越界和非法地址，`racecheck` 主要查 shared memory/data race：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer" \
    --tool racecheck \
    build/cmake-debug/benchmark_reduce \
    --impl shuffle --size 1003 --warmup 0 --iters 1
```

如果 sanitizer 报错，先修 kernel，不要继续看性能指标。若使用 PyTorch benchmark，
可以加 `CUDA_LAUNCH_BLOCKING=1` 让错误更接近触发位置，但它只用于调试：

```shell
CUDA_LAUNCH_BLOCKING=1 uv run python benchmark.py \
    --op reduce_shuffle --size 1003 --warmup 0 --iters 1
```

### 第三步：用 nsys 看完整时间线

先 profile 一个 reduce 实现：

```shell
mkdir -p reports/reduce reports/sgemm
"${CUDA_HOME:-/usr/local/cuda}/bin/nsys" profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output=reports/reduce/shuffle \
    build/cmake-profile/benchmark_reduce \
    --impl shuffle --size 1048576 --warmup 10 --iters 20 --profile
```

再 profile 一个 SGEMM 实现：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/nsys" profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --force-overwrite=true \
    --output=reports/sgemm/multi-workload \
    build/cmake-profile/benchmark_sgemm \
    --impl multi_workload --m 1024 --n 1024 --k 1024 \
    --warmup 10 --iters 20 --profile
```

查看命令行统计：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/nsys" stats \
    reports/reduce/shuffle.nsys-rep
```

查看 GUI：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/nsys-ui" \
    reports/sgemm/multi-workload.nsys-rep
```

第一次只看三件事：

1. CUDA API 是否在等待 CPU 或 memcpy。
2. kernel launch 之间是否有不必要的空洞（gap）。
3. 同一个 kernel 的执行时间是否稳定，warmup 是否被排除。

不要先研究所有 CPU sampling、OS runtime 和 CUDA API 选项；当前 benchmark 已通过
`cudaProfilerStart/Stop` 和 NVTX 把正式循环标出来。

### 第四步：用 ncu 看单个 kernel

`ncu` 一次只分析一个实现和一个 kernel。先使用少量指标，不要一开始使用 full set：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/ncu" \
    --profile-from-start=off \
    --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --section SpeedOfLight \
    --kernel-name-base=demangled \
    --kernel-name='regex:reduce_shuffle' \
    --launch-skip 10 \
    --launch-count 1 \
    --export=reports/reduce/shuffle \
    build/cmake-profile/benchmark_reduce \
    --impl shuffle --size 1048576 --warmup 10 --iters 1 --profile
```

SGEMM 只替换 kernel 过滤条件和程序参数：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/ncu" \
    --profile-from-start=off \
    --set=full \
    --kernel-name-base=demangled \
    --kernel-name='regex:sgemm_multi_workload' \
    --launch-skip 10 --launch-count 1 \
    --export=reports/sgemm/multi-workload \
    build/cmake-profile/benchmark_sgemm \
    --impl multi_workload --m 1024 --n 1024 --k 1024 \
    --warmup 10 --iters 1 --profile
```

查看 GUI：

```shell
"${CUDA_HOME:-/usr/local/cuda}/bin/ncu-ui" \
    reports/reduce/shuffle.ncu-rep
```

打开 ncu 报表后按以下顺序看：

| 顺序 | 位置/指标 | 先问的问题 |
| --- | --- | --- |
| 1 | Launch Statistics | grid/block 是否是预期？register、shared memory 是否过大？ |
| 2 | Occupancy | 是否因为 register/shared memory 限制了 active warps？ |
| 3 | Speed of Light | Compute、DRAM、L2 哪一个接近瓶颈？ |
| 4 | Memory Workload | 是否有低效的 global load、低带宽或 bank conflict？ |
| 5 | Warp State Statistics | warp stall 主要来自 memory、barrier 还是执行依赖？ |

如果普通用户执行 ncu 出现 `ERR_NVGPUCTRPERM`，这是驱动性能计数器权限问题，不是
kernel 或命令参数错误。使用管理员允许的方式运行：

```shell
sudo "${CUDA_HOME:-/usr/local/cuda}/bin/ncu" ...
```

`nsys` 通常不需要该权限；不要为了消除 ncu 权限错误修改 kernel。

### nsys 和 ncu 的结果如何串起来

以 SGEMM 为例：

```text
nsys 发现 kernel 占总时间 95%，且没有 memcpy 空洞
  -> ncu 查看该 kernel 是 memory-bound 还是 compute-bound
  -> 若 DRAM throughput 高、SM busy 低：优化 global/shared memory 访问
  -> 若 occupancy 低：检查 register、shared memory、block size
  -> 若 barrier stall 高：检查 __syncthreads、tile pipeline、同步粒度
  -> 修改后重新跑 sanitizer、nsys、ncu，再用 Release benchmark 复测
```

reduce 通常重点看 memory throughput、atomic serialization 和 warp stall；SGEMM 通常
重点看 tensor/core 使用（如果适用）、global/shared memory throughput、occupancy、
register pressure、bank conflict 和 math pipeline 利用率。

## 新增算子/新版本时的固定清单

以后新增反量化、reduce 或 SGEMM 版本时，不需要修改 nsys/ncu 本身，但需要保证下面的
入口都同步：

1. **kernel/launcher**：在 `src/<op>/*.cu` 定义 kernel 和 launcher；launcher 使用
   PyTorch current stream 或公共 `cudaStream_t stream`，并执行 kernel launch check。
2. **公共接口/registry**：在 `include/cuda_learning/kernels.h` 声明接口；SGEMM 或
   reduce 版本加入 `src/<op>` 的 registry，保证 `--impl <name>` 能选到它。
3. **CMake 自动发现**：本项目会自动发现 `src/**/*.cu`，通常不需要手动复制 source；
   但新增 benchmark/test target 时要更新 `CMakeLists.txt`。
4. **正确性测试**：在 `tests/cpp/test_<op>.cu` 添加 reference 对比，覆盖小尺寸、非
   整除尺寸、空/边界输入；需要 Python 时再同步 `tests/test_<op>.py`。
5. **benchmark 入口**：优先复用现有 `benchmarks/benchmark_<op>.cu` 的 `--impl`，不要
   为每个版本复制一个 benchmark。只有参数模型不同才扩展 CLI。
6. **profile kernel 名**：先用 `ncu --kernels` 或 nsys timeline 确认真实 kernel 名，
   再写 `--kernel-name='regex:<name>'`；不要猜名字。
7. **NVTX/profile range**：让 benchmark 的正式迭代包在 `cudaProfilerStart/Stop` 和
   NVTX range 内，否则 nsys/ncu 会把初始化和 warmup 混进结果。
8. **构建模式**：修改 `.cu` 后分别重建 `build/cmake-debug` 和 `build/cmake-profile`；
   PyTorch binding 则使用 `CUDA_BUILD_MODE=profile ... setup.py ... --force`。
9. **验证顺序**：先 CTest/pytest，再 Compute Sanitizer，再 nsys，再 ncu，最后 Release
   benchmark。任何 sanitizer 错误未修复前不要比较性能。

反量化项目建议直接复制这套目录边界：

```text
src/dequant/                 kernel、不同实现和 launcher
include/cuda_learning/       公共接口/registry
tests/cpp/test_dequant.cu    correctness reference
benchmarks/benchmark_dequant.cu  warmup、CUDA Event、NVTX、profile range
```

第一版反量化只做一个明确 baseline（例如 scalar FP32 accumulation），然后依次增加
向量化 load、shared-memory staging、融合 scale/zero-point、不同 block mapping。每次
只改变一个因素，并用同一组 shape、同一套 nsys/ncu 命令比较；这样当前项目就能作为
反量化项目的实验基线，而不是只停留在“能运行”的样例集合。

## 新增 SGEMM 版本

例如新增 `sgemm_shared_tiled`：

1. 新建 `src/sgemm/sgemm_shared_tiled.cu`，实现 kernel 和 launcher，签名必须匹配：

   ```cpp
   void launch_sgemm_shared_tiled(
       float* output, const float* a, const float* b,
       int m, int n, int k, cudaStream_t stream);
   ```

2. 在 `src/sgemm/sgemm.cu` 声明 launcher，并加入 registry：

   ```cpp
   {"shared_tiled", launch_sgemm_shared_tiled},
   ```

3. CMake 和 `setup.py` 会自动发现 `src/**/*.cu`，不需要复制 benchmark/main 或手动维护
   source 列表。
4. 重新构建后运行：

   ```shell
   # debug 
    cmake --build build/cmake-debug --target benchmark_sgemm -j 4

    compute-sanitizer --tool memcheck \
    build/cmake-debug/benchmark_sgemm \
    --impl float4 \
    --m 16 --n 16 --k 16 \
    --warmup 0 --iters 1

    # profile
    cmake --build build/cmake-profile --clean-first -j

    build/cmake-profile/test_sgemm
    
    build/cmake-profile/benchmark_sgemm \
        --impl shared_tiled --m 1024 --n 1024 --k 1024 \
        --warmup 10 --iters 100
   ```

5. 先运行 `compute-sanitizer`，再运行 `nsys/ncu`，最后用 Release 构建做性能复测。
6. 只有需要 Python 调用时，才在 `bindings/pytorch/api.cpp` 注册 wrapper，并在
   `tests/test_sgemm.py` 添加 Python API 测试。

新增 SGEMM 版本必须覆盖：小矩阵、非方阵、M/N/K 非 tile 整除、边界加载/写回、同步、
shared-memory 和向量化 load/store 的对齐问题。

## 新增其他算子

1. 在 `src/<op>/` 添加 framework-agnostic kernel/launcher。
2. 在 `include/cuda_learning/kernels.h` 或独立 header 声明公共接口。
3. 添加 `tests/cpp/test_<op>.cu`，使用 CPU/CUB/cuBLAS/PyTorch 等合适 reference。
4. 添加 `benchmarks/benchmark_<op>.cu`；复用 `standalone_utils.h`，不要复制测试框架。
5. 在 `CMakeLists.txt` 添加 test/benchmark target（SGEMM registry 版本不需要新增 target）。
6. 需要 Python 集成时再添加 binding、pytest 和 `benchmark.py` 入口。

推荐顺序：

```text
CTest/reference correctness
    -> Compute Sanitizer
    -> cuda-gdb（需要时）
    -> Nsight Systems
    -> Nsight Compute
    -> Release 性能复测
    -> 可选 PyTorch 集成
```

## 常见注意事项

- `Debug/-G`、`CUDA_LAUNCH_BLOCKING=1`、sanitizer、cuda-gdb 的耗时都不能用于性能结论。
- CMake kernel 源文件由 `cuda_kernels` 统一编译；不要再把同一个 `.cu` 手动加入 benchmark
  executable，否则会产生 `multiple definition`。
- 修改源文件或 registry 后，如果出现 `ninja: no work to do` 但 binary 没有新实现，使用：

  ```shell
  cmake --build build/cmake-profile --clean-first -j
  ```

- 普通用户运行 `ncu` 可能遇到 `ERR_NVGPUCTRPERM`。
- 当前 CUDA Toolkit 13.3 与 PyTorch cu130 存在 minor-version mismatch 警告；本机 CMake、
  CTest、PyTorch extension 和 pytest 均已验证通过。
