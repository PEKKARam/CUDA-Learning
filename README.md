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
