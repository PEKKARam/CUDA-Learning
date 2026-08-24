# `sgemm_double_buffers` 学习笔记

这个实现用于学习 tiled SGEMM、`float4` 向量化访存、shared-memory padding，以及双
buffer（ping-pong buffer）的基本组织方式。它优先保证边界正确和代码可读性。

## 1. 线程和 tile 的映射

当前模板参数是：

```cpp
BM = 64, BN = 64, BK = 32
TM = 4,  TN = 4
```

一个 block 计算 `C` 的 `64x64` tile，block 大小为：

```cpp
dim3 block(BN / TN, BM / TM); // 16x16 = 256 threads
```

每个线程计算 `4x4` 个输出元素：

```text
thread (tx, ty)
  rows = block_row + ty * 4 + [0, 1, 2, 3]
  cols = block_col + tx * 4 + [0, 1, 2, 3]
```

因此一个 block 的 256 个线程共同覆盖 `64x64` 个结果元素。

## 2. 为什么需要 K 方向分块

矩阵乘法中：

```text
C[M][N] = A[M][K] * B[K][N]
```

如果每次都从 global memory 读取 A 和 B，同一个数据会被很多线程重复读取。这里把
K 维切成 `BK=32` 的 tile：

```text
for tile_k = 0, 32, 64, ...:
    load A[:, tile_k:tile_k+32] 到 shared memory
    load B[tile_k:tile_k+32, :] 到 shared memory
    使用 shared memory 完成 32 次 FMA
```

shared memory 能让一个 block 中的线程复用同一个 tile，减少 global-memory 流量。

## 3. 双 buffer 的数据结构

代码中有两份 A/B shared-memory tile：

```cpp
__shared__ float a_smem[2][BM][BK + PAD];
__shared__ float b_smem[2][BK][BN + PAD];
```

第一维的 `0` 和 `1` 是两个 buffer：

```text
buffer 0: 当前正在计算的 tile
buffer 1: 下一轮准备使用的 tile
```

每轮通过下面的公式选择 buffer：

```cpp
const int cur = tile_index & 1;
const int next = cur ^ 1;
```

注意这里使用的是 tile 编号，而不是元素偏移。如果写成 `tile_k & 1`，当 `BK=32`
时 `tile_k` 是 `0, 32, 64, ...`，结果永远为 0，双 buffer 就不会切换。

## 4. 当前版本的同步顺序

当前实现的时序是：

```text
加载 tile 0 到 buffer 0
__syncthreads()

计算 buffer 0，同时普通地加载 buffer 1
__syncthreads()

计算 buffer 1，同时普通地加载 buffer 0
__syncthreads()
```

`__syncthreads()` 有两个作用：

1. 第一次确保所有线程完成初始 tile 的写入。
2. 每一轮确保下一 buffer 的写入完成后，才允许下一轮读取它。

重要：当前加载使用普通的 global load（包括 `float4` load），因此这份代码是
“双份 shared-memory 存储 + ping-pong 切换”的教学版本，还没有实现真正的 global-load
和 FMA 的硬件重叠。要实现真正的异步双缓冲，需要进一步使用 `cuda::memcpy_async`
或 `cp.async`。

## 5. `float4` 加载为什么安全

A 和 B 的连续方向各由线程加载 8 个 float，也就是两个 `float4`：

```cpp
const float4 v0 = *reinterpret_cast<const float4 *>(address);
const float4 v1 = *reinterpret_cast<const float4 *>(address + 4);
```

只有在以下条件满足时才走向量化快路径：

```cpp
aligned && row_is_valid && col + 7 < bound
```

其中 `aligned` 根据 `K` 和 `N` 是否为 4 的倍数判断。边界 tile 或不满足对齐时，代码
逐元素加载并用 `0.0f` 补齐，这样可以支持任意正尺寸测试。

## 6. 为什么 shared memory 加 padding

没有 padding 时，A/B 的行跨度分别是 32 或 64 个 float，正好是 bank 数量的倍数。
某些 warp 访问模式下，不同行会重复落到同一组 bank，可能产生 bank conflict。

代码使用：

```cpp
constexpr int PAD = 4;
```

逻辑 tile 仍然只使用前 `BK` 或前 `BN` 列，额外空间只用于改变物理行跨度。padding
不会改变矩阵数学含义，也不会参与 FMA。

## 7. 为什么它可能仍然比 `float4` 版本慢

双 buffer 不是自动加速开关。当前实现仍然使用普通同步 load，因此它增加了：

- 两份 shared memory 占用；
- buffer 索引和地址计算；
- 每个 tile 的同步开销；
- 可能的寄存器和 occupancy 压力。

只有当“加载下一个 tile”和“计算当前 tile”真正重叠时，双 buffer 才可能抵消这些成本。
下一步优化方向是：

1. 使用 `cuda::memcpy_async` / `cp.async`；
2. 在等待当前 tile 的同时预取下一个 tile；
3. 用 Nsight Compute 比较 shared bank conflict、barrier stall、occupancy；
4. 调整 `BM/BN/BK/TM/TN`，而不是只增加 buffer 数量。

## 8. 推荐验证流程

先做正确性测试：

```bash
cmake --build build/cmake-debug --target test_sgemm -j 4
build/cmake-debug/test_sgemm
```

再用 memcheck 检查边界：

```bash
compute-sanitizer --tool memcheck \
  build/cmake-debug/benchmark_sgemm \
  --impl double_buffers --m 33 --n 33 --k 33 \
  --warmup 0 --iters 1
```

最后在相同条件下比较性能：

```bash
build/cmake-profile/benchmark_sgemm \
  --impl float4 --m 1024 --n 1024 --k 1024 \
  --warmup 10 --iters 100

build/cmake-profile/benchmark_sgemm \
  --impl double_buffers --m 1024 --n 1024 --k 1024 \
  --warmup 10 --iters 100
```

性能比较必须保持矩阵尺寸、warmup、迭代次数和 build mode 一致。

## 9. Nsight Compute 观察重点

```bash
ncu --set=full \
  --kernel-name='regex:sgemm_double_buffers' \
  build/cmake-profile/benchmark_sgemm \
  --impl double_buffers --m 1024 --n 1024 --k 1024 \
  --warmup 0 --iters 1
```

重点比较：

```text
Shared Memory Bank Conflicts
Shared Load/Store Transactions
Barrier Stall
Memory Dependency Stall
Registers Per Thread
Shared Memory Per Block
Achieved Occupancy
```

如果看到 barrier stall 很高，而 global-memory latency 没有被隐藏，说明当前普通
load 版本还没有形成真正的异步重叠。
