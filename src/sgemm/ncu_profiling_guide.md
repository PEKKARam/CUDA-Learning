# Nsight Compute 入门与 SGEMM 分析指南

本文以项目中的 `sgemm_float4.cu` 和 `sgemm_float4_v2.cu` 为例，介绍如何从 NCU
报告定位瓶颈。建议先完成 correctness，再做 profile；不要用 Debug、`-G`、
Compute Sanitizer 或 `CUDA_LAUNCH_BLOCKING=1` 的时间做性能结论。

## 1. nsys 与 ncu 的分工

```text
nsys: 整个程序的时间线。看 kernel、memcpy、CPU 等待是否串行。
ncu : 单个 kernel 的硬件指标。看内存、occupancy、stall、指令和 bank conflict。
```

固定流程：

```text
正确性 -> memcheck -> nsys 时间线 -> ncu 单 kernel -> 修改 -> 重新验证
```

## 2. 准备 Profile 构建

Profile 构建保留 `-O3` 并加入 `-lineinfo`，可以把机器指令映射回源码：

```bash
cmake -S . -B build/cmake-profile -G Ninja \
  -DCMAKE_BUILD_TYPE=Profile \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/cmake-profile -j
```

确认实际架构：

```bash
nvidia-smi
nvcc --version
```

如果出现 `ERR_NVGPUCTRPERM`，先解决性能计数器权限；程序能够打印 benchmark 结果
不代表 NCU 已成功采集硬件指标。

## 3. 第一次只采集一个 kernel

不要一开始使用 `--set=full` 和大迭代次数。先用一个小输入验证 kernel 名称和报告：

```bash
mkdir -p reports/sgemm
ncu --set=basic \
  --kernel-name-base=demangled \
  --kernel-name='regex:sgemm_float4_v2' \
  --export=reports/sgemm/float4-v2-basic \
  build/cmake-profile/benchmark_sgemm \
  --impl float4_v2 --m 128 --n 128 --k 128 \
  --warmup 0 --iters 1
```

确认报告后再采集完整指标：

```bash
ncu --set=full \
  --kernel-name-base=demangled \
  --kernel-name='regex:sgemm_float4_v2' \
  --export=reports/sgemm/float4-v2-full \
  build/cmake-profile/benchmark_sgemm \
  --impl float4_v2 --m 1024 --n 1024 --k 1024 \
  --warmup 0 --iters 1
```

图形界面：

```bash
ncu-ui reports/sgemm/float4-v2-full.ncu-rep
```

命令行查看报告：

```bash
ncu --import reports/sgemm/float4-v2-full.ncu-rep --page details
ncu --import reports/sgemm/float4-v2-full.ncu-rep --page raw
```

## 4. 先看 Summary，再看 Details

### 4.1 Speedup 建议不是实测保证

NCU 的 `Est. Speedup` 是基于当前计数器的估算，表示“如果消除该类浪费，理论上
可能获得的上限”。它不是修改代码后必然得到的加速比。多个 rule 的百分比也不能
直接相加，因为它们可能由同一批指令产生。

### 4.2 Memory Throughput

```text
Memory Throughput 很低 + Mem Busy 不高:
    可能不是 DRAM 带宽瓶颈，应继续看指令、occupancy、stall。

Memory Throughput 高 + Max Bandwidth 接近上限:
    重点减少 global memory 流量或提高复用。
```

### 4.3 Occupancy 与寄存器

在 `Launch Statistics` 查看：

```text
Registers Per Thread
Shared Memory Per Block
Theoretical/ Achieved Occupancy
```

寄存器增加后 occupancy 降低，或者出现 `Local Memory Load/Store`，说明可能发生
register spilling；此时“更多寄存器缓存”反而可能更慢。

## 5. 如何理解 global memory coalescing

GPU 的合并是以 **一个 warp 的一条内存指令** 为单位分析的，不是以“一个线程总共
读取了多少字节”为单位。需要区分：

```text
float4       = 16 bytes = 4 floats
32B sector   = cache/request 的一个统计粒度
warp request = 32 个线程在同一条 load/store 指令上的地址集合
```

如果一个线程执行两条 `float4` load：

```cpp
v0 = *(float4*)(ptr + 0);
v1 = *(float4*)(ptr + 4);
```

这是两条独立的内存指令。不能把它们合并成“一个线程一次访问 32B”，然后据此判断
coalescing。每条指令都要单独检查 warp 内 32 个线程的地址是否连续、是否跨越过多
的 32B sector。

## 6. 为什么 C 写回也可能被 NCU 报不合并

### 情况 A：v2 的 C 写回是标量写

`sgemm_float4_v2.cu` 当前写法是：

```cpp
for (int c = 0; c < TN; ++c) {
    C[(std::size_t)row * N + col0 + c] = sum[r][c];
}
```

v2 的 `TN=2`，所以每个线程执行两次标量 store，而不是一次 `float4` store。NCU 的
`L1TEX Global Store Access Pattern` 报告直接对应这些 store，很正常。

### 情况 B：即使是 float4 store，也要看 warp 排布

如果使用：

```cpp
*reinterpret_cast<float4*>(C + row * N + col0) = value;
```

每个线程确实写 16B，但当前常见 block 是 `16x16`。一个 warp 的 32 个 lane 通常由
两行 `ty` 组成：前 16 个线程写一行，后 16 个线程写另一行。于是同一条 store 指令
在 warp 内形成两个相隔 `N` 的地址区间，而不是一个 32-lane 连续区间。

这不一定是错误：每个半 warp 仍可能分别合并；但 sector 利用率和 transaction 数量
取决于具体地址对齐、硬件和编译器生成的指令。应在 `Source Counters`/`SASS` 中确认
实际是哪一条 store 被计数，不能只看 C++ 源码直觉。

### 情况 C：你对 sector 的推断哪里不准确

“两个 `float4` 总共 32B，刚好一个 sector”这个前提不成立：

```text
一个 float4 = 16B
两个 float4 = 32B，但它们是两条指令；
而 NCU 的 sector 利用率是按每条 warp 指令统计的。
```

另外，sector 是 cache/request 统计单位，不等于“一个线程独占一个 sector”。相邻
线程的 16B 请求可能共享 sector，也可能因为 stride/对齐不能共享。

## 7. 读取 Source Counters 的方法

看到 rule 后按这个顺序定位：

1. 展开 rule，记录 `L2 Theoretical Sectors Global Excessive` 或 `L1 Wavefronts Shared Excessive` 中的源码行。
2. 点击源码行，确认是 global load、global store 还是 shared load/store。
3. 回到 C++，写出一个 warp 中 lane 0、lane 1、lane 16 的地址。
4. 计算相邻 lane 的字节 stride，以及是否跨 32B/128B 边界。
5. 用 `ncu --page source` 或 GUI 的 Source/SASS 对照实际指令。

例如 `float4` 写回可先计算：

```text
lane 地址 = (row * N + block_col + tx * 4) * sizeof(float)
```

如果 warp 中 `row` 在 lane 16 发生跳变，说明一个 warp 被拆成多个行区间；如果同一
行内 lane 的地址每次增加 16B，则该半 warp 内是连续的。

## 8. Shared-memory bank conflict 的阅读方法

NCU 中常见的两类提示：

```text
Shared Load Bank Conflicts
Shared Store Bank Conflicts
```

看到它们时立刻联想到：

```text
一个 warp 的线程访问了同一 bank 的不同地址，硬件需要拆成多个 wavefront。
```

优先检查：

```text
二维数组行跨度是否是 32 的倍数；
warp 内线程是否按列访问转置数据；
float4 shared load/store 是否跨 bank 边界；
是否可以加 PAD=1/4，或改变 tile/线程映射。
```

padding 不是万能的，必须修改后重新看 `wavefronts` 和 kernel 时间。

## 9. 对两版 kernel 的解读

### `float4_v2`

优点：

```text
BM=BN=32、TM=TN=2，shared memory 占用较小；
每个线程只维护 4 个累加器，寄存器压力较低；
A/B tile 加载每线程一个 float4，代码直观。
```

缺点：

```text
C 写回仍是标量 store；
16x16 block 让一个 warp 跨两行输出；
没有 shared-memory padding，规则可能报告 shared load/store 冲突；
每个线程计算量较小，地址和同步开销占比可能较高。
```

### `float4`

优点：

```text
BM=BN=64、TM=TN=4，单线程计算 16 个 FMA，数据复用更强；
可使用 float4 B load 和 float4 C store；
更大的 tile 减少 block/grid 管理开销。
```

缺点：

```text
每线程累加器更多，寄存器压力更高；
shared memory 使用量和 tile 边界成本更大；
若 warp 排布没有针对输出布局设计，C store 仍可能出现额外 sector；
shared-memory padding、float4 对齐和尾部处理必须同时正确。
```

比较时不能只看某个 speedup rule。应同时记录：

```text
kernel time / cycles
registers per thread
achieved occupancy
global load/store sectors
shared wavefronts/conflicts
warp stall reason
```

## 10. 常用的定向采集命令

只看 global memory：

```bash
ncu --metrics \
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.sum.per_cycle_active,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.sum.per_cycle_active \
  --kernel-name='regex:sgemm_float4_v2' \
  build/cmake-profile/benchmark_sgemm \
  --impl float4_v2 --m 1024 --n 1024 --k 1024 --warmup 0 --iters 1
```

指标名称会随 GPU/驱动版本变化；先用：

```bash
ncu --query-metrics | rg -i 'global|sector|shared|bank'
```

再选择当前设备实际支持的名字。

## 11. 每次修改后的最小闭环

```bash
# 1. 正确性
build/cmake-debug/test_sgemm

# 2. 边界和非法访问
compute-sanitizer --tool memcheck \
  build/cmake-debug/benchmark_sgemm \
  --impl float4_v2 --m 33 --n 33 --k 33 --warmup 0 --iters 1

# 3. 性能
build/cmake-profile/benchmark_sgemm \
  --impl float4_v2 --m 1024 --n 1024 --k 1024 \
  --warmup 10 --iters 100

# 4. NCU 指标
ncu --set=full --kernel-name='regex:sgemm_float4_v2' ...
```

每次只改变一个因素，例如只改 `TN`、只加 padding、只改 block 排布。这样才能把
NCU 指标变化和代码改动建立因果关系。
