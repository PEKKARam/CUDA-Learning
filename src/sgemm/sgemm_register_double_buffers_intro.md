# 寄存器双缓冲 SGEMM

`register_double_buffers` 是一个用于学习“寄存器预取”的版本。它和
`double_buffers` 的区别是：

```text
double_buffers:
    在 shared memory 中准备两个 A/B tile

register_double_buffers:
    在 shared memory 中只保留一个 tile，
    在 K 循环中用两个寄存器槽位保存当前/下一个 A、B 片段
```

## 1. “双寄存器”到底是什么

这里的“双寄存器”不是指硬件上只有两个寄存器，而是两个逻辑槽位：

```cpp
float a_reg[2][TM];
float4 b_reg[2];
```

每个槽位保存一个 K 位置所需的寄存器片段：

```text
a_reg[slot][r] : 当前线程负责的 TM 个 A 值
b_reg[slot]    : 当前线程负责的 4 个 B 值
```

`cur` 表示正在参与 FMA 的槽位，`next` 表示准备中的槽位：

```cpp
const int tmp = cur;
cur = next;
next = tmp;
```

## 2. K 循环中的时序

每次循环先把下一个 K 位置从 shared memory 读入另一个寄存器槽位，然后使用当前
槽位完成 FMA：

```text
预取 k+1 到 register[next]
使用 register[cur] 做 FMA
交换 cur / next
```

理想的指令调度可以把下一次 shared-memory load 放到当前 FMA 指令之间，减少
shared-memory load latency 暴露出来的时间。最终是否真的重排，需要由编译器和硬件
调度器决定，不能仅凭源码保证。

## 3. 与 shared-memory 双缓冲的区别

这两个优化针对不同层次：

| 版本 | 双缓冲的位置 | 主要目的 |
| --- | --- | --- |
| `double_buffers` | shared memory | 准备下一个 K tile |
| `register_double_buffers` | registers | 预取当前 tile 内的 A/B 片段 |

寄存器双缓冲不会自动实现 global memory 与计算重叠，因为数据已经先进入 shared
memory。它主要减少计算循环中反复读取 shared memory 的暴露延迟。

## 4. 寄存器压力的代价

这个版本增加了：

```cpp
float a_reg[2][TM];
float4 b_reg[2];
float sum[TM][TN];
```

寄存器使用量可能上升，导致 occupancy 降低，甚至发生 register spilling。使用 Nsight
Compute 时要同时观察：

```text
Registers Per Thread
Local Memory / Local Load
Achieved Occupancy
```

如果出现 local memory load/store，说明寄存器不够，数组被溢出到 local memory，性能
可能比没有寄存器双缓冲更差。

## 5. `cuda::memcpy_async` 和 `cp.async` 是什么

寄存器双缓冲和异步 global-to-shared copy 是两个不同概念。

### 普通 load

```cpp
value = A[index];
shared_value = value;
```

线程需要等待 load 指令完成后再继续执行。

### `cuda::memcpy_async`

这是 CUDA C++ 对异步拷贝的高层封装，可以请求硬件把 global memory 数据搬到 shared
memory，并通过 pipeline/barrier 查询拷贝何时完成。概念示例：

```cpp
// 伪代码，具体重载随 CUDA 版本和 pipeline scope 变化
cuda::memcpy_async(block, shared_dst, global_src, bytes, pipeline);
pipeline.commit();
pipeline.wait_prior<0>();
```

重点是：发起 copy 后，线程可以继续准备其他工作，之后再等待 copy 完成。

### `cp.async`

`cp.async` 是更底层的 PTX/SASS 级异步 global-to-shared copy 指令，通常用于 Ampere
及更新架构。它允许：

```text
异步加载 tile N+1 到 shared memory
同时计算 tile N
```

典型概念流程是：

```text
cp.async 发起若干次拷贝
cp.async.commit_group
计算当前 tile
cp.async.wait_group
切换 shared-memory buffer
```

`cuda::memcpy_async` 更容易学习和维护，`cp.async` 更接近硬件，控制也更细。学习顺序
建议是：

1. 先理解当前 `double_buffers` 的 ping-pong 索引和同步；
2. 再学习 `cuda::memcpy_async` 的 pipeline/barrier；
3. 最后阅读 PTX 文档中的 `cp.async` 和 CUDA Samples。

## 6. 验证与比较

```bash
cmake --build build/cmake-debug --target test_sgemm -j 4
build/cmake-debug/test_sgemm

compute-sanitizer --tool memcheck \
  build/cmake-debug/benchmark_sgemm \
  --impl register_double_buffers --m 33 --n 33 --k 33 \
  --warmup 0 --iters 1

build/cmake-profile/benchmark_sgemm \
  --impl register_double_buffers --m 1024 --n 1024 --k 1024 \
  --warmup 10 --iters 100
```

必须与其他实现使用相同的矩阵尺寸、warmup、迭代次数和构建模式。优化是否有效，
最终以 benchmark 和 Nsight Compute 的寄存器/occupancy/local-memory 数据为准。
