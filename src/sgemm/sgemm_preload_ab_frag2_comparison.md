# Preload AB + 两组计算 fragment 实验

RTX 5060 Ti，Profile 构建，sm_120，M=N=K=1024。两者均保持计算 K 循环展开 32。
新增版本保持 AB global 提前加载；首组 shared fragment 在计算循环前加载，在 k 的 FMA 前发出 k+1 的 shared load，使用两组 float4 A/B 寄存器交替保存。最后一个 k 不访问 k+1。
不增加 shared buffer，不跨 tile 预取；global 加载、边界处理、同步、写回和 launch 配置与 preload_ab 保持一致。原有算子文件未修改。

每轮预热 30 次，正式计时 300 次，5 轮交替运行顺序；下表每轮是 CUDA Event 平均耗时。

| 实现 | 第1轮 μs | 第2轮 μs | 第3轮 μs | 第4轮 μs | 第5轮 μs | 中位数 μs |
|---|---:|---:|---:|---:|---:|---:|
| float4_v3_swizzle_preload_ab | 203.147 | 204.066 | 204.691 | 209.594 | 205.785 | 204.691 |
| float4_v3_swizzle_preload_ab_frag2 | 206.022 | 207.810 | 205.919 | 206.073 | 204.090 | 206.022 |

两者均为 80 registers/thread、16 KiB 静态 shared、STACK=0、LOCAL=0。展开代码均有 64 条 LDS.128 和 512 条 FFMA。
SASS 中新版本在首轮 FFMA 前已发出 k=0 和 k=1 的 A/B LDS；后续 LDS 与 FFMA 交错。对照版也已被编译器安排了类似预取，所以源码双 fragment 并未带来全新的调度模式。
本次未观察到明确提速：对照中位耗时 204.691 μs，新版本 206.022 μs，差约 0.65%，波动范围重叠。不能据此认定 shared 预取无效；这里只验证了展开 32 下的这个实现。
验证：test_sgemm 对八个注册实现的 1024³、2048³ 全部通过；新版本还通过 1×1×1、31×29×17、64×64×32、65×68×36、68×65×35、127×132×65 的 cuBLAS 正确性比较。
