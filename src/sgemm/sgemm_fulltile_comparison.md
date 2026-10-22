# 完整 tile 特化实验

基于 preload_ab_frag2 新增独立实现 float4_v3_swizzle_preload_ab_frag2_fulltile；已有算子文件未修改。
host 使用 (M % BM == 0) && (N % BN == 0) && (K % BK == 0)，其中 BM=BN=64、BK=32，选择 FULL_TILE=true/false。
输入加载和输出写回以 if constexpr 分开；true 分支不生成边界、对齐检查、标量补零和标量输出，false 分支保留通用处理。计算、AB global 预加载、双 fragment、展开 32 和同步不变。

编译检查（Profile，sm_120）：

| 实现/特化 | 寄存器/线程 | 静态 shared | STACK | LOCAL |
|---|---:|---:|---:|---:|
| 原 frag2 | 80 | 16384 B | 0 | 0 |
| FULL_TILE=false | 80 | 16384 B | 0 | 0 |
| FULL_TILE=true | 63 | 16384 B | 0 | 0 |

PTX 核查：true 仅有四条 16 字节 global load、四条 16 字节 global store，无标量 global 加载/写回，且不再读取 M 参数；false 保留标量兜底。K tile 循环与必要的循环控制保留，不应将“无边界判断”理解为 kernel 完全没有控制流。

正确性：test_sgemm_swizzle 中九个实现均通过 15 组尺寸，包括完整 tile、M/N/K 各自单独尾部、非对齐行跨度、K=0，以及 512³、768³、1024³、2048³。

RTX 5060 Ti，1024³，5 轮交替顺序，每轮预热 30 次、CUDA Event 计时 300 次：

| 实现 | 第1轮 μs | 第2轮 μs | 第3轮 μs | 第4轮 μs | 第5轮 μs | 中位数 μs |
|---|---:|---:|---:|---:|---:|---:|
| float4_v3_swizzle_preload_ab_frag2 | 204.034 | 209.432 | 205.377 | 203.607 | 207.797 | 205.377 |
| float4_v3_swizzle_preload_ab_frag2_fulltile | 203.841 | 205.772 | 204.441 | 204.103 | 204.143 | 204.143 |

本次中位耗时降低约 0.60%，波动范围有重叠，暂不足以确认稳定加速；可以确认的是边界兜底代码已删除、寄存器数从 80 降至 63。
