# Fulltile register tile 4×8 / 8×8 实验

2026-10-22，RTX 5060 Ti，Profile 构建，sm_120，M=N=K=1024。
只启用 swizzle 基线、原 4×4 fulltile 和两个新 register tile；其他已测试版本已注释 registry 及专项测试名单。现有 kernel 文件未修改。

| 实现 | TM×TN | BM×BN×BK | 寄存器/线程（完整 tile） | 静态 shared KiB |
|---|---|---|---:|---:|
| float4_v3_swizzle | 4x4 | 64x64x32 | 80 | 16 |
| float4_v3_swizzle_preload_ab_frag2_fulltile | 4x4 | 64x64x32 | 63 | 16 |
| float4_v3_swizzle_fulltile_rt4x8 | 4x8 | 64x128x16 | 72 | 12 |
| float4_v3_swizzle_fulltile_rt8x8 | 8x8 | 128x128x16 | 118 | 16 |
| cublas | — | — | — | — |

新版本均为 256 threads/block（16×16）。协作 global 加载按 tile 大小推导：4×8 每线程读 A 一个 float4、B 两个 float4；8×8 每线程读 A/B 各两个 float4。SASS 中这些 global load 均先于 shared store。
两组 fragment 在 k 维交替，计算循环完整展开 BK=16。完整 tile 路径不生成边界/补零代码，其他尺寸使用同 kernel 的 FULL_TILE=false 特化。
两者 STACK=0、LOCAL=0，未生成 LDL/STL。非完整 tile 路径分别使用 77、127 个寄存器。

## 4×4 小块与 bank 映射

4×8 的输出列是 `tx*4 + ni*64`（ni=0,1）；8×8 再将行拆为 `ty*4 + mi*64`（mi=0,1）。
每个线程持有两个或四个 4×4 accumulator 小块，不是对连续 8 列使用 tx*8 的映射。A fragment 被不同列组复用，B fragment 被不同行组复用。
共享 A 按 `[k][row]` 存放，BK=16 专用 swizzle 为 `physical_row = row ^ ((k >> 2) << 3)`。
对 A 转置写入的每条标量 STS，一个 warp 的 row=0..7、k-group=0..3；row 的低三位和 XOR mask 的第 3/4 位共同覆盖 32 个 bank。
mask 的低三位为零，因此四个连续行仍能以对齐 float4 读取。相邻线程计算时的 B float4 起始地址只相差 4 floats，第二列组在 64 列之后；重复 A/B 地址由广播处理。
已枚举所有 warp、k、加载轮次，验证标量 A store 和每个 128B 向量服务组不存在同 bank 异址冲突；另外用 NCU source 计数核查：

| 配置 | 操作 | 实际 shared wavefronts | 理想 shared wavefronts | Excessive |
|---|---|---:|---:|---:|
| 4x8 | LDS | 10485760 | 10485760 | 0 |
| 4x8 | STS | 786432 | 786432 | 0 |
| 8x8 | LDS | 6291456 | 6291456 | 0 |
| 8x8 | STS | 524288 | 524288 | 0 |

首次只采集 bank-conflict 汇总 counter 时，4×8 load 有少量非零计数（291）；后续 source 级所有 LDS/STS 均为 actual=ideal、excessive=0。这里以 source 级证据判断布局，不宣称每次汇总 counter 必然为零。
NCU source 报告：

- `reports/sgemm/register_tile_rt4x8_20261022.ncu-rep`
- `reports/sgemm/register_tile_rt8x8_20261022.ncu-rep`

NCU profiler 内 benchmark 打印的 CUDA Event 时间包含采集开销，不参与下面的性能比较。

## 正确性与性能

专项测试的四个实现全部通过 21 组尺寸，含完整 tile、M/N/K 独立尾部、非对齐跨度、K=0、512³、768³、1024³、2048³。
两个新版本另对去除大尺寸的 17 组测试执行 Compute Sanitizer：memcheck 0 errors；racecheck 0 hazards、0 errors、0 warnings。
普通 benchmark 每轮预热 30 次、CUDA Event 计时 300 次，共五轮并轮换五个实现的运行顺序。每轮均通过 cuBLAS 正确性检查。

| 实现 | 第1轮 μs | 第2轮 μs | 第3轮 μs | 第4轮 μs | 第5轮 μs | 中位数 μs |
|---|---:|---:|---:|---:|---:|---:|
| float4_v3_swizzle | 225.707 | 224.653 | 222.981 | 225.922 | 225.925 | 225.707 |
| float4_v3_swizzle_preload_ab_frag2_fulltile | 220.209 | 221.592 | 218.002 | 217.145 | 220.330 | 220.209 |
| float4_v3_swizzle_fulltile_rt4x8 | 203.516 | 203.995 | 203.012 | 203.668 | 198.674 | 203.516 |
| float4_v3_swizzle_fulltile_rt8x8 | 172.469 | 168.491 | 169.286 | 175.169 | 170.807 | 170.807 |
| cublas | 172.682 | 174.416 | 171.748 | 173.945 | 170.669 | 172.682 |

相对原 fulltile，4×8 中位耗时降低 7.58%，8×8 降低 22.43%。
8×8 本轮与 cuBLAS 处于接近的测量区间，不能据此认定稳定超越 cuBLAS。
NCU 理论 occupancy：4×8 为 50%（每 SM 3 blocks），8×8 为 33.33%（2 blocks）；较低 occupancy 没有阻止 8×8 在本轮获得更低耗时。
本次同时修改 register tile、block tile、BK 和地址映射，不能把全部性能变化归因于单一参数。
