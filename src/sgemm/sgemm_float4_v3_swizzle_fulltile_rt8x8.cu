#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_learning/kernels.h"

// 计算行主序矩阵乘法 C[M, N] = A[M, K] * B[K, N]，直接覆盖 C。
// 本实现使用普通 FP32 FMA：沿 K 维分块，将 A/B 搬入共享内存，
// 再由每个线程在寄存器中累加自己的 8×8 输出元素。
//
// 当前启动配置：
//   输出块 BM×BN = 128×128，K 方向每轮 BK = 16；
//   线程块为 (16, 16)，共 256 个线程，8 个 warp；
//   每线程 64 个 FP32 累加器，共同覆盖 128×128 个输出元素。
// 数据流：global A/B -> 临时寄存器 -> shared A/B -> 片段寄存器 -> sum -> global C。
namespace cuda_learning {
namespace {

// A 在全局内存中按 [row][k] 存放，在共享内存中改为 [k][physical_row]。
// 转置后，计算阶段可以用一次 float4 读取同一 k 对应的连续 4 行 A。
//
// BK=16 时，k>>2 为 0、1、2、3，分别对应一个 A 行内的 4 个 float4。
// 将该编号左移 3 位，再与逻辑行号 XOR，即将 row 的第 3、4 位扰动：
//   k=0..3: row^0；k=4..7: row^8；k=8..11: row^16；k=12..15: row^24。
// 对固定 k，这是可逆置换；写入和读取使用同一映射即可还原逻辑坐标。
// 最低 3 位不变，所以连续且按 4 对齐的四行仍连续，可保持 float4 读取。
//
// 搬运 A 时，一个 warp 覆盖 8 行，每行 4 个 float4。若只做转置，
// 不同 k 组的线程会写入相同的共享内存 bank；XOR 将这些组分散到不同 bank。
// 这里依赖 float 为 4 字节、通常的 32-bank 布局，以及下方固定的分块约束。
__device__ __forceinline__ int swizzle_a_row(int k, int row) {
    return row ^ ((k >> 2) << 3);
}

// 从行主序矩阵的 (row, col) 读取四个连续元素，stride 为行跨度（元素数）。
// 此处 A/B 均紧密存储，因此 stride 同时也是合法列数：A 为 K，B 为 N。
// 调用方保证 col 是 4 的倍数；常规 CUDA 分配的基址满足 float4 的 16 字节对齐。
// FULL_TILE 是编译期参数，整块版本会直接去掉边界检查及标量补零分支。
template <bool FULL_TILE>
__device__ __forceinline__ float4 load_global4(const float *matrix, int row,
                                               int col, int rows, int stride) {
    if constexpr (FULL_TILE) {
        // 启动器保证 M/N/K 都按 tile 整除，此时每次向量读取均合法且对齐。
        return *reinterpret_cast<const float4 *>(
            matrix + static_cast<std::size_t>(row) * stride + col);
    } else {
        // 行跨度也必须是 4 的倍数，否则后续行的起点可能不再按 16 字节对齐。
        if (stride % 4 == 0 && row < rows && col + 3 < stride) {
            return *reinterpret_cast<const float4 *>(
                matrix + static_cast<std::size_t>(row) * stride + col);
        }
        // 向量跨越边界或行跨度不对齐时退回标量读取，越界分量补 0。
        // 因而最后一个 K tile 仍能执行完整的 BK 次乘加，而不引入额外贡献。
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i] = row < rows && col + i < stride
                     ? matrix[static_cast<std::size_t>(row) * stride + col + i]
                     : 0.0f;
        }
        return make_float4(v[0], v[1], v[2], v[3]);
    }
}

// 读取当前 k 的计算片段。AG/BG 是行/列方向的“四元素组”数量。
// 当前 AG=BG=2：每线程读取 8 个 A 值、8 个 B 值，形成一次 8×8 外积。
// 逻辑行：ty*4 + mi*(BM/AG) + r；逻辑列：tx*4 + ni*(BN/BG) + c。
// mi/ni 选分离的条带，r/c 选条带内的四个相邻元素。
template <int BM, int BN, int BK, int AG, int BG>
__device__ __forceinline__ void
load_fragments(float4 (&af)[AG], float4 (&bf)[BG], const float (&a)[BK][BM],
               const float (&b)[BK][BN], int k, int tx, int ty) {
#pragma unroll
    for (int mi = 0; mi < AG; ++mi) {
        const int row = ty * 4 + mi * (BM / AG);
        // A 写入时做过转置和 swizzle，读取时必须使用相同的物理行映射。
        af[mi]
            = *reinterpret_cast<const float4 *>(&a[k][swizzle_a_row(k, row)]);
    }
#pragma unroll
    for (int ni = 0; ni < BG; ++ni) {
        const int col = tx * 4 + ni * (BN / BG);
        bf[ni] = *reinterpret_cast<const float4 *>(&b[k][col]);
    }
}

template <int BM, int BN, int BK, int TM, int TN, bool FULL_TILE>
__global__ void sgemm_float4_v3_swizzle_fulltile_rt8x8(
    float *__restrict__ C, const float *__restrict__ A,
    const float *__restrict__ B, int M, int N, int K) {

    // BM/BN/BK：线程块在 M/N/K 三个维度上的 tile 大小。
    // TM/TN：每线程计算的输出行数/列数；AG/BG 将其拆成若干个 4 元素组。
    // 当前 AG=BG=2，THREADS_X=THREADS_Y=16。
    constexpr int AG = TM / 4, BG = TN / 4;
    constexpr int THREADS_X = BN / TN, THREADS_Y = BM / TM;
    constexpr int THREADS = THREADS_X * THREADS_Y;
    // 每次搬运 4 个 float，均匀分配整个 tile 的 global -> shared 工作。
    // 当前 A/B 各有 2048 个 float，每线程各搬运 2 个 float4（各 8 个 float）。
    constexpr int LOAD_A = BM * BK / (THREADS * 4);
    constexpr int LOAD_B = BK * BN / (THREADS * 4);

    // 约束 swizzle、向量对齐、线程布局和均匀搬运成立所需的模板参数。
    static_assert(BK == 16 && BM % 32 == 0 && BN % 32 == 0);
    static_assert((TM == 4 || TM == 8) && TN == 8);
    static_assert(THREADS_X == 16 && THREADS_Y == 16);
    static_assert(BM * BK % (THREADS * 4) == 0 && BK * BN % (THREADS * 4) == 0);

    // A 采用 [k][swizzled_row]，B 保持 [k][col]，便于计算时沿行/列取向量。
    // 两块共享内存各占 16*128*4 = 8192 字节，合计 16 KiB。
    // 16 字节对齐配合四元素对齐的偏移，支持共享内存 float4 访问。
    __shared__ __align__(16) float a_smem[BK][BM];
    __shared__ __align__(16) float b_smem[BK][BN];
    const int tx = threadIdx.x, ty = threadIdx.y;
    // CUDA 线程线性编号中 x 维变化最快，一个 warp 对应相邻的两行线程。
    const int tid = ty * THREADS_X + tx;

    // 每线程负责四个 4×4 子块，合计 8×8 个输出，而不是连续的 8 行/8 列：
    //   行组起点为 ty*4 和 ty*4+64；列组起点为 tx*4 和 tx*4+64。
    // 例如线程 (tx,ty)=(0,0) 计算行 {0..3,64..67}、列 {0..3,64..67} 的组合。
    // sum[mi][ni][r][c] 对应该组合中的一个 C 元素，初始化为 0，跨所有 K tile 累加。
    float sum[AG][BG][4][4] = {};

    // 每轮累加 A 的 BM×BK 子块与 B 的 BK×BN 子块的乘积。
    for (int tile_k = 0; tile_k < K; tile_k += BK) {
        float4 global_a[LOAD_A];
        float4 global_b[LOAD_B];
        // 先把本线程负责的全部 A/B 向量读到临时寄存器，再写共享内存。
        // 独立的全局读取可以为编译器调度提供空间；这不是跨 K tile 的异步流水。
        // 向量数量由 tile 元素数推导，而非在代码里固定写成 2。
#pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            // 将 A tile 视作 [BM][BK] 的一维数组：每线程取连续 4 元素，
            // 下一轮 i 跳过全块线程本轮覆盖的 THREADS*4 个元素。
            const int index = tid * 4 + i * THREADS * 4;
            global_a[i] = load_global4<FULL_TILE>(
                A, blockIdx.y * BM + index / BK, tile_k + index % BK, M, K);
        }
#pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            const int index = tid * 4 + i * THREADS * 4;
            // B tile 按 [BK][BN] 展平；相邻线程读取相邻四列，利于合并访存。
            global_b[i] = load_global4<FULL_TILE>(
                B, tile_k + index / BN, blockIdx.x * BN + index % BN, K, N);
        }
#pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            const int index = tid * 4 + i * THREADS * 4;
            const int row = index / BK, k = index % BK;
            // 这里 k 总是 4 的倍数，同一 float4 的 k..k+3 属于同一个 k 组，
            // 因此它的四个分量共享 physical_row，但写入四个不同的共享内存行。
            const int physical_row = swizzle_a_row(k, row);
            a_smem[k][physical_row] = global_a[i].x;
            a_smem[k + 1][physical_row] = global_a[i].y;
            a_smem[k + 2][physical_row] = global_a[i].z;
            a_smem[k + 3][physical_row] = global_a[i].w;
        }
#pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            const int index = tid * 4 + i * THREADS * 4;
            // B 无需转置或 swizzle，直接沿连续列以 float4 写入。
            *reinterpret_cast<float4 *>(&b_smem[index / BN][index % BN])
                = global_b[i];
        }
        // 全块屏障：计算会读取其他线程搬运的数据，必须等所有 shared 写入完成。
        __syncthreads();

        // 两套片段寄存器交替使用：计算当前 k 前，先发起下一 k 的 shared 读取。
        // 这提供隐藏共享内存读取延迟的机会；实际指令调度由编译器决定。
        float4 a_reg[2][AG], b_reg[2][BG];
        load_fragments<BM, BN, BK, AG, BG>(a_reg[0], b_reg[0], a_smem, b_smem,
                                           0, tx, ty);
        // 先填好 k=0 的片段，此后 cur 指向本轮计算所需的数据。
        int cur = 0;
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const int next = cur ^ 1;
            // 最后一轮无需预读，避免访问共享内存的第 BK 行。
            if (k + 1 < BK) {
                load_fragments<BM, BN, BK, AG, BG>(
                    a_reg[next], b_reg[next], a_smem, b_smem, k + 1, tx, ty);
            }
#pragma unroll
            for (int mi = 0; mi < AG; ++mi) {
                // 一个 A 分量在列方向复用，一个 B 分量在行方向复用。
                // 每对 (mi,ni) 执行一个 4×4 外积，当前配置每个 k 共做 64 次 FMA。
                const float4 av = a_reg[cur][mi];
                const float af[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
                for (int ni = 0; ni < BG; ++ni) {
                    const float4 bv = b_reg[cur][ni];
                    const float bf[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            // fmaf 执行 a*b+sum 的融合乘加，只对最终结果舍入一次。
                            sum[mi][ni][r][c]
                                = fmaf(af[r], bf[c], sum[mi][ni][r][c]);
                        }
                    }
                }
            }
            // 切换到已预读的片段；最后一轮切换后不再读取它。
            cur = next;
        }
        // 所有线程读完本轮共享内存后，才能由下一 K tile 的搬运覆盖它。
        // 此屏障避免某些 warp 提前写入，破坏其他 warp 尚未读取的数据。
        __syncthreads();
    }

    // 将四个 4×4 子块写回各自的全局坐标；每次沿 N 方向组织一个 float4。
    // 输出映射与 load_fragments 的逻辑行/列映射一致，各线程的输出互不重叠。
#pragma unroll
    for (int mi = 0; mi < AG; ++mi) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = blockIdx.y * BM + ty * 4 + mi * (BM / AG) + r;
#pragma unroll
            for (int ni = 0; ni < BG; ++ni) {
                const int col = blockIdx.x * BN + tx * 4 + ni * (BN / BG);
                const float4 out
                    = make_float4(sum[mi][ni][r][0], sum[mi][ni][r][1],
                                  sum[mi][ni][r][2], sum[mi][ni][r][3]);
                // 整块路径：行列都合法，且 N 是 128 的倍数，可直接向量化写回。
                if constexpr (FULL_TILE) {
                    *reinterpret_cast<float4 *>(
                        C + static_cast<std::size_t>(row) * N + col) = out;
                } else if (row < M) {
                    // 通用路径先判断行，再检查向量范围和每行起点的对齐条件。
                    if (N % 4 == 0 && col + 3 < N) {
                        *reinterpret_cast<float4 *>(
                            C + static_cast<std::size_t>(row) * N + col) = out;
                    } else {
                        // 右边界不足四列，或行跨度不满足向量对齐：逐元素检查并写入。
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            if (col + c < N) {
                                C[static_cast<std::size_t>(row) * N + col + c]
                                    = sum[mi][ni][r][c];
                            }
                        }
                    }
                }
            }
        }
    }
}
} // namespace

// 主机端入口：在调用方传入的 stream 上提交 kernel；这里不做同步或错误查询。
// 约定 M/N/K 非负，A/B/output 指向足够大的、满足向量对齐要求的设备内存。
void launch_sgemm_float4_v3_swizzle_fulltile_rt8x8(float *output,
                                                   const float *A,
                                                   const float *B, int M, int N,
                                                   int K, cudaStream_t stream) {
    constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
    // 没有输出元素时直接返回，避免提交 grid 某个维度为 0 的 kernel。
    if (M == 0 || N == 0) {
        return;
    }
    // 只有三个维度都整除，才能让所有块、所有 K 轮省去边界判断。
    // 即使仅有边缘不完整，也会为整个 grid 选择带检查的版本。
    // K=0 时循环不执行，非空 C 会被写为全 0，符合空内积的定义。
    const bool full_tile = M % BM == 0 && N % BN == 0 && K % BK == 0;
    // x 维覆盖输出列，y 维覆盖输出行；grid 向上取整以包含边缘 tile。
    const dim3 block(BN / TN, BM / TM);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    // 编译生成两个实例，用 if constexpr 消除整块实例中的通用路径分支。
    if (full_tile) {
        sgemm_float4_v3_swizzle_fulltile_rt8x8<BM, BN, BK, TM, TN, true>
            <<<grid, block, 0, stream>>>(output, A, B, M, N, K);
    } else {
        sgemm_float4_v3_swizzle_fulltile_rt8x8<BM, BN, BK, TM, TN, false>
            <<<grid, block, 0, stream>>>(output, A, B, M, N, K);
    }
}
} // namespace cuda_learning
