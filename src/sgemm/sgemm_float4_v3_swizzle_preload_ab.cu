#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_learning/kernels.h"

namespace cuda_learning {

namespace {
__device__ __forceinline__ int swizzle_a_row(int smem_first_idx,
                                             int logical_row) {
    return logical_row ^ (((smem_first_idx >> 2) & 7) << 2);
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_float4_v3_swizzle_preload_ab(float *__restrict__ C,
                                                   const float *__restrict__ A,
                                                   const float *__restrict__ B,
                                                   int M, int N, int K) {
    // Global 合并读取，A 在写入 shared memory 时转置。
    constexpr int PAD = 0;

    constexpr int A_HALF_TILE = BM * BK / 2;
    constexpr int B_HALF_TILE = BK * BN / 2;

    constexpr int THREADS = (BM / TM) * (BN / TN);
    static_assert(BM * BK == THREADS * 8 && BK * BN == THREADS * 8,
                  "Each thread must load two float4 vectors per tile.");
    static_assert(BK % 4 == 0 && BN % 4 == 0,
                  "Global float4 loads must stay within tile rows.");
    static_assert((BM + PAD) % 4 == 0 && (BN + PAD) % 4 == 0,
                  "Shared-memory rows must preserve float4 alignment.");
    static_assert(TM == 4 && TN == 4, "This implementation assumes TM=TN=4.");

    // A: [row][k] -> [k][row]；固定 k 时，四个输出行对应的 A 连续。
    __shared__ float a_smem[BK][BM + PAD];
    __shared__ float b_smem[BK][BN + PAD];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int row0 = blockIdx.y * BM + ty * TM;
    const int col0 = blockIdx.x * BN + tx * TN;

    // 每个线程计算 TM*TN = 16 个输出元素。
    float sum[TM][TN];

#pragma unroll
    for (int r = 0; r < TM; ++r) {
#pragma unroll
        for (int c = 0; c < TN; ++c) {
            sum[r][c] = 0.0f;
        }
    }

    // ============================================================
    // 1. Cooperative loading：A
    //
    // 原先的映射让相邻线程的 float4 起始列相差 8 个 float。
    // 新映射让相邻线程的起始位置相差 4 个 float。
    //
    // 每个线程加载：
    //   第一个 float4：线性 tile 索引 tid*4
    //   第二个 float4：线性 tile 索引 tid*4 + A_HALF_TILE
    //
    // 对当前配置：
    //   A tile = 64*32 = 2048 floats
    //   A_HALF_TILE = 1024 floats
    // ============================================================

    const int a_i0 = tid * 4;
    const int a_i1 = a_i0 + A_HALF_TILE;

    const int a_row0 = a_i0 / BK;
    const int a_col0 = a_i0 % BK;

    const int a_row1 = a_i1 / BK;
    const int a_col1 = a_i1 % BK;

    // ============================================================
    // 2. Cooperative loading：B
    //
    // 对当前配置：
    //   B tile = 32*64 = 2048 floats
    //   B_HALF_TILE = 1024 floats
    // ============================================================

    const int b_i0 = tid * 4;
    const int b_i1 = b_i0 + B_HALF_TILE;

    const int b_row0 = b_i0 / BN;
    const int b_col0 = b_i0 % BN;

    const int b_row1 = b_i1 / BN;
    const int b_col1 = b_i1 % BN;

    // 全局矩阵的行跨度决定 float4 地址是否对齐。
    const bool a_aligned = (K % 4 == 0);
    const bool b_aligned = (N % 4 == 0);

    for (int tile_k = 0; tile_k < K; tile_k += BK) {

        // 当前 tile 的四个 global float4 load 均先于 shared store。
        const int ga_row0 = blockIdx.y * BM + a_row0;
        const int ga_col0 = tile_k + a_col0;
        const int ga_row1 = blockIdx.y * BM + a_row1;
        const int ga_col1 = tile_k + a_col1;
        const int gb_row0 = tile_k + b_row0;
        const int gb_col0 = blockIdx.x * BN + b_col0;
        const int gb_row1 = tile_k + b_row1;
        const int gb_col1 = blockIdx.x * BN + b_col1;
        if (a_aligned && ga_row0 < M && ga_col0 + 3 < K && ga_row1 < M
            && ga_col1 + 3 < K && b_aligned && gb_row0 < K && gb_col0 + 3 < N
            && gb_row1 < K && gb_col1 + 3 < N) {
            const float4 a0 = *reinterpret_cast<const float4 *>(
                A + static_cast<std::size_t>(ga_row0) * K + ga_col0);
            const float4 a1 = *reinterpret_cast<const float4 *>(
                A + static_cast<std::size_t>(ga_row1) * K + ga_col1);
            const float4 b0 = *reinterpret_cast<const float4 *>(
                B + static_cast<std::size_t>(gb_row0) * N + gb_col0);
            const float4 b1 = *reinterpret_cast<const float4 *>(
                B + static_cast<std::size_t>(gb_row1) * N + gb_col1);
            const int ar0 = swizzle_a_row(a_col0, a_row0);
            const int ar1 = swizzle_a_row(a_col1, a_row1);
            a_smem[a_col0][ar0] = a0.x;
            a_smem[a_col0 + 1][ar0] = a0.y;
            a_smem[a_col0 + 2][ar0] = a0.z;
            a_smem[a_col0 + 3][ar0] = a0.w;
            a_smem[a_col1][ar1] = a1.x;
            a_smem[a_col1 + 1][ar1] = a1.y;
            a_smem[a_col1 + 2][ar1] = a1.z;
            a_smem[a_col1 + 3][ar1] = a1.w;
            *reinterpret_cast<float4 *>(&b_smem[b_row0][b_col0]) = b0;
            *reinterpret_cast<float4 *>(&b_smem[b_row1][b_col1]) = b1;
        } else {
            // 边界和非对齐路径沿用基线的独立加载与补零。
            // --------------------------------------------------------
            // A: global 按 K 维连续读取，shared 转置写入 [k][row]。
            // --------------------------------------------------------

            // 第一个 float4

            if (a_aligned && ga_row0 < M && ga_col0 + 3 < K) {

                const float4 v = *reinterpret_cast<const float4 *>(
                    A + static_cast<std::size_t>(ga_row0) * K + ga_col0);

                const int a_swizzled_row0 = swizzle_a_row(a_col0, a_row0);

                a_smem[a_col0][a_swizzled_row0] = v.x;
                a_smem[a_col0 + 1][a_swizzled_row0] = v.y;
                a_smem[a_col0 + 2][a_swizzled_row0] = v.z;
                a_smem[a_col0 + 3][a_swizzled_row0] = v.w;

            } else {
// 边界处理：越界元素填 0。
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    a_smem[a_col0 + v][swizzle_a_row(a_col0 + v, a_row0)]
                        = (ga_row0 < M && ga_col0 + v < K)
                            ? A[static_cast<std::size_t>(ga_row0) * K + ga_col0
                                + v]
                            : 0.0f;
                }
            }

            // 第二个 float4

            if (a_aligned && ga_row1 < M && ga_col1 + 3 < K) {

                const float4 v = *reinterpret_cast<const float4 *>(
                    A + static_cast<std::size_t>(ga_row1) * K + ga_col1);

                const int a_swizzled_row1 = swizzle_a_row(a_col1, a_row1);

                a_smem[a_col1][a_swizzled_row1] = v.x;
                a_smem[a_col1 + 1][a_swizzled_row1] = v.y;
                a_smem[a_col1 + 2][a_swizzled_row1] = v.z;
                a_smem[a_col1 + 3][a_swizzled_row1] = v.w;

            } else {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    a_smem[a_col1 + v][swizzle_a_row(a_col1 + v, a_row1)]
                        = (ga_row1 < M && ga_col1 + v < K)
                            ? A[static_cast<std::size_t>(ga_row1) * K + ga_col1
                                + v]
                            : 0.0f;
                }
            }

            // --------------------------------------------------------
            // B: global memory -> shared memory非边界对齐情况
            // --------------------------------------------------------

            // 第一个 float4

            if (b_aligned && gb_row0 < K && gb_col0 + 3 < N) {

                const float4 v = *reinterpret_cast<const float4 *>(
                    B + static_cast<std::size_t>(gb_row0) * N + gb_col0);

                *reinterpret_cast<float4 *>(&b_smem[b_row0][b_col0]) = v;

            } else {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    b_smem[b_row0][b_col0 + v]
                        = (gb_row0 < K && gb_col0 + v < N)
                            ? B[static_cast<std::size_t>(gb_row0) * N + gb_col0
                                + v]
                            : 0.0f;
                }
            }

            // 第二个 float4

            if (b_aligned && gb_row1 < K && gb_col1 + 3 < N) {

                const float4 v = *reinterpret_cast<const float4 *>(
                    B + static_cast<std::size_t>(gb_row1) * N + gb_col1);

                *reinterpret_cast<float4 *>(&b_smem[b_row1][b_col1]) = v;

            } else {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    b_smem[b_row1][b_col1 + v]
                        = (gb_row1 < K && gb_col1 + v < N)
                            ? B[static_cast<std::size_t>(gb_row1) * N + gb_col1
                                + v]
                            : 0.0f;
                }
            }
        }

        // 确保整个 tile 加载完成，再开始计算。
        __syncthreads();

        // --------------------------------------------------------
        // 3. Compute：A/B 都沿 shared memory 行连续读取 float4。
        // --------------------------------------------------------

        // 各版本只改变此 K 计算循环的展开因子。
#pragma unroll 32
        for (int k = 0; k < BK; ++k) {
            // 读取时再次XOR找到实际的物理地址
            const int a_read_row = swizzle_a_row(k, ty * TM);

            const float4 a_vec
                = *reinterpret_cast<const float4 *>(&a_smem[k][a_read_row]);
            const float a_frag[TM] = {a_vec.x, a_vec.y, a_vec.z, a_vec.w};

            // 每个线程读取 B 的连续 4 个 float。
            // TN=4，tx*TN 为 4 的倍数；
            // BN+PAD=64，shared-memory 行跨度也满足 float4 对齐。
            const float4 b_vec
                = *reinterpret_cast<const float4 *>(&b_smem[k][tx * TN]);

#pragma unroll
            for (int r = 0; r < TM; ++r) {
                sum[r][0] = fmaf(a_frag[r], b_vec.x, sum[r][0]);
                sum[r][1] = fmaf(a_frag[r], b_vec.y, sum[r][1]);
                sum[r][2] = fmaf(a_frag[r], b_vec.z, sum[r][2]);
                sum[r][3] = fmaf(a_frag[r], b_vec.w, sum[r][3]);
            }
        }

        // 确保本轮计算结束后，再覆盖 shared-memory tile。
        __syncthreads();
    }

    // ============================================================
    // 4. Store C：连续 4 个 float 尽可能使用一次 float4 store
    // ============================================================

#pragma unroll
    for (int r = 0; r < TM; ++r) {
        const int row = row0 + r;

        if (row < M) {
            float *c_ptr = C + static_cast<std::size_t>(row) * N + col0;

            // N 是 4 的倍数时，每行起始地址保持 16B 对齐。
            // col0 也是 4 的倍数，因此可以安全进行 float4 store。
            if ((N % 4 == 0) && (col0 + 3 < N)) {

                const float4 out
                    = make_float4(sum[r][0], sum[r][1], sum[r][2], sum[r][3]);

                *reinterpret_cast<float4 *>(c_ptr) = out;

            } else {
// 边界或不满足对齐条件时，使用标量写回。
#pragma unroll
                for (int c = 0; c < TN; ++c) {
                    if (col0 + c < N) {
                        c_ptr[c] = sum[r][c];
                    }
                }
            }
        }
    }
}

} // namespace

void launch_sgemm_float4_v3_swizzle_preload_ab(float *output, const float *A,
                                               const float *B, int M, int N,
                                               int K, cudaStream_t stream) {
    constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
    // BM, BN: block on M, N dimension processing numbers
    // TM, TN: thread on M, N dimension processing numbers

    const dim3 block(BN / TN, BM / TM); // (16, 16) = 256 线程
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_v3_swizzle_preload_ab<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(output, A, B, M, N, K);
}

} // namespace cuda_learning
