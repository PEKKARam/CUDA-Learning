#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_learning/kernels.h"

namespace cuda_learning {

namespace {
__device__ __forceinline__ int swizzle_a_row(int smem_first_idx,
                                             int logical_row) {
    return logical_row ^ (((smem_first_idx >> 2) & 7) << 2);
}

// Experiments share one implementation; each registry entry changes one policy.
template <bool FULL_TILE>
__device__ __forceinline__ float4 load_vector(const float *matrix, int row,
                                              int col, int rows, int stride) {
    if (FULL_TILE || (stride % 4 == 0 && row < rows && col + 3 < stride)) {
        return *reinterpret_cast<const float4 *>(
            matrix + static_cast<std::size_t>(row) * stride + col);
    }
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = (row < rows && col + i < stride)
                 ? matrix[static_cast<std::size_t>(row) * stride + col + i]
                 : 0.0f;
    }
    return make_float4(v[0], v[1], v[2], v[3]);
}

template <int BM, int BN, int BK, int TM, int TN, int UNROLL, bool LOAD_ALL,
          bool FULL_TILE>
__device__ __forceinline__ void
swizzle_experiment_body(float *__restrict__ C, const float *__restrict__ A,
                        const float *__restrict__ B, int M, int N, int K) {
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

        // 只预取当前 tile 的四个独立向量，不增加 shared buffer。
        // 源码把 LDG 放在 STS 前；是否保留该顺序要检查最终 SASS。
        if constexpr (LOAD_ALL) {
            const float4 av0 = load_vector<FULL_TILE>(
                A, blockIdx.y * BM + a_row0, tile_k + a_col0, M, K);
            const float4 av1 = load_vector<FULL_TILE>(
                A, blockIdx.y * BM + a_row1, tile_k + a_col1, M, K);
            const float4 bv0 = load_vector<FULL_TILE>(
                B, tile_k + b_row0, blockIdx.x * BN + b_col0, K, N);
            const float4 bv1 = load_vector<FULL_TILE>(
                B, tile_k + b_row1, blockIdx.x * BN + b_col1, K, N);

            const int ar0 = swizzle_a_row(a_col0, a_row0);
            const int ar1 = swizzle_a_row(a_col1, a_row1);
            a_smem[a_col0][ar0] = av0.x;
            a_smem[a_col0 + 1][ar0] = av0.y;
            a_smem[a_col0 + 2][ar0] = av0.z;
            a_smem[a_col0 + 3][ar0] = av0.w;
            a_smem[a_col1][ar1] = av1.x;
            a_smem[a_col1 + 1][ar1] = av1.y;
            a_smem[a_col1 + 2][ar1] = av1.z;
            a_smem[a_col1 + 3][ar1] = av1.w;
            *reinterpret_cast<float4 *>(&b_smem[b_row0][b_col0]) = bv0;
            *reinterpret_cast<float4 *>(&b_smem[b_row1][b_col1]) = bv1;
        } else {
            // --------------------------------------------------------
            // A: global 按 K 维连续读取，shared 转置写入 [k][row]。
            // --------------------------------------------------------

            // 第一个 float4
            const int ga_row0 = blockIdx.y * BM + a_row0;
            const int ga_col0 = tile_k + a_col0;

            if (FULL_TILE || (a_aligned && ga_row0 < M && ga_col0 + 3 < K)) {

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
            const int ga_row1 = blockIdx.y * BM + a_row1;
            const int ga_col1 = tile_k + a_col1;

            if (FULL_TILE || (a_aligned && ga_row1 < M && ga_col1 + 3 < K)) {

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
            // B: global memory -> shared memory
            // --------------------------------------------------------

            // 第一个 float4
            const int gb_row0 = tile_k + b_row0;
            const int gb_col0 = blockIdx.x * BN + b_col0;

            if (FULL_TILE || (b_aligned && gb_row0 < K && gb_col0 + 3 < N)) {

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
            const int gb_row1 = tile_k + b_row1;
            const int gb_col1 = blockIdx.x * BN + b_col1;

            if (FULL_TILE || (b_aligned && gb_row1 < K && gb_col1 + 3 < N)) {

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

#pragma unroll UNROLL
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

        if (FULL_TILE || row < M) {
            float *c_ptr = C + static_cast<std::size_t>(row) * N + col0;

            // N 是 4 的倍数时，每行起始地址保持 16B 对齐。
            // col0 也是 4 的倍数，因此可以安全进行 float4 store。
            if (FULL_TILE || ((N % 4 == 0) && (col0 + 3 < N))) {

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

// Keep launch bounds out of the control and other single-factor experiments.
template <int UNROLL, bool LOAD_ALL, bool FULL_TILE>
__global__ void
sgemm_swizzle_experiment(float *__restrict__ C, const float *__restrict__ A,
                         const float *__restrict__ B, int M, int N, int K) {
    swizzle_experiment_body<64, 64, 32, 4, 4, UNROLL, LOAD_ALL, FULL_TILE>(
        C, A, B, M, N, K);
}

template <int UNROLL, bool LOAD_ALL, bool FULL_TILE>
__global__ __launch_bounds__(256, 4) void sgemm_swizzle_bounds4(
    float *__restrict__ C, const float *__restrict__ A,
    const float *__restrict__ B, int M, int N, int K) {
    swizzle_experiment_body<64, 64, 32, 4, 4, UNROLL, LOAD_ALL, FULL_TILE>(
        C, A, B, M, N, K);
}

template <int UNROLL, int MIN_BLOCKS, bool LOAD_ALL, bool FAST_PATH>
void launch_experiment(float *output, const float *A, const float *B, int M,
                       int N, int K, cudaStream_t stream) {
    if (M == 0 || N == 0) {
        return;
    }
    const dim3 block(16, 16);
    const dim3 grid((N + 63) / 64, (M + 63) / 64);
    // FULL_TILE is selected only when all tiles and vector addresses are valid.
    if constexpr (FAST_PATH) {
        if (M % 64 == 0 && N % 64 == 0 && K % 32 == 0) {
            sgemm_swizzle_experiment<UNROLL, LOAD_ALL, true>
                <<<grid, block, 0, stream>>>(output, A, B, M, N, K);
            return;
        }
    }
    if constexpr (MIN_BLOCKS == 4) {
        sgemm_swizzle_bounds4<UNROLL, LOAD_ALL, false>
            <<<grid, block, 0, stream>>>(output, A, B, M, N, K);
    } else {
        sgemm_swizzle_experiment<UNROLL, LOAD_ALL, false>
            <<<grid, block, 0, stream>>>(output, A, B, M, N, K);
    }
}

} // namespace

void launch_sgemm_float4_swizzle_control(float *output, const float *A,
                                         const float *B, int M, int N, int K,
                                         cudaStream_t stream) {
    launch_experiment<32, 1, false, false>(output, A, B, M, N, K, stream);
}

void launch_sgemm_float4_swizzle_u4(float *output, const float *A,
                                    const float *B, int M, int N, int K,
                                    cudaStream_t stream) {
    launch_experiment<4, 1, false, false>(output, A, B, M, N, K, stream);
}

void launch_sgemm_float4_swizzle_u8(float *output, const float *A,
                                    const float *B, int M, int N, int K,
                                    cudaStream_t stream) {
    launch_experiment<8, 1, false, false>(output, A, B, M, N, K, stream);
}

void launch_sgemm_float4_swizzle_bounds4(float *output, const float *A,
                                         const float *B, int M, int N, int K,
                                         cudaStream_t stream) {
    launch_experiment<32, 4, false, false>(output, A, B, M, N, K, stream);
}

void launch_sgemm_float4_swizzle_load_all(float *output, const float *A,
                                          const float *B, int M, int N, int K,
                                          cudaStream_t stream) {
    launch_experiment<32, 1, true, false>(output, A, B, M, N, K, stream);
}

void launch_sgemm_float4_swizzle_fast(float *output, const float *A,
                                      const float *B, int M, int N, int K,
                                      cudaStream_t stream) {
    launch_experiment<32, 1, false, true>(output, A, B, M, N, K, stream);
}

} // namespace cuda_learning
