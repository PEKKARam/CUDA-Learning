#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_learning/kernels.h"

namespace cuda_learning {
namespace {

// BK=16: a warp loads four float4 segments from each of eight A rows.
// XOR k-group into row bits 3:4, leaving the four-float vector intact.
__device__ __forceinline__ int swizzle_a_row(int k, int row) {
    return row ^ ((k >> 2) << 3);
}

template <bool FULL_TILE>
__device__ __forceinline__ float4 load_global4(const float *matrix, int row,
                                               int col, int rows, int stride) {
    if constexpr (FULL_TILE) {
        return *reinterpret_cast<const float4 *>(
            matrix + static_cast<std::size_t>(row) * stride + col);
    } else {
        if (stride % 4 == 0 && row < rows && col + 3 < stride) {
            return *reinterpret_cast<const float4 *>(
                matrix + static_cast<std::size_t>(row) * stride + col);
        }
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

template <int BM, int BN, int BK, int AG, int BG>
__device__ __forceinline__ void
load_fragments(float4 (&af)[AG], float4 (&bf)[BG], const float (&a)[BK][BM],
               const float (&b)[BK][BN], int k, int tx, int ty) {
#pragma unroll
    for (int mi = 0; mi < AG; ++mi) {
        const int row = ty * 4 + mi * (BM / AG);
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
    constexpr int AG = TM / 4, BG = TN / 4;
    constexpr int THREADS_X = BN / TN, THREADS_Y = BM / TM;
    constexpr int THREADS = THREADS_X * THREADS_Y;
    constexpr int LOAD_A = BM * BK / (THREADS * 4);
    constexpr int LOAD_B = BK * BN / (THREADS * 4);
    static_assert(BK == 16 && BM % 32 == 0 && BN % 32 == 0);
    static_assert((TM == 4 || TM == 8) && TN == 8);
    static_assert(THREADS_X == 16 && THREADS_Y == 16);
    static_assert(BM * BK % (THREADS * 4) == 0 && BK * BN % (THREADS * 4) == 0);

    __shared__ __align__(16) float a_smem[BK][BM];
    __shared__ __align__(16) float b_smem[BK][BN];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;

    // 8 columns are two four-column stripes, separated by BN/2.
    // For TM=8 the rows are similarly split at BM/2. Every sub-tile is 4x4.
    float sum[AG][BG][4][4] = {};

    for (int tile_k = 0; tile_k < K; tile_k += BK) {
        float4 global_a[LOAD_A];
        float4 global_b[LOAD_B];
        // Preload all independent vectors before shared stores. The number
        // of vectors is derived from the tile instead of assuming two each.
#pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            const int index = tid * 4 + i * THREADS * 4;
            global_a[i] = load_global4<FULL_TILE>(
                A, blockIdx.y * BM + index / BK, tile_k + index % BK, M, K);
        }
#pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            const int index = tid * 4 + i * THREADS * 4;
            global_b[i] = load_global4<FULL_TILE>(
                B, tile_k + index / BN, blockIdx.x * BN + index % BN, K, N);
        }
#pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            const int index = tid * 4 + i * THREADS * 4;
            const int row = index / BK, k = index % BK;
            const int physical_row = swizzle_a_row(k, row);
            a_smem[k][physical_row] = global_a[i].x;
            a_smem[k + 1][physical_row] = global_a[i].y;
            a_smem[k + 2][physical_row] = global_a[i].z;
            a_smem[k + 3][physical_row] = global_a[i].w;
        }
#pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            const int index = tid * 4 + i * THREADS * 4;
            *reinterpret_cast<float4 *>(&b_smem[index / BN][index % BN])
                = global_b[i];
        }
        __syncthreads();

        float4 a_reg[2][AG], b_reg[2][BG];
        load_fragments<BM, BN, BK, AG, BG>(a_reg[0], b_reg[0], a_smem, b_smem,
                                           0, tx, ty);
        int cur = 0;
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const int next = cur ^ 1;
            if (k + 1 < BK) {
                load_fragments<BM, BN, BK, AG, BG>(
                    a_reg[next], b_reg[next], a_smem, b_smem, k + 1, tx, ty);
            }
#pragma unroll
            for (int mi = 0; mi < AG; ++mi) {
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
                            sum[mi][ni][r][c]
                                = fmaf(af[r], bf[c], sum[mi][ni][r][c]);
                        }
                    }
                }
            }
            cur = next;
        }
        __syncthreads();
    }

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
                if constexpr (FULL_TILE) {
                    *reinterpret_cast<float4 *>(
                        C + static_cast<std::size_t>(row) * N + col)
                        = out;
                } else if (row < M) {
                    if (N % 4 == 0 && col + 3 < N) {
                        *reinterpret_cast<float4 *>(
                            C + static_cast<std::size_t>(row) * N + col)
                            = out;
                    } else {
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

void launch_sgemm_float4_v3_swizzle_fulltile_rt8x8(float *output,
                                                   const float *A,
                                                   const float *B, int M, int N,
                                                   int K, cudaStream_t stream) {
    constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
    if (M == 0 || N == 0) {
        return;
    }
    const bool full_tile = M % BM == 0 && N % BN == 0 && K % BK == 0;
    const dim3 block(BN / TN, BM / TM);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    if (full_tile) {
        sgemm_float4_v3_swizzle_fulltile_rt8x8<BM, BN, BK, TM, TN, true>
            <<<grid, block, 0, stream>>>(output, A, B, M, N, K);
    } else {
        sgemm_float4_v3_swizzle_fulltile_rt8x8<BM, BN, BK, TM, TN, false>
            <<<grid, block, 0, stream>>>(output, A, B, M, N, K);
    }
}
} // namespace cuda_learning
