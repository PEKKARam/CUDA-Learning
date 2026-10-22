#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_learning/kernels.h"

namespace cuda_learning {
namespace {

// Register ping-pong version:
// each thread computes a 4x4 output tile. During the FMA for slot `cur`,
// slot `next` is filled from shared memory. This is register prefetching,
// not global-memory asynchronous copy.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_register_double_buffers(float *__restrict__ C,
                                              const float *__restrict__ A,
                                              const float *__restrict__ B,
                                              int M, int N, int K) {
    constexpr int PAD = 4;
    constexpr int THREADS = (BM / TM) * (BN / TN);
    static_assert(BM * BK == THREADS * 8 && BK * BN == THREADS * 8,
                  "Each thread must load two float4 vectors per tile.");

    constexpr int A_HALF_TILE = BM * BK / 2;
    constexpr int B_HALF_TILE = BK * BN / 2;

    __shared__ float a_smem[BM][BK + PAD];
    __shared__ float b_smem[BK][BN + PAD];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // 当前线程处理的区域 在output对应的起始地址
    const int row0 = blockIdx.y * BM + ty * TM;
    const int col0 = blockIdx.x * BN + tx * TN;

    float sum[TM][TN];
#pragma unroll
    for (int r = 0; r < TM; ++r) {
#pragma unroll
        for (int c = 0; c < TN; ++c) {
            sum[r][c] = 0.0f;
        }
    }

    // 与 float4_v2 一致：相邻线程的 float4 起始位置相差 4 个 float，
    // 第二个 float4 位于 tile 的后半部分。
    // A load
    const int a_i0 = tid * 4;
    const int a_i1 = a_i0 + A_HALF_TILE;

    const int a_row0 = a_i0 / BK;
    const int a_col0 = a_i0 % BK;

    const int a_row1 = a_i1 / BK;
    const int a_col1 = a_i1 % BK;

    // B load
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
        const int ga_row0 = blockIdx.y * BM + a_row0;
        const int ga_col0 = tile_k + a_col0;

        // first float4 of A
        if (a_aligned && ga_row0 < M && ga_col0 + 3 < K) {
            const float4 v = *reinterpret_cast<const float4 *>(
                A + static_cast<std::size_t>(ga_row0) * K + ga_col0);

            *reinterpret_cast<float4 *>(&a_smem[a_row0][a_col0]) = v;

        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                a_smem[a_row0][a_col0 + v]
                    = (ga_row0 < M && ga_col0 + v < K)
                        ? A[static_cast<std::size_t>(ga_row0) * K + ga_col0 + v]
                        : 0.0f;
            }
        }

        // second float4 of A
        const int ga_row1 = blockIdx.y * BM + a_row1;
        const int ga_col1 = tile_k + a_col1;

        if (a_aligned && ga_row1 < M && ga_col1 + 3 < K) {

            const float4 v = *reinterpret_cast<const float4 *>(
                A + static_cast<std::size_t>(ga_row1) * K + ga_col1);

            *reinterpret_cast<float4 *>(&a_smem[a_row1][a_col1]) = v;

        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                a_smem[a_row1][a_col1 + v]
                    = (ga_row1 < M && ga_col1 + v < K)
                        ? A[static_cast<std::size_t>(ga_row1) * K + ga_col1 + v]
                        : 0.0f;
            }
        }

        // B
        // 第一个 float4
        const int gb_row0 = tile_k + b_row0;
        const int gb_col0 = blockIdx.x * BN + b_col0;

        if (b_aligned && gb_row0 < K && gb_col0 + 3 < N) {

            const float4 v = *reinterpret_cast<const float4 *>(
                B + static_cast<std::size_t>(gb_row0) * N + gb_col0);

            *reinterpret_cast<float4 *>(&b_smem[b_row0][b_col0]) = v;

        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                b_smem[b_row0][b_col0 + v]
                    = (gb_row0 < K && gb_col0 + v < N)
                        ? B[static_cast<std::size_t>(gb_row0) * N + gb_col0 + v]
                        : 0.0f;
            }
        }

        // 第二个 float4
        const int gb_row1 = tile_k + b_row1;
        const int gb_col1 = blockIdx.x * BN + b_col1;

        if (b_aligned && gb_row1 < K && gb_col1 + 3 < N) {

            const float4 v = *reinterpret_cast<const float4 *>(
                B + static_cast<std::size_t>(gb_row1) * N + gb_col1);

            *reinterpret_cast<float4 *>(&b_smem[b_row1][b_col1]) = v;

        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                b_smem[b_row1][b_col1 + v]
                    = (gb_row1 < K && gb_col1 + v < N)
                        ? B[static_cast<std::size_t>(gb_row1) * N + gb_col1 + v]
                        : 0.0f;
            }
        }

        __syncthreads();

        // Two register slots. A slot contains TM A values and one B float4.
        float a_reg[2][TM];
        float4 b_reg[2];
        int cur = 0;
        int next = 1;

        for (int inner = 0; inner < BK; ++inner) {
            // Prefetch the next k position before using the current slot.
            const int next_inner = inner + 1;
            if (next_inner < BK) {
#pragma unroll
                for (int r = 0; r < TM; ++r) {
                    a_reg[next][r] = a_smem[ty * TM + r][next_inner];
                }
                b_reg[next] = *reinterpret_cast<const float4 *>(
                    &b_smem[next_inner][tx * TN]);
            }

            if (inner == 0) {
#pragma unroll
                for (int r = 0; r < TM; ++r) {
                    a_reg[cur][r] = a_smem[ty * TM + r][0];
                }
                b_reg[cur]
                    = *reinterpret_cast<const float4 *>(&b_smem[0][tx * TN]);
            }

#pragma unroll
            for (int r = 0; r < TM; ++r) {
                sum[r][0] = fmaf(a_reg[cur][r], b_reg[cur].x, sum[r][0]);
                sum[r][1] = fmaf(a_reg[cur][r], b_reg[cur].y, sum[r][1]);
                sum[r][2] = fmaf(a_reg[cur][r], b_reg[cur].z, sum[r][2]);
                sum[r][3] = fmaf(a_reg[cur][r], b_reg[cur].w, sum[r][3]);
            }
            const int tmp = cur;
            cur = next;
            next = tmp;
        }
        __syncthreads();
    }

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

void launch_sgemm_register_double_buffers(float *output, const float *a,
                                          const float *b, int m, int n, int k,
                                          cudaStream_t stream) {
    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
    const dim3 block(BN / TN, BM / TM);
    const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    sgemm_register_double_buffers<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(output, a, b, m, n, k);
}

} // namespace cuda_learning
