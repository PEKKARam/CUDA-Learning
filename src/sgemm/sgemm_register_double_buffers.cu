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
__global__ void sgemm_register_double_buffers(
    float* __restrict__ C, const float* __restrict__ A,
    const float* __restrict__ B, int M, int N, int K) {
    constexpr int PAD = 4;
    __shared__ float a_tile[BM][BK + PAD];
    __shared__ float b_tile[BK][BN + PAD];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row0 = blockIdx.y * BM + ty * TM;
    const int col0 = blockIdx.x * BN + tx * TN;

    float sum[TM][TN] = {};

    // Every thread moves eight values for A and B (two float4 vectors).
    const int a_row = blockIdx.y * BM + tid / (BK / 8);
    const int a_col_offset = (tid % (BK / 8)) * 8;
    const int b_row_offset = tid / (BN / 8);
    const int b_col = blockIdx.x * BN + (tid % (BN / 8)) * 8;
    const bool vector_aligned = (K % 4 == 0) && (N % 4 == 0);

    for (int tile_k = 0; tile_k < K; tile_k += BK) {
        const int a_col = tile_k + a_col_offset;
        const int b_row = tile_k + b_row_offset;

        if (vector_aligned && a_row < M && a_col + 7 < K) {
            *reinterpret_cast<float4*>(&a_tile[a_row - blockIdx.y * BM][a_col_offset]) =
                *reinterpret_cast<const float4*>(A + static_cast<std::size_t>(a_row) * K + a_col);
            *reinterpret_cast<float4*>(&a_tile[a_row - blockIdx.y * BM][a_col_offset + 4]) =
                *reinterpret_cast<const float4*>(A + static_cast<std::size_t>(a_row) * K + a_col + 4);
        } else {
            const int local_row = a_row - blockIdx.y * BM;
            for (int v = 0; v < 8; ++v) {
                const int c = a_col + v;
                a_tile[local_row][a_col_offset + v] =
                    (a_row < M && c < K) ? A[static_cast<std::size_t>(a_row) * K + c] : 0.0f;
            }
        }

        if (vector_aligned && b_row < K && b_col + 7 < N) {
            *reinterpret_cast<float4*>(&b_tile[b_row_offset][(b_col - blockIdx.x * BN)]) =
                *reinterpret_cast<const float4*>(B + static_cast<std::size_t>(b_row) * N + b_col);
            *reinterpret_cast<float4*>(&b_tile[b_row_offset][(b_col - blockIdx.x * BN) + 4]) =
                *reinterpret_cast<const float4*>(B + static_cast<std::size_t>(b_row) * N + b_col + 4);
        } else {
            const int local_col = b_col - blockIdx.x * BN;
            for (int v = 0; v < 8; ++v) {
                const int c = b_col + v;
                b_tile[b_row_offset][local_col + v] =
                    (b_row < K && c < N) ? B[static_cast<std::size_t>(b_row) * N + c] : 0.0f;
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
                    a_reg[next][r] = a_tile[ty * TM + r][next_inner];
                }
                b_reg[next] = *reinterpret_cast<const float4*>(&b_tile[next_inner][tx * TN]);
            }

            if (inner == 0) {
#pragma unroll
                for (int r = 0; r < TM; ++r) {
                    a_reg[cur][r] = a_tile[ty * TM + r][0];
                }
                b_reg[cur] = *reinterpret_cast<const float4*>(&b_tile[0][tx * TN]);
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

    for (int r = 0; r < TM; ++r) {
        const int row = row0 + r;
        if (row >= M) break;
        for (int c = 0; c < TN; ++c) {
            if (col0 + c < N) C[static_cast<std::size_t>(row) * N + col0 + c] = sum[r][c];
        }
    }
}

}  // namespace

void launch_sgemm_register_double_buffers(float* output, const float* a,
                                           const float* b, int m, int n, int k,
                                           cudaStream_t stream) {
    if (m == 0 || n == 0 || k == 0) return;
    constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
    const dim3 block(BN / TN, BM / TM);
    const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    sgemm_register_double_buffers<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(output, a, b, m, n, k);
}

}  // namespace cuda_learning
