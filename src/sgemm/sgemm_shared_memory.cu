#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "cuda_learning/kernels.h"

namespace cuda_learning {

template <const int BM, const int BN, const int BK>
__global__ void sgemm_shared_memory(float *output, const float *A,
                                    const float *B, int M, int N, int K) {
    __shared__ float a_shared[BM][BK];
    __shared__ float b_shared[BK][BN];

    // block inner tid
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int tx = threadIdx.x;                      // col
    const int ty = threadIdx.y;                      // row
    const int threads_num = blockDim.x * blockDim.y; // threads number

    // global tid
    const int col = blockDim.x * blockIdx.x + tx;
    const int row = blockDim.y * blockIdx.y + ty;

    float sum = 0.0F;

    for (int tile = 0; tile < K; tile += BK) {
        // 沿 K 维分块存入shared memory, 一个线程可能需要存储多个数据
        for (int idx = tid; idx < BM * BK; idx += threads_num) {
            int r = idx / BK;
            int c = idx % BK;

            int g_r = blockIdx.y * BM + r;
            int g_k = tile + c;

            a_shared[r][c] = (g_r < M && g_k < K) ? A[g_r * K + g_k] : 0.0f;
        }

        for (int idx = tid; idx < BK * BN; idx += threads_num) {
            int r = idx / BN;
            int c = idx % BN;

            int g_k = tile + r;
            int g_c = blockIdx.x * BN + c;

            b_shared[r][c] = (g_k < K && g_c < N) ? B[g_k * N + g_c] : 0.0f;
        }
        // const int a_offset = tile + tx; // A col offset
        // a_shared[ty][tx]
        //     = (row < M && a_offset < K) ? A[row * K + a_offset] : 0.0f;
        // const int b_offset = tile + ty; // B row offset
        // b_shared[ty][tx]
        //     = (b_offset < K && col < N) ? B[b_offset * N + col] : 0.0f;

        __syncthreads(); // 两次同步会引入额外开销，可能导致性能更差

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            sum += a_shared[ty][k] * b_shared[k][tx];
        }

        if (tile + BK < K) {
            __syncthreads();
        }
    }
    if (row < M && col < N) {
        output[row * N + col] = sum;
    }
}

void launch_sgemm_shared_memory(float *output, const float *a, const float *b,
                                int m, int n, int k, cudaStream_t stream) {
    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    constexpr int BM = 16, BN = 16, BK = 32;
    constexpr dim3 BLOCK_SIZE(BN, BM);
    const dim3 GRID_SIZE((n + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
                         (m + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);
    sgemm_shared_memory<BM, BN, BK>
        <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(output, a, b, m, n, k);
}

} // namespace cuda_learning
