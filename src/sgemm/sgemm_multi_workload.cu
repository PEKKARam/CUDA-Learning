#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "cuda_learning/kernels.h"

namespace cuda_learning {

template <const int STRIDE>
__global__ void sgemm_multi_workload(float *output, const float *A, const float *B, int M, int N,
                                     int K) {
    constexpr int BM = 16, BN = 16, BK = 16;
    constexpr int STEP = BK * STRIDE; // tiling processing step length
    __shared__ float a_shared[BM * STRIDE][STEP];
    __shared__ float b_shared[STEP][BN * STRIDE];

    // block inner thread index
    const int tx = threadIdx.x; // col
    const int ty = threadIdx.y; // row

    // global memory output index in ouput matrix of current thread
    const int row = (blockDim.y * STRIDE) * blockIdx.y + ty;
    const int col = (blockDim.x * STRIDE) * blockIdx.x + tx;

    // global index of current block
    const int block_row = (blockDim.y * STRIDE) * blockIdx.y;
    const int block_col = (blockDim.x * STRIDE) * blockIdx.x;

    float sum[STRIDE][STRIDE] = {}; // 存放2 * 2个元素，一个线程计算四个和

    // tiling
    for (int tile = 0; tile < K; tile += STEP) {
        // 一个线程拿四个数，一个双重循环
        for (int r = 0; r < STRIDE; ++r) {
            for (int c = 0; c < STRIDE; ++c) {
                const int row_s = r * blockDim.y;
                const int col_s = c * blockDim.x;

                const int a_row = block_row + ty + row_s;
                const int a_col = tile + tx + col_s;

                const int b_row = tile + ty + row_s;
                const int b_col = block_col + tx + col_s;

                a_shared[ty + row_s][tx + col_s]
                    = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
                b_shared[ty + row_s][tx + col_s]
                    = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
            }
        }
        __syncthreads();

        // caculator
        for (int r = 0; r < STRIDE; ++r) {
            for (int c = 0; c < STRIDE; ++c) {
                const int row_s = r * blockDim.y;
                const int col_s = c * blockDim.x;
                for (int k = 0; k < STEP; ++k) {
                    sum[r][c] += a_shared[ty + row_s][k] * b_shared[k][tx + col_s];
                }
            }
        }
        __syncthreads();
    }

    // write back
    for (int r = 0; r < STRIDE; ++r) {
        for (int c = 0; c < STRIDE; ++c) {
            const int output_row = row + r * blockDim.y;
            const int output_col = col + c * blockDim.x;

            if (output_row < M && output_col < N) {
                output[output_row * N + output_col] = sum[r][c];
            }
        }
    }
}

void launch_sgemm_multi_workload(float *output, const float *A, const float *B, int M, int N, int K,
                                 cudaStream_t stream) {
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    constexpr int STRIDE = 2;
    constexpr dim3 BLOCK_SIZE(16, 16);
    constexpr int OUTPUT_TILE = BLOCK_SIZE.x * STRIDE;
    // block数量减成原来的四分之一，每个线程处理4倍数据
    const dim3 GRID_SIZE((N + OUTPUT_TILE - 1) / OUTPUT_TILE, (M + OUTPUT_TILE - 1) / OUTPUT_TILE);

    sgemm_multi_workload<STRIDE><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(output, A, B, M, N, K);
}

} // namespace cuda_learning
