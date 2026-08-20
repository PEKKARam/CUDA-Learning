#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "cuda_learning/kernels.h"

namespace cuda_learning {

__global__ void sgemm_shared_memory(float *output, const float *A, const float *B, int M, int N,
                                    int K) {
    constexpr int BM = 16, BN = 16, BK = 16;
    __shared__ float a_shared[BM][BK];
    __shared__ float b_shared[BK][BN];

    // block inner tid
    const int tx = threadIdx.x; // col
    const int ty = threadIdx.y; // row

    // global tid
    const int col = blockDim.x * blockIdx.x + tx;
    const int row = blockDim.y * blockIdx.y + ty;

    float sum = 0.0F;

    for (int tile = 0; tile < K; tile += BK) { // 沿 K 维分块存入shared memory
        const int a_offset = tile + tx;        // A col offset
        a_shared[ty][tx] = (row < M && a_offset < K) ? A[row * K + a_offset] : 0.0f;
        const int b_offset = tile + ty; // B row offset
        b_shared[ty][tx] = (b_offset < K && col < N) ? B[b_offset * N + col] : 0.0f;

        __syncthreads(); // 两次同步会引入额外开销，可能导致性能更差

#pragma unroll
        for (int k = 0; k < BK; ++k) { sum += a_shared[ty][k] * b_shared[k][tx]; }
        __syncthreads();
    }
    if (row < M && col < N) {
        output[row * N + col] = sum;
    }
}

void launch_sgemm_shared_memory(float *output, const float *a, const float *b, int m, int n, int k,
                                cudaStream_t stream) {
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    constexpr dim3 BLOCK_SIZE(16, 16);
    const dim3 GRID_SIZE((n + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
                         (m + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);
    sgemm_shared_memory<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(output, a, b, m, n, k);
}

} // namespace cuda_learning
