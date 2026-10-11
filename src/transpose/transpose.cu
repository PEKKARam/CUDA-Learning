#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_learning/kernels.h"

namespace cuda_learning {
namespace {
__global__ void transpose_naive(const float *input, float *output, const size_t N, const size_t M) {
    const size_t col = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < M && row < N) {
        size_t src_idx = row * M + col;
        size_t trans_idx = col * N + row;

        output[trans_idx] = input[src_idx];
    }
}

} // namespace

void launch_transpose_naive(const float *input, float *output, const size_t N, const size_t M, cudaStream_t stream) {
    if (N <= 0 || M <= 0) {
        return;
    }

    dim3 block_size(32, 8);
    dim3 gird_size((M + block_size.x - 1) / block_size.x,
                   (N + block_size.y - 1) / block_size.y);

    transpose_naive<<<block_size, gird_size, 0, stream>>>(input, output, N, M);
}

} // namespace cuda_learning