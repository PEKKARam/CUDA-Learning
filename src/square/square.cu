#include <cuda_runtime.h>

#include "cuda_learning/kernels.h"

namespace cuda_learning {
namespace {

__global__ void square_kernel(float* output, const float* input,
                              std::size_t n) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = input[index] * input[index];
    }
}

}  // namespace

void launch_square(float* output, const float* input, std::size_t n,
                   cudaStream_t stream) {
    if (n == 0) {
        return;
    }

    constexpr unsigned int threads = 256;
    const unsigned int blocks =
        static_cast<unsigned int>((n + threads - 1) / threads);
    square_kernel<<<blocks, threads, 0, stream>>>(output, input, n);
}

}  // namespace cuda_learning
