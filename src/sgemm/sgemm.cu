#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "cuda_learning/kernels.h"

namespace cuda_learning {
namespace {

__global__ void sgemm_naive_kernel(float* output, const float* a,
                                   const float* b, int m, int n, int k) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0F;
    for (int inner = 0; inner < k; ++inner) {
        sum += a[row * k + inner] * b[inner * n + col];
    }
    output[row * n + col] = sum;
}

void launch_sgemm_naive(float* output, const float* a, const float* b, int m,
                        int n, int k, cudaStream_t stream) {
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    constexpr dim3 threads(16, 16);
    const dim3 blocks((n + threads.x - 1) / threads.x,
                      (m + threads.y - 1) / threads.y);
    sgemm_naive_kernel<<<blocks, threads, 0, stream>>>(output, a, b, m, n, k);
}

}  // namespace

const std::vector<SgemmImplementation>& sgemm_implementations() {
    static const std::vector<SgemmImplementation> implementations = {
        {"naive", launch_sgemm_naive},
    };
    return implementations;
}

const SgemmImplementation* find_sgemm_implementation(std::string_view name) {
    for (const auto& implementation : sgemm_implementations()) {
        if (implementation.name == name) {
            return &implementation;
        }
    }
    return nullptr;
}

void launch_sgemm(std::string_view implementation, float* output,
                  const float* a, const float* b, int m, int n, int k,
                  cudaStream_t stream) {
    const auto* selected = find_sgemm_implementation(implementation);
    if (selected == nullptr) {
        throw std::invalid_argument("unknown SGEMM implementation");
    }
    selected->launch(output, a, b, m, n, k, stream);
}

}  // namespace cuda_learning
