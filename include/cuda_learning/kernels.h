#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string_view>
#include <vector>

namespace cuda_learning {

using SgemmLauncher = void (*)(float* output, const float* a, const float* b,
                               int m, int n, int k, cudaStream_t stream);

struct SgemmImplementation {
    std::string_view name;
    SgemmLauncher launch;
};

enum class ReduceImplementation {
    NoBankConflict,
    AddDuringLoad,
    AddDuringLoadV2,
    UnrollLastWarp,
    CompletelyUnroll,
    MultiAdd,
    Shuffle,
};

void launch_square(float* output, const float* input, std::size_t n,
                   cudaStream_t stream);

// Row-major C[M, N] = A[M, K] * B[K, N].
void launch_sgemm(std::string_view implementation, float* output,
                  const float* a, const float* b, int m, int n, int k,
                  cudaStream_t stream);

const std::vector<SgemmImplementation>& sgemm_implementations();
const SgemmImplementation* find_sgemm_implementation(std::string_view name);

void launch_reduce(ReduceImplementation implementation, const float* input,
                   float* output, std::size_t n, cudaStream_t stream);

const char* implementation_name(ReduceImplementation implementation);

}  // namespace cuda_learning
