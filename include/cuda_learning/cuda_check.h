#pragma once

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        const cudaError_t cuda_check_error = (call);                                               \
        if (cuda_check_error != cudaSuccess) {                                                     \
            std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__ << ": "                   \
                      << cudaGetErrorString(cuda_check_error) << std::endl;                        \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (false)
