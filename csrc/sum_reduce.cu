/* Add reduce Kernel */

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

template<typename T>
__global__ void sum_reduce(const T *d_input, T *d_output) {
    
}