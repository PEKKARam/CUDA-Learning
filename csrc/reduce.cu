/* Sum reduction kernel. */

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void reduce_no_bankconflict(const float* d_input, float* d_output,
                                       const size_t N) {
    extern __shared__ float smem[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    smem[tid] = idx < N ? d_input[idx] : 0.0f;

    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_output, smem[0]);
    }
}

void launch_reduce_no_bankconflict_kernel(const float* input, float* output,
                                          const size_t N, dim3 grid_size,
                                          dim3 block_size) {
    auto stream = c10::cuda::getCurrentCUDAStream();
    const size_t shared_memory = block_size.x * sizeof(float);
    reduce_no_bankconflict<<<grid_size, block_size, shared_memory, stream>>>(input, output, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
