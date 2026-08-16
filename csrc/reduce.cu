/* Sum reduction kernel. */

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* reduce: no bank conflict version */
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

/*
reduce: add during load
    线程一次从全局内存搬运两个数据并且求和写入共享内存
    plan A: 保持block内线程数不变，减半block数量
*/
__global__ void reduce_add_during_load(const float* d_input, float* d_output,
                                       const size_t N) {
    extern __shared__ float smem[];

    size_t tid = threadIdx.x;
    size_t idx1 = blockIdx.x * (blockDim.x * 2) + tid;
    size_t idx2 = idx1 + blockDim.x;
    const float data1 = idx1 < N ? d_input[idx1] : 0.0f;
    const float data2 = idx2 < N ? d_input[idx2] : 0.0f;
    smem[tid] = data1 + data2;

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

/*
reduce: add during load
    线程一次从全局内存搬运两个数据并且求和写入共享内存
    plan B: 保持block数量不变，减半block内线程数（共享内存也要减半）
*/
__global__ void reduce_add_during_load_v2(const float* d_input, float* d_output,
                                          const size_t N) {
    extern __shared__ float smem[];

    size_t tid = threadIdx.x;
    size_t idx1 = blockIdx.x * (blockDim.x * 2) + tid;
    size_t idx2 = idx1 + blockDim.x;
    const float data1 = idx1 < N ? d_input[idx1] : 0.0f;
    const float data2 = idx2 < N ? d_input[idx2] : 0.0f;
    smem[tid] = data1 + data2;

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

/*
reduce: unroll_last_warp
    因为不同warp的线程运行不会同步，
    在add_during_load后，必须等所有元素操作完成
    因此在add之后需要同步，并且跨步长reduce时for循环内也要同步

    当循环进行到末尾阶段，剩余可执行的线程数量会减少到一个warp就可以包含，
    同一个warp内的线程是同步的，就会多运行几次__syncthreads()
    最后几次同步运行没有意义，反而会增加耗时
    解决同步问题
*/
__device__ void warpReduce(float* cache, unsigned int tid);

__global__ void reduce_unroll_last_warp(const float* d_input, float* d_output,
                                        const size_t N) {
    extern __shared__ float smem[];

    size_t tid = threadIdx.x;
    size_t idx1 = blockIdx.x * (blockDim.x * 2) + tid;
    size_t idx2 = idx1 + blockDim.x;
    const float data1 = idx1 < N ? d_input[idx1] : 0.0f;
    const float data2 = idx2 < N ? d_input[idx2] : 0.0f;
    smem[tid] = data1 + data2;

    __syncthreads();  // 同步一个block里的所有线程

    for (size_t stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(smem, tid);
    }
    if (tid == 0) {
        atomicAdd(d_output, smem[0]);
    }
}

__device__ void warpReduce(float* cache, unsigned int tid) {
    const unsigned int mask = __activemask();

    cache[tid] += cache[tid + 32];
    __syncwarp(mask);  // 内存访问同步
    if (tid < 16) cache[tid] += cache[tid + 16];
    __syncwarp(mask);
    if (tid < 8) cache[tid] += cache[tid + 8];
    __syncwarp(mask);
    if (tid < 4) cache[tid] += cache[tid + 4];
    __syncwarp(mask);
    if (tid < 2) cache[tid] += cache[tid + 2];
    __syncwarp(mask);
    if (tid < 1) cache[tid] += cache[tid + 1];
    __syncwarp(mask);
}

/*
reduce: completly unroll
    把循环展开, 降低循环带来的开销，block size = 256
*/
__global__ void reduce_completely_unroll(const float* d_input, float* d_output,
                                         const size_t N) {
    extern __shared__ float smem[];

    const size_t tid = threadIdx.x;
    const size_t idx1 = blockIdx.x * (blockDim.x * 2) + tid;
    const size_t idx2 = idx1 + blockDim.x;

    smem[tid] =
        (idx1 < N ? d_input[idx1] : 0.0f) + (idx2 < N ? d_input[idx2] : 0.0f);

    __syncthreads();  // 同步一个block里的所有线程}

    // loop completely unroll
    if (tid < 128) {
        smem[tid] += smem[tid + 128];
    }
    __syncthreads();

    if (tid < 64) {
        smem[tid] += smem[tid + 64];
    }
    __syncthreads();

    if (tid < 32) {
        warpReduce(smem, tid);
    }

    if (tid == 0) {  // block之间汇总
        atomicAdd(d_output, smem[0]);
    }
}

/*
reduce: multi add
    让每个线程处理的数据更多
*/
template <const size_t NUM_PER_THREAD, const size_t THREAD_PER_BLOCK>
__global__ void reduce_multi_add(const float* d_input, float* d_output,
                                 const size_t N) {
    extern __shared__ float smem[];

    const size_t tid = threadIdx.x;
    const size_t idx = tid + blockIdx.x * (THREAD_PER_BLOCK * NUM_PER_THREAD);
    smem[tid] = 0.0f;
#pragma unroll
    for (size_t i = 0; i < NUM_PER_THREAD; ++i) {
        const size_t input_idx = idx + i * THREAD_PER_BLOCK;
        if (input_idx < N) {
            smem[tid] += d_input[input_idx];
        }
    }

    __syncthreads();  // 同步一个block里的所有线程

    // loop completely unroll
    if (tid < 128) {
        smem[tid] += smem[tid + 128];
    }
    __syncthreads();

    if (tid < 64) {
        smem[tid] += smem[tid + 64];
    }
    __syncthreads();

    if (tid < 32) {
        warpReduce(smem, tid);
    }

    if (tid == 0) {  // block之间汇总
        atomicAdd(d_output, smem[0]);
    }
}

/*
reduce shuffle:
    Shuffle指令是一组针对warp的指令。Shuffle指令最重要的特性就是warp内的寄存器可以相互访问。
    在没有shuffle指令的时候，各个线程在进行通信时只能通过shared
    memory来访问彼此的寄存器。
    而采用了shuffle指令之后，warp内的线程可以直接对其他线程的寄存器进行访存。
    通过这种方式可以减少访存的延时。除此之外，带来的最大好处就是可编程性提高了，
    在某些场景下，就不用shared memory了。
    毕竟，开发者要自己去控制 shared memory还是挺麻烦的一个事。
*/

__device__  float warp_reduce_shuffle(float value) {
    constexpr unsigned int FULL_MASK = 0xffffffffu;
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(FULL_MASK, value, offset);
    }
    return value;
}

template <size_t NUM_PER_THREAD, size_t THREAD_PER_BLOCK>
__global__ void reduce_shuffle(const float* input, float* output,
                               const size_t N) {
    constexpr size_t WARP_SIZE = 32;
    static_assert(THREAD_PER_BLOCK % WARP_SIZE == 0,
                  "THREAD_PER_BLOCK must be a multiple of warp size");
    constexpr size_t NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;

    float sum = 0.0f;

    const size_t tid = threadIdx.x;
    const size_t idx = tid + blockIdx.x * (THREAD_PER_BLOCK * NUM_PER_THREAD);

#pragma unroll
    for (size_t i = 0; i < NUM_PER_THREAD; ++i) {
        const size_t input_idx = idx + i * THREAD_PER_BLOCK;
        if (input_idx < N) {
            sum += input[input_idx];
        }
    }

    // Level 1: each warp reduces its 32 register values to lane 0.
    sum = warp_reduce_shuffle(sum);

    __shared__ float warp_sums[NUM_WARPS];
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Level 2: the first warp reduces the per-warp sums.
    if (warp_id == 0) {
        sum = lane_id < NUM_WARPS ? warp_sums[lane_id] : 0.0f;
        sum = warp_reduce_shuffle(sum);

        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}
/*
  ==============================================================
  ------------------------- Launchers --------------------------
  ==============================================================
*/

void launch_reduce_no_bankconflict_kernel(const float* input, float* output,
                                          const size_t N, dim3 grid_size,
                                          dim3 block_size) {
    auto stream = c10::cuda::getCurrentCUDAStream();
    const size_t shared_memory = block_size.x * sizeof(float);
    reduce_no_bankconflict<<<grid_size, block_size, shared_memory, stream>>>(
        input, output, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_reduce_add_during_load_kernel(const float* input, float* output,
                                          const size_t N, dim3 grid_size,
                                          dim3 block_size) {
    auto stream = c10::cuda::getCurrentCUDAStream();
    const size_t shared_memory = block_size.x * sizeof(float);
    reduce_add_during_load<<<grid_size, block_size, shared_memory, stream>>>(
        input, output, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_reduce_add_during_load_v2_kernel(const float* input, float* output,
                                             const size_t N, dim3 grid_size,
                                             dim3 block_size) {
    auto stream = c10::cuda::getCurrentCUDAStream();
    block_size.x /= 2;
    const size_t shared_memory = block_size.x * sizeof(float);
    reduce_add_during_load_v2<<<grid_size, block_size, shared_memory, stream>>>(
        input, output, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_reduce_unroll_last_warp_kernel(const float* input, float* output,
                                           const size_t N, dim3 grid_size,
                                           dim3 block_size) {
    auto stream = c10::cuda::getCurrentCUDAStream();
    block_size.x /= 2;
    const size_t shared_memory = block_size.x * sizeof(float);
    reduce_unroll_last_warp<<<grid_size, block_size, shared_memory, stream>>>(
        input, output, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_reduce_completely_unroll_kernel(const float* input, float* output,
                                            const size_t N, dim3 grid_size,
                                            dim3 block_size) {
    auto stream = c10::cuda::getCurrentCUDAStream();
    const size_t shared_memory = block_size.x * sizeof(float);
    reduce_completely_unroll<<<grid_size, block_size, shared_memory, stream>>>(
        input, output, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_reduce_multi_add_kernel(const float* input, float* output,
                                    const size_t N) {
    constexpr size_t NUM_PER_THREAD = 4;
    constexpr size_t THREAD_PER_BLOCK = 256;
    constexpr size_t NUM_PER_BLOCK = NUM_PER_THREAD * THREAD_PER_BLOCK;
    const dim3 block_size(THREAD_PER_BLOCK);
    const dim3 grid_size((N + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK);
    auto stream = c10::cuda::getCurrentCUDAStream();
    constexpr size_t shared_memory = THREAD_PER_BLOCK * sizeof(float);
    reduce_multi_add<NUM_PER_THREAD, THREAD_PER_BLOCK>
        <<<grid_size, block_size, shared_memory, stream>>>(input, output, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_reduce_shuffle_kernel(const float* input, float* output,
                                  const size_t N) {
    constexpr size_t NUM_PER_THREAD = 4;
    constexpr size_t THREAD_PER_BLOCK = 256;
    constexpr size_t NUM_PER_BLOCK = NUM_PER_THREAD * THREAD_PER_BLOCK;
    const dim3 block_size(THREAD_PER_BLOCK);
    const dim3 grid_size((N + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK);
    auto stream = c10::cuda::getCurrentCUDAStream();

    reduce_shuffle<NUM_PER_THREAD, THREAD_PER_BLOCK>
        <<<grid_size, block_size, 0, stream>>>(input, output, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
