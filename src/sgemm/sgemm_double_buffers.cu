#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_learning/kernels.h"

namespace cuda_learning {

namespace {

template <int BM, int BK, int PAD>
__device__ __forceinline__ void
loadA_to_buffer(const float *__restrict__ A, int M, int K,
                float (*a_smem)[BM][BK + PAD], int load_idx, int ga_r, int ga_c,
                int a_lr, int a_lk, const bool aligned) {

    if (aligned && ga_r < M && ga_c + 7 < K) { // 快路径：纯 float4
        const float4 v0 = *reinterpret_cast<const float4 *>(
            A + (std::size_t)ga_r * K + ga_c);
        const float4 v1 = *reinterpret_cast<const float4 *>(
            A + (std::size_t)ga_r * K + ga_c + 4);
        *reinterpret_cast<float4 *>(&a_smem[load_idx][a_lr][a_lk]) = v0;
        *reinterpret_cast<float4 *>(&a_smem[load_idx][a_lr][a_lk + 4]) = v1;
    } else { // 慢路径：只有边缘 tile 走
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            a_smem[load_idx][a_lr][a_lk + v]
                = (ga_r < M && ga_c + v < K)
                    ? A[(std::size_t)ga_r * K + ga_c + v]
                    : 0.0f;
        }
    }
}

template <int BK, int BN, int PAD>
__device__ __forceinline__ void
loadB_to_buffer(const float *__restrict__ B, int N, int K,
                float (*b_smem)[BK][BN + PAD], int load_idx, int gb_r, int gb_c,
                int b_lk, int b_lc, const bool aligned) {

    if (aligned && gb_r < K && gb_c + 7 < N) {
        const float4 v0 = *reinterpret_cast<const float4 *>(
            B + (std::size_t)gb_r * N + gb_c);
        const float4 v1 = *reinterpret_cast<const float4 *>(
            B + (std::size_t)gb_r * N + gb_c + 4);
        *reinterpret_cast<float4 *>(&b_smem[load_idx][b_lk][b_lc]) = v0;
        *reinterpret_cast<float4 *>(&b_smem[load_idx][b_lk][b_lc + 4]) = v1;
    } else {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            b_smem[load_idx][b_lk][b_lc + v]
                = (gb_r < K && gb_c + v < N)
                    ? B[(std::size_t)gb_r * N + gb_c + v]
                    : 0.0f;
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void
sgemm_double_buffers(float *__restrict__ C, const float *__restrict__ A,
                     const float *__restrict__ B, int M, int N, int K) {
    constexpr int PAD = 4;

    __shared__ float a_smem[2][BM][BK + PAD];
    __shared__ float b_smem[2][BK][BN + PAD];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // block inner tid
    const int tx = threadIdx.x;                             // 0..15，负责 TN 列
    const int ty = threadIdx.y;                             // 0..15，负责 TM 行

    // start index of processing area
    const int row0 = blockIdx.y * BM + ty * TM;
    const int col0 = blockIdx.x * BN + tx * TN;

    float sum[TM][TN] = {0.0f}; // accelarators of each thread

    // 每线程负责搬运 8 个 float，跨8个bank（两个 float4) load_row/load_col
    // A矩阵 BK / 8 = 4: 一行4个线程，每个线程搬运8个float
    const int a_lr = tid / (BK / 8);       // A tile 的行 0..BM-1
    const int a_lk = (tid % (BK / 8)) * 8; // A tile 的列 0,8,16,24

    // B矩阵 BN / 8 = 8： 一行八个线程，每个线程搬运8个float
    const int b_lk = tid / (BN / 8);       // B tile 的行 0..BK-1
    const int b_lc = (tid % (BN / 8)) * 8; // B tile 的列 0..BN-8

    const bool aligned = (K % 4 == 0) && (N % 4 == 0); // 循环不变量，只判一次

    // ---------- 预加载 A 子块 [BM, BK] ----------
    const int ga_r = blockIdx.y * BM + a_lr;
    const int ga_c = a_lk;

    loadA_to_buffer<BM, BK, PAD>(A, M, K, a_smem, 0, ga_r, ga_c, a_lr, a_lk,
                            aligned);

    // ---------- 预加载 B 子块 [BK, BN] ----------
    const int gb_r = b_lk;
    const int gb_c = blockIdx.x * BN + b_lc;

    loadB_to_buffer<BK, BN, PAD>(B, N, K, b_smem, 0, gb_r, gb_c, b_lk, b_lc,
                            aligned);

    // 所有线程必须先完成首个 tile 的加载，才能开始读取 shared memory。
    __syncthreads();

    const int num_tiles = (K + BK - 1) / BK; // 分块数量，K 是运行时参数

    // ---------- 遍历所有 K 方向的分块 ----------
    for (int tile_k = 0; tile_k < num_tiles; ++tile_k) {
        // tile_k 是分块编号
        int cur = tile_k & 1;
        int next = cur ^ 1;

        // ---------- 计算当前 buffer ：每线程 16 个 FMA / 8 次 smem 读
        // ---------- 外积累加法：列×行 变成一个部分和矩阵，然后逐次累加
        // a_frag:(TM 行, 1 列float), b_vec: (1 行， 4 列float)
#pragma unroll
        for (int k = 0; k < BK; ++k) { // 分块中 BK 层的和
            float a_frag[TM];          // 4 次读，跨 tx 广播
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                a_frag[r] = a_smem[cur][ty * TM + r][k];
            }

            const float4 b_vec
                = *reinterpret_cast<const float4 *>(&b_smem[cur][k][tx * TN]);
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                sum[r][0] = fmaf(a_frag[r], b_vec.x, sum[r][0]);
                sum[r][1] = fmaf(a_frag[r], b_vec.y, sum[r][1]);
                sum[r][2] = fmaf(a_frag[r], b_vec.z, sum[r][2]);
                sum[r][3] = fmaf(a_frag[r], b_vec.w, sum[r][3]);
            }
        }

        // ---------- 加载下一个 buffer ----------
        if (tile_k + 1 < num_tiles) {
            // 下一个 tile 的全局 K 偏移必须随 tile 改变。
            const int next_tile_k = (tile_k + 1) * BK;
            loadA_to_buffer<BM, BK, PAD>(A, M, K, a_smem, next, ga_r,
                                    next_tile_k + a_lk, a_lr, a_lk, aligned);
            loadB_to_buffer<BK, BN, PAD>(B, N, K, b_smem, next, next_tile_k + b_lk,
                                    gb_c, b_lk, b_lc, aligned);
        }
        __syncthreads();

        cur ^= 1;
        next ^= 1;
    }

// ---------- K 层循环计算完成，写回 ----------
#pragma unroll
    for (int r = 0; r < TM; ++r) {
        const int row = row0 + r;
        if (row >= M) {
            break;
        }
        if (aligned && col0 + TN <= N) {
            *reinterpret_cast<float4 *>(C + (std::size_t)row * N + col0)
                = make_float4(sum[r][0], sum[r][1], sum[r][2], sum[r][3]);
        } else {
#pragma unroll
            for (int c = 0; c < TN; ++c) {
                if (col0 + c < N) {
                    C[(std::size_t)row * N + col0 + c] = sum[r][c];
                }
            }
        }
    }
}
} // namespace

void launch_sgemm_double_buffers(float *output, const float *A, const float *B,
                                 int M, int N, int K, cudaStream_t stream) {
    constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
    // BM, BN: block on M, N dimension processing numbers
    // TM, TN: thread on M, N dimension processing numbers

    const dim3 block(BN / TN, BM / TM); // (16, 16) = 256 线程
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_double_buffers<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(output, A, B, M, N, K);
}

} // namespace cuda_learning
