#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_learning/kernels.h"

namespace cuda_learning {

namespace {
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_float4(float *__restrict__ C, const float *__restrict__ A,
                             const float *__restrict__ B, int M, int N, int K) {
    // constexpr int PAD = 4; // 消 bank conflict + 保持 16B 对齐
    // shared memory 行跨度不能是32的倍数，否则在计算bank时会导致bank conflict
    // 在float4 访存条件下不成立，不使用PAD
    __shared__ float a_smem[BM][BK];
    __shared__ float b_smem[BK][BN];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // block inner tid
    const int tx = threadIdx.x;                             // 0..15，负责 TN 列
    const int ty = threadIdx.y;                             // 0..15，负责 TM 行

    // 当前线程处理的区域 在output对应的起始地址
    const int row0 = blockIdx.y * BM + ty * TM;
    const int col0 = blockIdx.x * BN + tx * TN;

    float sum[TM][TN]; // accelarators of each thread
#pragma unroll
    for (int r = 0; r < TM; ++r) {
#pragma unroll
        for (int c = 0; c < TN; ++c) {
            sum[r][c] = 0.0f;
        }
    }

    // 每线程负责搬运 4 个 float，跨4个bank（1个 float4) load_row/load_col
    // 每个线程应该搬运4个float，八个float使用了两条指令也就会导致两个load
    // A矩阵 BK / 4 = 8: 一行8个线程，每个线程搬运4个float
    const int a_lr = tid / (BK / 4);       // A tile 的行 0..BM-1
    const int a_lk = (tid % (BK / 4)) * 4; // A tile 的列 0,4,8,12

    // B矩阵 BN / 4 = 16： 一行16个线程，每个线程搬运4个float
    const int b_lk = tid / (BN / 4);       // B tile 的行 0..BK-1
    const int b_lc = (tid % (BN / 4)) * 4; // B tile 的列 0..BN-8

    const bool aligned = (K % 4 == 0) && (N % 4 == 0); // 循环不变量，只判一次

    // ---------- 遍历所有 K 方向的分块 ----------
    for (int tile_k = 0; tile_k < K; tile_k += BK) {
        // ---------- 加载 A 子块 [BM, BK] ----------
        const int ga_r = blockIdx.y * BM + a_lr;
        const int ga_c = tile_k + a_lk;
        if (aligned && ga_r < M && ga_c + 3 < K) { // 快路径：纯 float4
            const float4 v = *reinterpret_cast<const float4 *>(
                A + (std::size_t)ga_r * K + ga_c);

            *reinterpret_cast<float4 *>(&a_smem[a_lr][a_lk]) = v;
        } else { // 慢路径：只有边缘 tile 走
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                a_smem[a_lr][a_lk + v] = (ga_r < M && ga_c + v < K)
                                           ? A[(std::size_t)ga_r * K + ga_c + v]
                                           : 0.0f;
            }
        }
        // ---------- 加载 B 子块 [BK, BN] ----------
        const int gb_r = tile_k + b_lk;
        const int gb_c = blockIdx.x * BN + b_lc;
        if (aligned && gb_r < K && gb_c + 3 < N) {
            const float4 v = *reinterpret_cast<const float4 *>(
                B + (std::size_t)gb_r * N + gb_c);

            *reinterpret_cast<float4 *>(&b_smem[b_lk][b_lc]) = v;
        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                b_smem[b_lk][b_lc + v] = (gb_r < K && gb_c + v < N)
                                           ? B[(std::size_t)gb_r * N + gb_c + v]
                                           : 0.0f;
            }
        }
        __syncthreads();

        // ---------- 计算：每线程 16 个 FMA / 8 次 smem 读 ----------
        // 外积累加法：列×行 变成一个部分和矩阵，然后逐次累加
        // a_frag:(TM 行, 1 列float), b_vec: (1 行， 4 列float)
#pragma unroll
        for (int k = 0; k < BK; ++k) { // 分块中 BK 层的和
            float a_frag[TM];          // 4 次读，跨 tx，存放转置
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                a_frag[r] = a_smem[ty * TM + r][k];
            }

            float b_frag[TN];
#pragma unroll
            for (int c = 0; c < TN; ++c) {
                b_frag[c] = b_smem[k][tx * TN + c];
            }

#pragma unroll
            for (int r = 0; r < TM; ++r) {
#pragma unroll
                for (int c = 0; c < TN; ++c) {
                    sum[r][c] = fmaf(a_frag[r], b_frag[c], sum[r][c]);
                }
            }
        }
        __syncthreads();
    }

// ---------- K 层循环计算完成，写回 ----------
#pragma unroll
    for (int r = 0; r < TM; ++r) {
        const int row = row0 + r;
        if (row >= M) {
            break;
        }
#pragma unroll
        for (int c = 0; c < TN; ++c) {
            if (col0 + c < N) {
                C[(std::size_t)row * N + col0 + c] = sum[r][c];
            }
        }
    }
}
} // namespace

void launch_sgemm_float4(float *output, const float *A, const float *B, int M,
                         int N, int K, cudaStream_t stream) {
    constexpr int BM = 32, BN = 32, BK = 32, TM = 2, TN = 2;
    // BM, BN: block on M, N dimension processing numbers
    // TM, TN: thread on M, N dimension processing numbers

    const dim3 block(BN / TN, BM / TM); // (32, 8) = 256 线程
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(output, A, B, M, N, K);
}

} // namespace cuda_learning
