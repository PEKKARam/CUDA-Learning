#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_learning/kernels.h"

namespace cuda_learning {

namespace {
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_float4(float *__restrict__ C, const float *__restrict__ A,
                             const float *__restrict__ B, int M, int N, int K) {
    constexpr int PAD = 4; // 消 bank conflict + 保持 16B 对齐
    __shared__ float a_s[BM][BK + PAD];
    __shared__ float b_s[BK][BN + PAD];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // block inner tid
    const int tx = threadIdx.x;                             // 0..15，负责 TN 列
    const int ty = threadIdx.y;                             // 0..15，负责 TM 行

    // current thread processing item's starting index(float4* [0])
    const int row0 = blockIdx.y * BM + ty * TM;
    const int col0 = blockIdx.x * BN + tx * TN;

    float sum[TM][TN];
#pragma unroll
    for (int r = 0; r < TM; ++r) {
#pragma unroll
        for (int c = 0; c < TN; ++c) { sum[r][c] = 0.0f; }
    }

    // 每线程负责搬运 8 个 float（两个 float4）
    const int a_lr = tid / (BK / 8);       // A tile 的行 0..BM-1
    const int a_lk = (tid % (BK / 8)) * 8; // A tile 的列 0,8,16,24
    const int b_lk = tid / (BN / 8);       // B tile 的行 0..BK-1
    const int b_lc = (tid % (BN / 8)) * 8; // B tile 的列 0..BN-8

    const bool aligned = (K % 4 == 0) && (N % 4 == 0); // 循环不变量，只判一次

    for (int tile_k = 0; tile_k < K; tile_k += BK) {
        // ---------- 加载 A 子块 [BM, BK] ----------
        const int ga_r = blockIdx.y * BM + a_lr;
        const int ga_c = tile_k + a_lk;
        if (aligned && ga_r < M && ga_c + 7 < K) { // 快路径：纯 float4
            const float4 v0 = *reinterpret_cast<const float4 *>(
                A + (std::size_t)ga_r * K + ga_c);
            const float4 v1 = *reinterpret_cast<const float4 *>(
                A + (std::size_t)ga_r * K + ga_c + 4);
            *reinterpret_cast<float4 *>(&a_s[a_lr][a_lk]) = v0;
            *reinterpret_cast<float4 *>(&a_s[a_lr][a_lk + 4]) = v1;
        } else { // 慢路径：只有边缘 tile 走
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                a_s[a_lr][a_lk + v] = (ga_r < M && ga_c + v < K)
                                        ? A[(std::size_t)ga_r * K + ga_c + v]
                                        : 0.0f;
            }
        }
        // ---------- 加载 B 子块 [BK, BN] ----------
        const int gb_r = tile_k + b_lk;
        const int gb_c = blockIdx.x * BN + b_lc;
        if (aligned && gb_r < K && gb_c + 7 < N) {
            const float4 v0 = *reinterpret_cast<const float4 *>(
                B + (std::size_t)gb_r * N + gb_c);
            const float4 v1 = *reinterpret_cast<const float4 *>(
                B + (std::size_t)gb_r * N + gb_c + 4);
            *reinterpret_cast<float4 *>(&b_s[b_lk][b_lc]) = v0;
            *reinterpret_cast<float4 *>(&b_s[b_lk][b_lc + 4]) = v1;
        } else {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                b_s[b_lk][b_lc + v] = (gb_r < K && gb_c + v < N)
                                        ? B[(std::size_t)gb_r * N + gb_c + v]
                                        : 0.0f;
            }
        }
        __syncthreads();

// ---------- 计算：每线程 16 个 FMA / 8 次 smem 读 ----------
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_frag[TM]; // 4 次读，跨 tx 广播
#pragma unroll
            for (int r = 0; r < TM; ++r) { a_frag[r] = a_s[ty * TM + r][k]; }
            const float4 bv
                = *reinterpret_cast<const float4 *>(&b_s[k][tx * TN]);
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                sum[r][0] = fmaf(a_frag[r], bv.x, sum[r][0]);
                sum[r][1] = fmaf(a_frag[r], bv.y, sum[r][1]);
                sum[r][2] = fmaf(a_frag[r], bv.z, sum[r][2]);
                sum[r][3] = fmaf(a_frag[r], bv.w, sum[r][3]);
            }
        }
        __syncthreads();
    }

// ---------- 写回 ----------
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

void launch_sgemm_float4(float *output, const float *A, const float *B, int M,
                         int N, int K, cudaStream_t stream) {
    constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
    // BM, BN: block on M, N dimension processing numbers
    // TM, TN: thread on M, N dimension processing numbers

    const dim3 block(BN / TN, BM / TM); // (16, 16) = 256 线程
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(output, A, B, M, N, K);
}

} // namespace cuda_learning
