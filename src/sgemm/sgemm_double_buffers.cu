#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_learning/kernels.h"

namespace cuda_learning {

namespace {

template <int BM, int BK, int PAD>
__device__ __forceinline__ void
loadA_to_buffer(const float *__restrict__ A, int M, int K,
                float (*a_smem)[BM][BK + PAD], int load_idx, int tid,
                int tile_k, const bool aligned) {
    // 与 float4_v2 一致：每轮相邻线程加载相邻的 float4。
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int a_i = tid * 4 + half * (BM * BK / 2);
        const int a_lr = a_i / BK;
        const int a_lk = a_i % BK;
        const int ga_r = blockIdx.y * BM + a_lr;
        const int ga_c = tile_k + a_lk;

        if (aligned && ga_r < M && ga_c + 3 < K) {
            const float4 v = *reinterpret_cast<const float4 *>(
                A + static_cast<std::size_t>(ga_r) * K + ga_c);
            *reinterpret_cast<float4 *>(&a_smem[load_idx][a_lr][a_lk]) = v;
        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                a_smem[load_idx][a_lr][a_lk + v]
                    = (ga_r < M && ga_c + v < K)
                        ? A[static_cast<std::size_t>(ga_r) * K + ga_c + v]
                        : 0.0f;
            }
        }
    }
}

template <int BK, int BN, int PAD>
__device__ __forceinline__ void
loadB_to_buffer(const float *__restrict__ B, int N, int K,
                float (*b_smem)[BK][BN + PAD], int load_idx, int tid,
                int tile_k, const bool aligned) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int b_i = tid * 4 + half * (BK * BN / 2);
        const int b_lk = b_i / BN;
        const int b_lc = b_i % BN;
        const int gb_r = tile_k + b_lk;
        const int gb_c = blockIdx.x * BN + b_lc;

        if (aligned && gb_r < K && gb_c + 3 < N) {
            const float4 v = *reinterpret_cast<const float4 *>(
                B + static_cast<std::size_t>(gb_r) * N + gb_c);
            *reinterpret_cast<float4 *>(&b_smem[load_idx][b_lk][b_lc]) = v;
        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                b_smem[load_idx][b_lk][b_lc + v]
                    = (gb_r < K && gb_c + v < N)
                        ? B[static_cast<std::size_t>(gb_r) * N + gb_c + v]
                        : 0.0f;
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void
sgemm_double_buffers(float *__restrict__ C, const float *__restrict__ A,
                     const float *__restrict__ B, int M, int N, int K) {
    constexpr int PAD = 4;
    constexpr int THREADS = (BM / TM) * (BN / TN);
    static_assert(BM * BK == THREADS * 8 && BK * BN == THREADS * 8,
                  "Each thread must load two float4 vectors per tile.");

    __shared__ float a_smem[2][BM][BK + PAD];
    __shared__ float b_smem[2][BK][BN + PAD];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // block inner tid
    const int tx = threadIdx.x;                             // 0..15，负责 TN 列
    const int ty = threadIdx.y;                             // 0..15，负责 TM 行

    // start index of processing area
    const int row0 = blockIdx.y * BM + ty * TM;
    const int col0 = blockIdx.x * BN + tx * TN;

    float sum[TM][TN] = {0.0f}; // accelarators of each thread

    const bool a_aligned = (K % 4 == 0);
    const bool b_aligned = (N % 4 == 0);

    // ---------- 预加载 A 子块 [BM, BK] ----------
    loadA_to_buffer<BM, BK, PAD>(A, M, K, a_smem, 0, tid, 0, a_aligned);

    // ---------- 预加载 B 子块 [BK, BN] ----------
    loadB_to_buffer<BK, BN, PAD>(B, N, K, b_smem, 0, tid, 0, b_aligned);

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
            loadA_to_buffer<BM, BK, PAD>(A, M, K, a_smem, next, tid,
                                        next_tile_k, a_aligned);
            loadB_to_buffer<BK, BN, PAD>(B, N, K, b_smem, next, tid,
                                        next_tile_k, b_aligned);
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
        if (b_aligned && col0 + TN <= N) {
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
