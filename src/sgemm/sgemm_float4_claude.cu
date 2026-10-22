#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_learning/kernels.h"

namespace cuda_learning {

namespace {

// 带边界保护的 float4 读取：越界补 0，不满足对齐则退化为标量读取。
__device__ __forceinline__ float4 ldg_f4_guarded(const float *__restrict__ base,
                                                 int row, int col, int rows,
                                                 int cols, int ld,
                                                 bool aligned) {
    if (aligned && row < rows && col + 3 < cols) {
        return *reinterpret_cast<const float4 *>(
            base + static_cast<std::size_t>(row) * ld + col);
    }
    float4 r;
    const std::size_t off = static_cast<std::size_t>(row) * ld + col;
    const bool rv = row < rows;
    r.x = (rv && col + 0 < cols) ? base[off + 0] : 0.0f;
    r.y = (rv && col + 1 < cols) ? base[off + 1] : 0.0f;
    r.z = (rv && col + 2 < cols) ? base[off + 2] : 0.0f;
    r.w = (rv && col + 3 < cols) ? base[off + 3] : 0.0f;
    return r;
}

__device__ __forceinline__ float pick(const float4 &v, int i) {
    return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
}

// A 转置写入 shared：dst[k][row]，行跨度 LD。
// rot 为线程私有的旋转量，使同一条 STS 指令中各线程落在不同 bank。
template <int LD>
__device__ __forceinline__ void store_a_transposed(float *dst, const float4 &v,
                                                   int col, int row, int rot) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int jj = (j + rot) & 3;
        dst[(col + jj) * LD + row] = pick(v, jj);
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_float4_v2_claude(float *__restrict__ C, const float *__restrict__ A,
                           const float *__restrict__ B, int M, int N, int K) {
    constexpr int PAD = 4;
    constexpr int A_LD = BM + PAD;
    constexpr int B_LD = BN + PAD;
    constexpr int A_HALF_TILE = BM * BK / 2;
    constexpr int B_HALF_TILE = BK * BN / 2;
    constexpr int THREADS = (BM / TM) * (BN / TN);

    static_assert(BM * BK == THREADS * 8 && BK * BN == THREADS * 8,
                  "Each thread must load two float4 vectors per tile.");
    static_assert(BK % 4 == 0 && BN % 4 == 0, "float4 must stay in a row.");
    static_assert(A_LD % 4 == 0 && B_LD % 4 == 0, "Keep float4 alignment.");
    static_assert(TM == 4 && TN == 4, "This implementation assumes TM=TN=4.");

    // 双缓冲：[2][BK][..]
    __shared__ __align__(16) float a_smem[2][BK][A_LD];
    __shared__ __align__(16) float b_smem[2][BK][B_LD];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const int row0 = blockIdx.y * BM + ty * TM;
    const int col0 = blockIdx.x * BN + tx * TN;

    float sum[TM][TN];
#pragma unroll
    for (int r = 0; r < TM; ++r)
#pragma unroll
        for (int c = 0; c < TN; ++c) {
            sum[r][c] = 0.0f;
        }

    // ---- 加载映射（与原版相同）----
    const int a_i0 = tid * 4, a_i1 = a_i0 + A_HALF_TILE;
    const int a_row0 = a_i0 / BK, a_col0 = a_i0 % BK;
    const int a_row1 = a_i1 / BK, a_col1 = a_i1 % BK;

    const int b_i0 = tid * 4, b_i1 = b_i0 + B_HALF_TILE;
    const int b_row0 = b_i0 / BN, b_col0 = b_i0 % BN;
    const int b_row1 = b_i1 / BN, b_col1 = b_i1 % BN;

    // 转置写入的 bank 旋转量：8 个线程负责同一行的 BK=32 列，
    // 用 (tid>>1)&3 让它们的写入列号 mod 8 互不相同。
    const int rot = (tid >> 1) & 3;

    const bool a_aligned = (K % 4 == 0);
    const bool b_aligned = (N % 4 == 0);

    const int a_grow0 = blockIdx.y * BM + a_row0;
    const int a_grow1 = blockIdx.y * BM + a_row1;
    const int b_gcol0 = blockIdx.x * BN + b_col0;
    const int b_gcol1 = blockIdx.x * BN + b_col1;

    // 寄存器暂存下一个 tile 的数据
    float4 ra0, ra1, rb0, rb1;

    auto load_regs = [&](int tile_k) {
        ra0 = ldg_f4_guarded(A, a_grow0, tile_k + a_col0, M, K, K, a_aligned);
        ra1 = ldg_f4_guarded(A, a_grow1, tile_k + a_col1, M, K, K, a_aligned);
        rb0 = ldg_f4_guarded(B, tile_k + b_row0, b_gcol0, K, N, N, b_aligned);
        rb1 = ldg_f4_guarded(B, tile_k + b_row1, b_gcol1, K, N, N, b_aligned);
    };

    auto store_smem = [&](int buf) {
        float *as = &a_smem[buf][0][0];
        store_a_transposed<A_LD>(as, ra0, a_col0, a_row0, rot);
        store_a_transposed<A_LD>(as, ra1, a_col1, a_row1, rot);
        *reinterpret_cast<float4 *>(&b_smem[buf][b_row0][b_col0]) = rb0;
        *reinterpret_cast<float4 *>(&b_smem[buf][b_row1][b_col1]) = rb1;
    };

    // ---- prologue：第 0 个 tile ----
    load_regs(0);
    store_smem(0);
    __syncthreads();

    int cur = 0;
    for (int tile_k = 0; tile_k < K; tile_k += BK) {
        const bool has_next = (tile_k + BK) < K;

        // 先发出下一个 tile 的 global load，此时不使用其结果
        if (has_next) {
            load_regs(tile_k + BK);
        }

        // 计算当前 tile：与上面的 global load 在途时间重叠
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float4 a_vec
                = *reinterpret_cast<const float4 *>(&a_smem[cur][k][ty * TM]);
            const float4 b_vec
                = *reinterpret_cast<const float4 *>(&b_smem[cur][k][tx * TN]);
            const float a_frag[TM] = {a_vec.x, a_vec.y, a_vec.z, a_vec.w};
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                sum[r][0] = fmaf(a_frag[r], b_vec.x, sum[r][0]);
                sum[r][1] = fmaf(a_frag[r], b_vec.y, sum[r][1]);
                sum[r][2] = fmaf(a_frag[r], b_vec.z, sum[r][2]);
                sum[r][3] = fmaf(a_frag[r], b_vec.w, sum[r][3]);
            }
        }

        // 把预取的数据写入另一块 buffer（它在上一轮已被所有线程读完）
        if (has_next) {
            store_smem(cur ^ 1);
        }
        // 每轮只需一次同步：既保证 cur^1 写完可见，
        // 也保证下一轮覆盖 cur 之前所有线程已读完
        __syncthreads();
        cur ^= 1;
    }

    // ---- Store C（与原版相同）----
#pragma unroll
    for (int r = 0; r < TM; ++r) {
        const int row = row0 + r;
        if (row < M) {
            float *c_ptr = C + static_cast<std::size_t>(row) * N + col0;
            if ((N % 4 == 0) && (col0 + 3 < N)) {
                *reinterpret_cast<float4 *>(c_ptr)
                    = make_float4(sum[r][0], sum[r][1], sum[r][2], sum[r][3]);
            } else {
#pragma unroll
                for (int c = 0; c < TN; ++c) {
                    if (col0 + c < N) {
                        c_ptr[c] = sum[r][c];
                    }
                }
            }
        }
    }
}

} // namespace

void launch_sgemm_float4_v2_claude(float *output, const float *A,
                                   const float *B, int M, int N, int K,
                                   cudaStream_t stream) {
    constexpr int BM = 64, BN = 64, BK = 32, TM = 4, TN = 4;
    const dim3 block(BN / TN, BM / TM);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_float4_v2_claude<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(output, A, B, M, N, K);
}

} // namespace cuda_learning