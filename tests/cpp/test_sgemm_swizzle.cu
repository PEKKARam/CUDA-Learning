#include <iostream>
#include <string_view>
#include <tuple>
#include <vector>

#include "cuda_learning/kernels.h"
#include "cuda_learning/standalone_utils.h"

int main() {
    const std::vector<std::tuple<int, int, int>> shapes = {
        // Complete tiles, each individual tail dimension, unaligned strides, K=0.
        {64, 64, 32}, {128, 64, 64}, {65, 64, 32}, {64, 65, 32},
        {64, 64, 33}, {1, 1, 1}, {31, 29, 17}, {65, 68, 36},
        {68, 65, 35}, {127, 132, 65}, {64, 64, 0},
        {512, 512, 512},
        {768, 768, 768},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
    };
    // The unroll8/4/1 entries now all preload AB; preload_ab is their unroll32 control.
    const std::vector<std::string_view> names
        = {"float4_v3_swizzle", "float4_v3_swizzle_unroll8",
           "float4_v3_swizzle_unroll4", "float4_v3_swizzle_unroll1",
           "float4_v3_swizzle_preload_a", "float4_v3_swizzle_preload_b",
           "float4_v3_swizzle_preload_ab",
           "float4_v3_swizzle_preload_ab_frag2",
           "float4_v3_swizzle_preload_ab_frag2_fulltile"};

    for (const auto &[m, n, k] : shapes) {
        const auto a
            = cuda_learning::random_floats(static_cast<std::size_t>(m) * k, 1);
        const auto b
            = cuda_learning::random_floats(static_cast<std::size_t>(k) * n, 2);
        std::vector<float> expected(static_cast<std::size_t>(m) * n, 0.0f);
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                for (int inner = 0; inner < k; ++inner) {
                    expected[row * n + col]
                        += a[row * k + inner] * b[inner * n + col];
                }
            }
        }

        // Allocate nonempty inputs for K=0; the kernel must not read them.
        cuda_learning::DeviceBuffer<float> da(a.empty() ? 1 : a.size());
        cuda_learning::DeviceBuffer<float> db(b.empty() ? 1 : b.size());
        cuda_learning::DeviceBuffer<float> dc(expected.size());
        if (!a.empty()) {
            da.copy_from(a);
        }
        if (!b.empty()) {
            db.copy_from(b);
        }
        for (const auto name : names) {
            // Poison previous output so missing stores cannot pass
            // accidentally.
            CUDA_CHECK(
                cudaMemset(dc.data(), 0xff, expected.size() * sizeof(float)));
            cuda_learning::launch_sgemm(name, dc.data(), da.data(), db.data(),
                                        m, n, k, nullptr);
            CUDA_CHECK(cudaGetLastError());
            if (!cuda_learning::check_vectors(dc.copy_to_host(), expected,
                                              1e-3f, 1e-3f)) {
                std::cerr << name << " failed: " << m << 'x' << n << 'x' << k
                          << '\n';
                return 1;
            }
        }
        std::cout << "PASS " << m << 'x' << n << 'x' << k << '\n';
    }
}
