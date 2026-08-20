#include <cstdlib>
#include <iostream>
#include <tuple>
#include <vector>

#include "cuda_learning/kernels.h"
#include "cuda_learning/standalone_utils.h"

std::vector<float> reference_sgemm(const std::vector<float> &a, const std::vector<float> &b, int m,
                                   int n, int k) {
    std::vector<float> output(static_cast<std::size_t>(m) * n, 0.0F);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            for (int inner = 0; inner < k; ++inner) {
                output[row * n + col] += a[row * k + inner] * b[inner * n + col];
            }
        }
    }
    return output;
}

int main() {
    const std::vector<std::tuple<int, int, int>> shapes = {
        {1, 1, 1},    {2, 20, 2},   {3, 11, 7},   {31, 29, 17}, {64, 64, 64},
        {16, 16, 16}, {17, 17, 17}, {32, 32, 32}, {33, 33, 33}, {65, 65, 65},
    };

    std::cout << "SGEMM test plan: " << shapes.size() << " shapes, "
              << cuda_learning::sgemm_implementations().size()
              << " registered implementations" << std::endl;

    for (std::size_t shape_index = 0; shape_index < shapes.size(); ++shape_index) {
        const auto &[m, n, k] = shapes[shape_index];
        std::cout << "\n[shape " << (shape_index + 1) << "/" << shapes.size()
                  << "] M=" << m << ", N=" << n << ", K=" << k << std::endl;

        const auto a = cuda_learning::random_floats(static_cast<std::size_t>(m) * k, 1);
        const auto b = cuda_learning::random_floats(static_cast<std::size_t>(k) * n, 2);
        const auto expected = reference_sgemm(a, b, m, n, k);

        cuda_learning::DeviceBuffer<float> device_a(a.size());
        cuda_learning::DeviceBuffer<float> device_b(b.size());
        cuda_learning::DeviceBuffer<float> device_output(expected.size());
        device_a.copy_from(a);
        device_b.copy_from(b);

        std::size_t implementation_index = 0;
        for (const auto &implementation : cuda_learning::sgemm_implementations()) {
            ++implementation_index;
            std::cout << "  [impl " << implementation_index << "/"
                      << cuda_learning::sgemm_implementations().size() << "] "
                      << implementation.name << " ... " << std::flush;

            implementation.launch(device_output.data(), device_a.data(), device_b.data(), m, n, k,
                                  nullptr);
            CUDA_CHECK(cudaGetLastError());
            const auto actual = device_output.copy_to_host();
            if (!cuda_learning::check_vectors(actual, expected, 1e-3F, 1e-3F)) {
                std::cerr << "FAIL\nSGEMM " << implementation.name
                          << " failed for M=" << m << ", N=" << n << ", K=" << k
                          << std::endl;
                return EXIT_FAILURE;
            }
            std::cout << "PASS" << std::endl;
        }
    }

    std::cout << "sgemm: all cases passed" << std::endl;
    return EXIT_SUCCESS;
}
