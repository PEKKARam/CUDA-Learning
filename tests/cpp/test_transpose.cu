#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuda_learning/kernels.h"
#include "cuda_learning/standalone_utils.h"

int main() {
    // Keep this correctness test intentionally small while exercising partial
    // 16x16 blocks in both dimensions.
    constexpr std::size_t rows = 17;
    constexpr std::size_t columns = 29;

    const auto input = cuda_learning::random_floats(rows * columns);
    std::vector<float> expected(columns * rows);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            expected[column * rows + row] = input[row * columns + column];
        }
    }

    cuda_learning::DeviceBuffer<float> device_input(input.size());
    cuda_learning::DeviceBuffer<float> device_output(expected.size());
    device_input.copy_from(input);

    cuda_learning::launch_transpose_naive(device_input.data(), device_output.data(), rows,
                                          columns, nullptr);
    CUDA_CHECK(cudaGetLastError());
    const auto actual = device_output.copy_to_host();

    if (!cuda_learning::check_vectors(actual, expected)) {
        std::cerr << "transpose failed for " << rows << "x" << columns << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "transpose: all cases passed" << std::endl;
    return EXIT_SUCCESS;
}
