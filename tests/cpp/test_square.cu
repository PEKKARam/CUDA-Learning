#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuda_learning/kernels.h"
#include "cuda_learning/standalone_utils.h"

int main() {
    for (const std::size_t size : {1U, 255U, 256U, 1003U, 65536U}) {
        const auto input = cuda_learning::random_floats(size);
        std::vector<float> expected(size);
        for (std::size_t index = 0; index < size; ++index) {
            expected[index] = input[index] * input[index];
        }

        cuda_learning::DeviceBuffer<float> device_input(size);
        cuda_learning::DeviceBuffer<float> device_output(size);
        device_input.copy_from(input);

        cuda_learning::launch_square(device_output.data(), device_input.data(),
                                     size, nullptr);
        CUDA_CHECK(cudaGetLastError());
        const auto actual = device_output.copy_to_host();
        if (!cuda_learning::check_vectors(actual, expected)) {
            std::cerr << "square failed for size " << size << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "square: all cases passed" << std::endl;
    return EXIT_SUCCESS;
}
