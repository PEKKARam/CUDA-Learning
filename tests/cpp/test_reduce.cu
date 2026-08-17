#include <array>
#include <cstdlib>
#include <iostream>

#include "cuda_learning/kernels.h"
#include "cuda_learning/standalone_utils.h"

int main() {
    constexpr std::array implementations = {
        cuda_learning::ReduceImplementation::NoBankConflict,
        cuda_learning::ReduceImplementation::AddDuringLoad,
        cuda_learning::ReduceImplementation::AddDuringLoadV2,
        cuda_learning::ReduceImplementation::UnrollLastWarp,
        cuda_learning::ReduceImplementation::CompletelyUnroll,
        cuda_learning::ReduceImplementation::MultiAdd,
        cuda_learning::ReduceImplementation::Shuffle,
    };

    for (const auto implementation : implementations) {
        for (const std::size_t size : {1U, 255U, 256U, 1003U, 65536U}) {
            const auto input = cuda_learning::random_floats(size);
            double expected = 0.0;
            for (const float value : input) {
                expected += value;
            }

            cuda_learning::DeviceBuffer<float> device_input(size);
            cuda_learning::DeviceBuffer<float> device_output(1);
            device_input.copy_from(input);
            device_output.zero();
            cuda_learning::launch_reduce(implementation, device_input.data(),
                                         device_output.data(), size, nullptr);
            CUDA_CHECK(cudaGetLastError());
            const float actual = device_output.copy_to_host()[0];
            if (!cuda_learning::close(actual, static_cast<float>(expected),
                                      1e-2F, 1e-3F)) {
                std::cerr << "reduce "
                          << cuda_learning::implementation_name(implementation)
                          << " failed for size " << size
                          << ": actual=" << actual << ", expected=" << expected
                          << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    std::cout << "reduce: all cases passed" << std::endl;
    return EXIT_SUCCESS;
}
