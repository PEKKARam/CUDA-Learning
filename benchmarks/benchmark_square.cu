#include <iostream>

#include "cuda_learning/kernels.h"
#include "cuda_learning/standalone_utils.h"

int main(int argc, char** argv) {
    const int size = cuda_learning::int_argument(argc, argv, "--size", 1 << 20);
    const int warmup = cuda_learning::int_argument(argc, argv, "--warmup", 10);
    const int iterations =
        cuda_learning::int_argument(argc, argv, "--iters", 100);
    const bool profile = cuda_learning::has_flag(argc, argv, "--profile");

    if (size <= 0 || warmup < 0 || iterations <= 0) {
        std::cerr << "size/iters must be positive and warmup non-negative"
                  << std::endl;
        return 2;
    }

    const auto input = cuda_learning::random_floats(size);
    cuda_learning::DeviceBuffer<float> device_input(size);
    cuda_learning::DeviceBuffer<float> device_output(size);
    device_input.copy_from(input);

    const float milliseconds =
        cuda_learning::benchmark_ms("square", warmup, iterations, profile, [&] {
            cuda_learning::launch_square(device_output.data(),
                                         device_input.data(), size, nullptr);
        });
    std::cout << "square size=" << size << ": " << milliseconds * 1000.0F
              << " us" << std::endl;
    return 0;
}
