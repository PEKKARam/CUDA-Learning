#include <array>
#include <iostream>
#include <string>

#include "cuda_learning/kernels.h"
#include "cuda_learning/standalone_utils.h"

namespace {

constexpr std::array implementations = {
    cuda_learning::ReduceImplementation::NoBankConflict,
    cuda_learning::ReduceImplementation::AddDuringLoad,
    cuda_learning::ReduceImplementation::AddDuringLoadV2,
    cuda_learning::ReduceImplementation::UnrollLastWarp,
    cuda_learning::ReduceImplementation::CompletelyUnroll,
    cuda_learning::ReduceImplementation::MultiAdd,
    cuda_learning::ReduceImplementation::Shuffle,
};

}  // namespace

int main(int argc, char** argv) {
    const int size = cuda_learning::int_argument(argc, argv, "--size", 1 << 20);
    const int warmup = cuda_learning::int_argument(argc, argv, "--warmup", 10);
    const int iterations =
        cuda_learning::int_argument(argc, argv, "--iters", 100);
    const std::string selected =
        cuda_learning::string_argument(argc, argv, "--impl", "all");
    const bool profile = cuda_learning::has_flag(argc, argv, "--profile");

    if (size <= 0 || warmup < 0 || iterations <= 0) {
        std::cerr << "size/iters must be positive and warmup non-negative"
                  << std::endl;
        return 2;
    }

    if (profile && selected == "all") {
        std::cerr << "--profile requires one --impl so the capture has one "
                     "implementation"
                  << std::endl;
        return 2;
    }

    const auto input = cuda_learning::random_floats(size);
    cuda_learning::DeviceBuffer<float> device_input(size);
    cuda_learning::DeviceBuffer<float> device_output(1);
    device_input.copy_from(input);

    bool matched = false;
    for (const auto implementation : implementations) {
        const std::string name =
            cuda_learning::implementation_name(implementation);
        if (selected != "all" && selected != name) {
            continue;
        }
        matched = true;
        device_output.zero();
        const float milliseconds = cuda_learning::benchmark_ms(
            "reduce_" + name, warmup, iterations, profile, [&] {
                cuda_learning::launch_reduce(
                    implementation, device_input.data(), device_output.data(),
                    size, nullptr);
            });
        const double bandwidth_gbs =
            (static_cast<double>(size) * sizeof(float)) /
            (milliseconds * 1.0e6);
        std::cout << name << ": " << milliseconds * 1000.0F << " us, "
                  << bandwidth_gbs << " GB/s" << std::endl;
    }

    if (!matched) {
        std::cerr << "Unknown --impl: " << selected << std::endl;
        return 2;
    }
    return 0;
}
