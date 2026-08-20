#include <cublas_v2.h>

#include <cstdlib>
#include <iostream>
#include <string>

#include "cuda_learning/kernels.h"
#include "cuda_learning/standalone_utils.h"

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        const cublasStatus_t cublas_status = (call);                       \
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {                      \
            std::cerr << "cuBLAS error at " << __FILE__ << ':' << __LINE__ \
                      << ": " << cublasGetStatusString(cublas_status)      \
                      << std::endl;                                        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (false)

namespace {

void launch_cublas_sgemm(cublasHandle_t handle, float* output, const float* a,
                         const float* b, int m, int n, int k) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;

    // cuBLAS is column-major. Computing C^T = B^T * A^T gives row-major C.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                             b, n, a, k, &beta, output, n));
}

}  // namespace

int main(int argc, char** argv) {
    const int m = cuda_learning::int_argument(argc, argv, "--m", 1024);
    const int n = cuda_learning::int_argument(argc, argv, "--n", 1024);
    const int k = cuda_learning::int_argument(argc, argv, "--k", 1024);
    const int warmup = cuda_learning::int_argument(argc, argv, "--warmup", 10);
    const int iterations =
        cuda_learning::int_argument(argc, argv, "--iters", 100);
    const std::string implementation =
        cuda_learning::string_argument(argc, argv, "--impl", "all");
    const bool profile = cuda_learning::has_flag(argc, argv, "--profile");

    if (m <= 0 || n <= 0 || k <= 0 || warmup < 0 || iterations <= 0) {
        std::cerr << "M/N/K/iters must be positive and warmup non-negative"
                  << std::endl;
        return 2;
    }
    if (implementation != "all" && implementation != "cublas" &&
        cuda_learning::find_sgemm_implementation(implementation) == nullptr) {
        std::cerr << "Unknown --impl: " << implementation << std::endl;
        return 2;
    }
    if (profile && implementation == "all") {
        std::cerr << "--profile requires one concrete --impl" << std::endl;
        return 2;
    }

    const auto a =
        cuda_learning::random_floats(static_cast<std::size_t>(m) * k, 1);
    const auto b =
        cuda_learning::random_floats(static_cast<std::size_t>(k) * n, 2);
    cuda_learning::DeviceBuffer<float> device_a(a.size());
    cuda_learning::DeviceBuffer<float> device_b(b.size());
    cuda_learning::DeviceBuffer<float> device_naive(
        static_cast<std::size_t>(m) * n);
    cuda_learning::DeviceBuffer<float> device_cublas(
        static_cast<std::size_t>(m) * n);
    device_a.copy_from(a);
    device_b.copy_from(b);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    cuda_learning::launch_sgemm("naive", device_naive.data(), device_a.data(),
                                device_b.data(), m, n, k, nullptr);
    launch_cublas_sgemm(handle, device_cublas.data(), device_a.data(),
                        device_b.data(), m, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto reference = device_cublas.copy_to_host();
    if (!cuda_learning::check_vectors(device_naive.copy_to_host(), reference,
                                      1e-2F, 1e-3F)) {
        std::cerr << "naive SGEMM does not match cuBLAS" << std::endl;
        CUBLAS_CHECK(cublasDestroy(handle));
        return EXIT_FAILURE;
    }

    const double operations = 2.0 * static_cast<double>(m) * n * k;
    auto report = [&](const std::string& name, float milliseconds) {
        const double gflops = operations / (milliseconds * 1.0e6);
        std::cout << name << " M=" << m << " N=" << n << " K=" << k << ": "
                  << milliseconds * 1000.0F << " us, " << gflops << " GFLOP/s"
                  << std::endl;
    };

    for (const auto& selected : cuda_learning::sgemm_implementations()) {
        if (implementation != "all" && implementation != selected.name) {
            continue;
        }
        const std::string name(selected.name);

        // Validate this implementation outside the timed region. The
        // benchmark should never report performance for an incorrect kernel.
        selected.launch(device_naive.data(), device_a.data(), device_b.data(),
                        m, n, k, nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!cuda_learning::check_vectors(device_naive.copy_to_host(),
                                          reference, 1e-2F, 1e-3F)) {
            std::cerr << "SGEMM " << name << " failed correctness for M=" << m
                      << ", N=" << n << ", K=" << k << std::endl;
            CUBLAS_CHECK(cublasDestroy(handle));
            return EXIT_FAILURE;
        }

        const float milliseconds = cuda_learning::benchmark_ms(
            "sgemm_" + name, warmup, iterations, profile, [&] {
                selected.launch(device_naive.data(), device_a.data(),
                                device_b.data(), m, n, k, nullptr);
            });
        report(name, milliseconds);
    }

    if (implementation == "all" || implementation == "cublas") {
        CUDA_CHECK(cudaMemset(device_cublas.data(), 0,
                              static_cast<std::size_t>(m) * n * sizeof(float)));
        launch_cublas_sgemm(handle, device_cublas.data(), device_a.data(),
                            device_b.data(), m, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!cuda_learning::check_vectors(device_cublas.copy_to_host(),
                                          reference, 1e-2F, 1e-3F)) {
            std::cerr << "cuBLAS SGEMM failed its reference check for M=" << m
                      << ", N=" << n << ", K=" << k << std::endl;
            CUBLAS_CHECK(cublasDestroy(handle));
            return EXIT_FAILURE;
        }

        const float milliseconds = cuda_learning::benchmark_ms(
            "sgemm_cublas", warmup, iterations, profile, [&] {
                launch_cublas_sgemm(handle, device_cublas.data(),
                                    device_a.data(), device_b.data(), m, n, k);
            });
        report("cublas", milliseconds);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return EXIT_SUCCESS;
}
