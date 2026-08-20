#pragma once

#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cuda_learning/cuda_check.h"

namespace cuda_learning {

// device buffer
template <typename T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ != 0) {
            CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *data() { return data_; }
    const T *data() const { return data_; }
    std::size_t size() const { return count_; }

    // host to device
    void copy_from(const std::vector<T> &host) {
        CUDA_CHECK(cudaMemcpy(data_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    // device to host
    std::vector<T> copy_to_host() const {
        std::vector<T> host(count_);
        CUDA_CHECK(cudaMemcpy(host.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }

    void zero(cudaStream_t stream = nullptr) {
        if (count_ != 0) {
            CUDA_CHECK(cudaMemsetAsync(data_, 0, count_ * sizeof(T), stream));
        }
    }

private:
    T *data_ = nullptr;
    std::size_t count_ = 0;
};

inline std::vector<float> random_floats(std::size_t count, unsigned int seed = 1) {
    // 随机数生成器
    std::mt19937 generator(seed);
    // 随机数生成范围器，限制随机数的生成范围
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);

    std::vector<float> values(count);
    for (float &value : values) { value = distribution(generator); }
    return values;
}

// 误差对比函数, atol: 绝对误差，rtol: 相对误差
inline bool close(float actual, float expected, float atol = 1e-4F, float rtol = 1e-3F) {
    return std::abs(actual - expected) <= atol + rtol * std::abs(expected);
}

inline bool check_vectors(const std::vector<float> &actual, const std::vector<float> &expected,
                          float atol = 1e-4F, float rtol = 1e-3F) {
    if (actual.size() != expected.size()) {
        return false;
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!close(actual[index], expected[index], atol, rtol)) {
            std::cerr << "Mismatch at " << index << ": actual=" << actual[index]
                      << ", expected=" << expected[index] << std::endl;
            return false;
        }
    }
    return true;
}

template <typename Launch>
float benchmark_ms(const std::string &label, int warmup, int iterations, bool profile_range,
                   Launch &&launch) {
    for (int iteration = 0; iteration < warmup; ++iteration) { launch(); }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    if (profile_range) {
        CUDA_CHECK(cudaProfilerStart());
    }
    nvtxRangePushA(label.c_str());
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) { launch(); }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    nvtxRangePop();
    if (profile_range) {
        CUDA_CHECK(cudaProfilerStop());
    }
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / static_cast<float>(iterations);
}

// 解析命令行参数
inline int int_argument(int argc, char **argv, const std::string &name, int default_value) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) {
            return std::stoi(argv[index + 1]);
        }
    }
    return default_value;
}

inline std::string string_argument(int argc, char **argv, const std::string &name,
                                   std::string default_value) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) {
            return argv[index + 1];
        }
    }
    return default_value;
}

inline bool has_flag(int argc, char **argv, const std::string &flag) {
    for (int index = 1; index < argc; ++index) {
        if (argv[index] == flag) {
            return true;
        }
    }
    return false;
}

} // namespace cuda_learning
