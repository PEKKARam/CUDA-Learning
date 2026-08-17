#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "cuda_learning/kernels.h"

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) \
    TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x)   \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x); \
    CHECK_FLOAT32(x)

cudaStream_t current_stream(const torch::Tensor& tensor) {
    return c10::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

torch::Tensor square(const torch::Tensor& input) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dim() == 1, "square expects a 1D tensor");

    auto output = torch::empty_like(input);
    cuda_learning::launch_square(output.data_ptr<float>(),
                                 input.data_ptr<float>(), input.numel(),
                                 current_stream(input));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor sgemm_impl(const torch::Tensor& a, const torch::Tensor& b,
                         std::string_view implementation) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "SGEMM expects two 2D tensors");
    TORCH_CHECK(a.device() == b.device(),
                "SGEMM inputs must be on the same CUDA device");
    TORCH_CHECK(a.size(1) == b.size(0), "SGEMM inner dimensions must match");

    const int m = static_cast<int>(a.size(0));
    const int n = static_cast<int>(b.size(1));
    const int k = static_cast<int>(a.size(1));
    auto output = torch::zeros({m, n}, a.options());

    cuda_learning::launch_sgemm(implementation, output.data_ptr<float>(),
                                a.data_ptr<float>(), b.data_ptr<float>(), m, n,
                                k, current_stream(a));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor sgemm_naive(const torch::Tensor& a, const torch::Tensor& b) {
    return sgemm_impl(a, b, "naive");
}

torch::Tensor reduce_impl(const torch::Tensor& input,
                          cuda_learning::ReduceImplementation implementation) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dim() == 1, "reduce expects a 1D tensor");

    auto output = torch::zeros({}, input.options());
    cuda_learning::launch_reduce(implementation, input.data_ptr<float>(),
                                 output.data_ptr<float>(), input.numel(),
                                 current_stream(input));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor reduce_no_bankconflict(const torch::Tensor& input) {
    return reduce_impl(input,
                       cuda_learning::ReduceImplementation::NoBankConflict);
}

torch::Tensor reduce_add_during_load(const torch::Tensor& input) {
    return reduce_impl(input,
                       cuda_learning::ReduceImplementation::AddDuringLoad);
}

torch::Tensor reduce_add_during_load_v2(const torch::Tensor& input) {
    return reduce_impl(input,
                       cuda_learning::ReduceImplementation::AddDuringLoadV2);
}

torch::Tensor reduce_unroll_last_warp(const torch::Tensor& input) {
    return reduce_impl(input,
                       cuda_learning::ReduceImplementation::UnrollLastWarp);
}

torch::Tensor reduce_completely_unroll(const torch::Tensor& input) {
    return reduce_impl(input,
                       cuda_learning::ReduceImplementation::CompletelyUnroll);
}

torch::Tensor reduce_multi_add(const torch::Tensor& input) {
    return reduce_impl(input, cuda_learning::ReduceImplementation::MultiAdd);
}

torch::Tensor reduce_shuffle(const torch::Tensor& input) {
    return reduce_impl(input, cuda_learning::ReduceImplementation::Shuffle);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("square", torch::wrap_pybind_function(square), "square");

    module.def("sgemm_naive", torch::wrap_pybind_function(sgemm_naive),
               "row-major naive SGEMM");
    module.def("sgemm_baseline", torch::wrap_pybind_function(sgemm_naive),
               "compatibility alias for naive SGEMM");
    module.def("matmul", torch::wrap_pybind_function(sgemm_naive),
               "compatibility alias for naive SGEMM");

    module.def("reduce_no_bankconflict",
               torch::wrap_pybind_function(reduce_no_bankconflict));
    module.def("reduce_add_during_load",
               torch::wrap_pybind_function(reduce_add_during_load));
    module.def("reduce_add_during_load_v2",
               torch::wrap_pybind_function(reduce_add_during_load_v2));
    module.def("reduce_unroll_last_warp",
               torch::wrap_pybind_function(reduce_unroll_last_warp));
    module.def("reduce_completely_unroll",
               torch::wrap_pybind_function(reduce_completely_unroll));
    module.def("reduce_multi_add",
               torch::wrap_pybind_function(reduce_multi_add));
    module.def("reduce_shuffle", torch::wrap_pybind_function(reduce_shuffle));
}
