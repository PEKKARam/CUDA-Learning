/*
This file contains a C++ wrapper function for each CUDA kernel.
It registers the functions with Pybind11, allowing them to be called from
Python.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT32(x) \
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Square kernel

void launch_square_kernel(float* out, const float* inp, int n, int grid_size,
                          int block_size);

torch::Tensor square(const torch::Tensor& inp) {
    CHECK_INPUT(inp)  // check for correct device and contiguous data
    TORCH_CHECK(inp.dim() == 1,
                "Expected input tensor to have 1 dimension, but has ",
                inp.dim());
    CHECK_FLOAT32(inp);

    int n = inp.size(0);
    auto out = torch::zeros({n}, inp.options());
    if (n == 0) return out;

    int block_size = 256;
    int grid_size = cdiv(n, block_size);

    launch_square_kernel(out.data_ptr<float>(), inp.data_ptr<float>(), n,
                         grid_size, block_size);

    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Matrix multiplication kernel

void launch_matmul_kernel(float* out, const float* A, const float* B, int h,
                          int w, int k, dim3 grid_size, dim3 block_size);

torch::Tensor matmul(const torch::Tensor& A, const torch::Tensor& B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT32(A);
    CHECK_FLOAT32(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "matmul expects two 2D tensors");
    TORCH_CHECK(A.device() == B.device(),
                "A and B must be on the same CUDA device");
    int h = A.size(0);
    int w = B.size(1);
    int k = A.size(1);
    TORCH_CHECK(k == B.size(0), "Size mismatch!");
    auto out = torch::zeros({h, w}, A.options());
    if (out.numel() == 0) return out;

    dim3 block_size(16, 16);
    dim3 grid_size(cdiv(w, block_size.x), cdiv(h, block_size.y));

    launch_matmul_kernel(out.data_ptr<float>(), A.data_ptr<float>(),
                         B.data_ptr<float>(), h, w, k, grid_size, block_size);

    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Reduce kernel

using ReduceLauncher = void (*)(const float*, float*, size_t, dim3, dim3);

void launch_reduce_no_bankconflict_kernel(const float* input, float* output,
                                          size_t n, dim3 grid_size,
                                          dim3 block_size);
void launch_reduce_add_during_load_kernel(const float* input, float* output,
                                          size_t n, dim3 grid_size,
                                          dim3 block_size);
void launch_reduce_add_during_load_v2_kernel(const float* input, float* output,
                                             size_t n, dim3 grid_size,
                                             dim3 block_size);
void launch_reduce_unroll_last_warp_kernel(const float* input, float* output,
                                           size_t n, dim3 grid_size,
                                           dim3 block_size);
void launch_reduce_completely_unroll_kernel(const float* input, float* output,
                                            size_t n, dim3 grid_size,
                                            dim3 block_size);

void launch_reduce_multi_add_kernel(const float* input, float* output,
                                    size_t n);
void launch_reduce_shuffle_kernel(const float* input, float* output, size_t n);

// python wrapper

torch::Tensor reduce_impl(const torch::Tensor& input, ReduceLauncher launcher,
                          unsigned int values_per_block) {
    CHECK_INPUT(input);
    CHECK_FLOAT32(input);
    TORCH_CHECK(input.dim() == 1, "reduce expects a 1D tensor");

    auto output = torch::zeros({}, input.options());
    if (input.numel() == 0) return output;

    dim3 block_size(256);
    dim3 grid_size(cdiv(input.numel(), block_size.x * values_per_block));
    launcher(input.data_ptr<float>(), output.data_ptr<float>(), input.numel(),
             grid_size, block_size);
    return output;
}

torch::Tensor reduce_no_bankconflict(const torch::Tensor& input) {
    return reduce_impl(input, launch_reduce_no_bankconflict_kernel, 1);
}

torch::Tensor reduce_add_during_load(const torch::Tensor& input) {
    // 每个线程处理两个元素，每个block处理两个block大小的数量，因此需要的block数量减半
    return reduce_impl(input, launch_reduce_add_during_load_kernel, 2);
}

torch::Tensor reduce_add_during_load_v2(const torch::Tensor& input) {
    // 每个线程处理两个元素，block size减半，block数量不变
    return reduce_impl(input, launch_reduce_add_during_load_v2_kernel, 1);
}

torch::Tensor reduce_unroll_last_warp(const torch::Tensor& input) {
    // 每个线程处理两个元素，block size减半，block数量不变，最后使用warp内reduce
    return reduce_impl(input, launch_reduce_unroll_last_warp_kernel, 1);
}

torch::Tensor reduce_completely_unroll(const torch::Tensor& input) {
    // 固定使用256线程，每个线程读取两个元素
    return reduce_impl(input, launch_reduce_completely_unroll_kernel, 2);
}

torch::Tensor reduce_multi_add(const torch::Tensor& input) {
    CHECK_INPUT(input);
    CHECK_FLOAT32(input);
    TORCH_CHECK(input.dim() == 1, "reduce expects a 1D tensor");

    auto output = torch::zeros({}, input.options());
    if (input.numel() == 0) return output;

    launch_reduce_multi_add_kernel(
        input.data_ptr<float>(), output.data_ptr<float>(), input.numel());

    return output;
}

torch::Tensor reduce_shuffle(const torch::Tensor& input) {
    CHECK_INPUT(input);
    CHECK_FLOAT32(input);
    TORCH_CHECK(input.dim() == 1, "reduce expects a 1D tensor");

    auto output = torch::zeros({}, input.options());
    if (input.numel() == 0) return output;

    launch_reduce_shuffle_kernel(
        input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
    return output;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Register the wrapper functions with Pybind to make them available in Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square", torch::wrap_pybind_function(square), "square");
    m.def("matmul", torch::wrap_pybind_function(matmul), "matmul");

    // reduce
    m.def("reduce_no_bankconflict",
          torch::wrap_pybind_function(reduce_no_bankconflict),
          "sum reduction without shared-memory bank conflicts");
    m.def("reduce_add_during_load",
          torch::wrap_pybind_function(reduce_add_during_load),
          "sum reduction that adds two values while loading");
    m.def("reduce_add_during_load_v2",
          torch::wrap_pybind_function(reduce_add_during_load_v2),
          "sum reduction with half as many threads per block");
    m.def("reduce_unroll_last_warp",
          torch::wrap_pybind_function(reduce_unroll_last_warp),
          "sum reduction with unrolling last warp");
    m.def("reduce_completely_unroll",
          torch::wrap_pybind_function(reduce_completely_unroll),
          "sum reduction with unrolling completely");
    m.def("reduce_multi_add", torch::wrap_pybind_function(reduce_multi_add),
          "sum reduction that adds multiple numbers while loading");
    m.def("reduce_shuffle", torch::wrap_pybind_function(reduce_shuffle),
          "sum reduction using shuffle");
}
