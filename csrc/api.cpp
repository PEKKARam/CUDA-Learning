/*
This file contains a C++ wrapper function for each CUDA kernel.
It registers the functions with Pybind11, allowing them to be called from Python.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

////////////////////////////////////////////////////////////////////////////////////////////////////

// Square kernel

void launch_square_kernel(float* out, const float* inp, int n, int grid_size, int block_size);

torch::Tensor square(const torch::Tensor& inp) {
    CHECK_INPUT(inp)  // check for correct device and contiguous data
    TORCH_CHECK(inp.dim() == 1, "Expected input tensor to have 1 dimension, but has ", inp.dim());
    CHECK_FLOAT32(inp);

    int n = inp.size(0);
    auto out = torch::zeros({n}, inp.options());
    if (n == 0) return out;

    int block_size = 256;
    int grid_size = cdiv(n, block_size);

    launch_square_kernel(out.data_ptr<float>(), inp.data_ptr<float>(), n, grid_size, block_size);

    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Matrix multiplication kernel

void launch_matmul_kernel(float* out, const float* A, const float* B, int h, int w, int k, dim3 grid_size, dim3 block_size); 

torch::Tensor matmul(const torch::Tensor& A, const torch::Tensor& B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    CHECK_FLOAT32(A); CHECK_FLOAT32(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "matmul expects two 2D tensors");
    TORCH_CHECK(A.device() == B.device(), "A and B must be on the same CUDA device");
    int h = A.size(0);
    int w = B.size(1);
    int k = A.size(1);
    TORCH_CHECK(k==B.size(0), "Size mismatch!");
    auto out = torch::zeros({h, w}, A.options());
    if (out.numel() == 0) return out;

    dim3 block_size(16, 16);
    dim3 grid_size(cdiv(w, block_size.x), cdiv(h, block_size.y));

    launch_matmul_kernel(out.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), h, w, k, grid_size, block_size);

    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Add reduce kernel

template<typename T>
void launch_sum_reduce_kernel(const T *input, T *output)

////////////////////////////////////////////////////////////////////////////////////////////////////

// Register the wrapper functions with Pybind to make them available in Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square", torch::wrap_pybind_function(square), "square");
    m.def("matmul", torch::wrap_pybind_function(matmul), "matmul");
}
