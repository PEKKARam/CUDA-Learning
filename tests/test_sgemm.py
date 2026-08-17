import pytest
import torch

my_cuda_kernels = (
    pytest.importorskip("my_cuda_kernels") if torch.cuda.is_available() else None
)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU is required"
)


@pytest.mark.parametrize("op_name", ["sgemm_naive", "sgemm_baseline"])
@pytest.mark.parametrize(
    "m,k,n",
    [(1, 1, 1), (2, 2, 20), (3, 7, 11), (31, 17, 29), (256, 256, 256)],
)
def test_sgemm(op_name, m, k, n):
    torch.manual_seed(1)
    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(k, n, device="cuda", dtype=torch.float32)

    actual = getattr(my_cuda_kernels, op_name)(a, b)
    expected = torch.matmul(a, b)

    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-2)
