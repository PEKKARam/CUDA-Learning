import torch
import pytest

# Note: We need to import the CUDA kernels *after* importing torch
my_cuda_kernels = pytest.importorskip("my_cuda_kernels") if torch.cuda.is_available() else None
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")


ABS_TOL = 1e-4
REL_TOL = 1e-1


@pytest.mark.parametrize("size", [1, 255, 256, 1003, 65536])
def test_reduce_no_bankconflict(size):
    torch.manual_seed(1)
    x = torch.randn(size, device="cuda", dtype=torch.float32)

    out = my_cuda_kernels.reduce_no_bankconflict(x)

    assert torch.isclose(out, x.sum(), atol=ABS_TOL, rtol=REL_TOL).item()
