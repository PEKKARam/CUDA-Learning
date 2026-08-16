import torch
import pytest

# Note: We need to import the CUDA kernels *after* importing torch
my_cuda_kernels = pytest.importorskip("my_cuda_kernels") if torch.cuda.is_available() else None
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")


ABS_TOL = 1e-4
REL_TOL = 1e-1


@pytest.mark.parametrize(
    "op_name", ["reduce_no_bankconflict",
                "reduce_add_during_load",
                "reduce_add_during_load_v2",
                "reduce_unroll_last_warp",
                "reduce_completely_unroll",
                "reduce_multi_add",
                "reduce_shuffle",
                ]
)
@pytest.mark.parametrize("size", [1, 255, 256, 1003, 65536])
def test_reduce(op_name, size):
    torch.manual_seed(1)
    x = torch.randn(size, device="cuda", dtype=torch.float32)

    out = getattr(my_cuda_kernels, op_name)(x)

    assert torch.isclose(out, x.sum(), atol=ABS_TOL, rtol=REL_TOL).item()
