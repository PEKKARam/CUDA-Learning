"""Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/setup.py"""

import os
from pathlib import Path
from packaging.version import parse

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

PACKAGE_NAME = "cuda_template"  # name of the Python package
PACKAGE_IMPORT_NAME = "my_cuda_kernels"  # the name that you will import in Python

CUDA_BUILD_MODE = os.getenv("CUDA_BUILD_MODE", "release").lower()
if CUDA_BUILD_MODE not in {"release", "profile", "debug"}:
    raise ValueError("CUDA_BUILD_MODE must be one of: release, profile, debug")

# PyTorch cannot infer architectures when no GPU is visible (common in CI and
# container builds). Keep this overrideable while avoiding an internal empty-list
# failure in torch.utils.cpp_extension.
if not os.getenv("TORCH_CUDA_ARCH_LIST") and not torch.cuda.is_available():
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

# Select your GPUs compute capability for faster compilation
COMPUTE_CAPABILITY = None  
# COMPUTE_CAPABILITY = "75"  # Turing 
# COMPUTE_CAPABILITY = "80"  # Ampere


# os.environ['CXX'] = '/usr/lib/ccache/g++'
# os.environ['CC'] = '/usr/lib/ccache/gcc'


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]


print("\nTorch version = {}".format(torch.__version__))
print(f"CUDA build mode = {CUDA_BUILD_MODE}")

if CUDA_HOME is not None:
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    print(f"CUDA version = {bare_metal_version}")
else:
    raise RuntimeError(
        "CUDA_HOME is not set. Install the CUDA toolkit or set CUDA_HOME "
        "before building this project."
    )
    
cc_flag = []
# An explicit value is useful on build machines without an attached GPU.
# Prefer TORCH_CUDA_ARCH_LIST when supplied by the PyTorch build tooling.
compute_capability = os.getenv("CUDA_COMPUTE_CAPABILITY", COMPUTE_CAPABILITY)
if compute_capability is not None:
    compute_capability = str(compute_capability).replace(".", "")
    if not compute_capability.isdigit():
        raise ValueError("CUDA_COMPUTE_CAPABILITY must look like 80 or 8.0")
    cc_flag.append("-gencode")
    cc_flag.append(f"arch=compute_{compute_capability},code=sm_{compute_capability}")

suffixes = [".cpp", ".cu"]
sources = [p for p in Path("csrc").rglob("*") if p.suffix in suffixes]

print(f"\nFound sources: {[str(p) for p in sources]}\n\n")

ext_modules = [
    CUDAExtension(
        name=PACKAGE_IMPORT_NAME,
        sources=sources,
        extra_compile_args={
            "cxx": (
                ["-O0", "-g"]
                if CUDA_BUILD_MODE == "debug"
                else ["-O2"]
            ),
            "nvcc": append_nvcc_threads(
                (
                    ["-O0", "-g", "-G"]
                    if CUDA_BUILD_MODE == "debug"
                    else ["-O2", "--generate-line-info"]
                    if CUDA_BUILD_MODE == "profile"
                    else ["-O2"]
                )
                + cc_flag
            ),
        },
        include_dirs=[],
    )
]

class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, (os.cpu_count() or 2) // 2)
            # Keep builds usable in minimal environments where psutil is absent.
            try:
                import psutil
                free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
                max_num_jobs_memory = max(1, int(free_memory_gb / 9))
                max_jobs = min(max_num_jobs_cores, max_num_jobs_memory)
            except ImportError:
                max_jobs = max_num_jobs_cores
            os.environ["MAX_JOBS"] = str(max(1, max_jobs))

        super().__init__(*args, **kwargs)

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "tests",
            "dist",
        )
    ),
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch",
    ],
)
