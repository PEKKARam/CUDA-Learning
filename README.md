# CUDA + PyTorch template

A clean and simple template for developing CUDA C++ kernels and testing them in Python/PyTorch 🚀🚀.

Tested on Ubuntu 20.04.

## Structure

```
.
├── README.md
├── benchmark.py   // use this script to benchmark your kernels
├── csrc           // C/C++ CUDA files
│   ├── api.cpp    // define the Python interface here
│   ├── matmul.cu  // a sample CUDA kernel
│   └── square.cu  // another sample CUDA kernel
├── requirements.txt
├── setup.py       // your code is compiled through this script
└── tests          // test the correctness of your kernels here
    ├── test_matmul.py
    └── test_square.py
```

## 💻 Installation

First, install CUDA and PyTorch. The preferred way to install CUDA is through Conda (see [here](https://x.com/jeremyphoward/status/1697435241152127369)). Also note that you will need an Nvidia GPU to run this.

```shell
conda create -n cuda-kernels  # create a new Conda environment
conda activate cuda-kernels  # activate the environment
conda install cuda -c nvidia/label/cuda-12.4.0  # choose the desired CUDA version (here we use 12.4)
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia/label/cuda-12.4.0  # install Pytorch using the previously mentioned CUDA version
```

Finally, install the remaining dependencies and this extension in editable mode:

```shell
pip install -r requirements.txt
pip install -e .
```

## 🔥 How to run

This repo contains two sample CUDA kernels that you can use as a starting point: `csrc/square.cu` and `csrc/matmul.cu`.

The first step is to compile the kernels, which is done by running `setup.py`.

```shell
pip install -e .
```

This automatically compiles the source files found in `csrc`. Re-run the command after changing CUDA sources. For faster builds, set `TORCH_CUDA_ARCH_LIST` (for example `8.0`) or `CUDA_COMPUTE_CAPABILITY=80`.

Once you've compiled, you can use the provided scripts to:

1. Test your kernels:

    ```shell
    pytest -v -s
    ```

2. Benchmark your kernels:

    ```shell
    python benchmark.py --op matmul --size 1000 --warmup 10 --iters 1000
    ```

These two commands should work out of the box for the two kernels mentioned above.

That's it! Now, you can start hacking away and create your own CUDA kernels.

## Detailed profiling

Once you start writing more serious kernels, you probably want to do more precise benchmarking.  The `benchmark.py` script is a simple script for timing your kernels, but it is not as precise as using a profiler. If you want to get detailed information about the performance bottlenecks of your kernels, consider using the `ncu` profiler. For example:

```shell
ncu --kernel-name square_kernel --launch-skip 10 --launch-count 1 python benchmark.py --op square --iters 1 --warmup 10
```

For timeline-level analysis:

```shell
nsys profile --trace=cuda,nvtx,osrt -o profile python benchmark.py --iters 100
```

`--kernel-name` filters the report to the selected kernel; `--launch-skip` avoids
collecting the warmup launches.

Note: this will not work on most cloud GPU instances out of the box. See the [running ncu profiler on a cloud GPU instance](#running-ncu-profiler-on-a-cloud-gpu-instance) section below to fix this.
    

## Running ncu profiler on a cloud GPU instance

The Nsight profiler (`ncu`) is a very useful tool to profile CUDA kernels.  However, it will not run out of the box on cloud GPUs. If you run `ncu`, you might get an output like this:

```shell
$ ncu ./benchmark
==PROF== Connected to process 2258 (/mnt/tobias/benchmark)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```
 

To fix this, you can run `ncu` with `sudo`. Note however that when you run `sudo`, your environment variables change, which means that `ncu` may no longer be on the PATH. This can be fixed by specifying the full path to `ncu`. E.g.:

```bash
which ncu  # check ncu path
sudo /opt/conda/envs/cuda-kernels/bin/ncu  # pass ncu path
```

In my case, ncu is provided through Conda. To make running ncu more convenient, you can directly add your Conda path to the "sudoers" file. Do this as follows:

```shell
sudo visudo
```

 Add your conda environment's bin directory to the Defaults secure_path line: 

```shell
Defaults secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/path/to/conda/env/bin"
```

Replace /path/to/conda/env/bin with the actual path to your conda environment's bin directory.


You can now run ncu simply by prepending `sudo`:

```shell
sudo ncu
```

## TODO

- [ ] Docker
