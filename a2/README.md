# HW2 - Assignment Group 40

Members: Yuanyu Zhang, Hongen Wang, Xinyang Li

Code and steps to reproduce the answers of HW2. `report.tex` is the source of the report (`HW2_Group40.pdf`).

## Layout

```
README.md                 this file
report.tex                report source
kth-logo.png              cover logo
Q1/                       1D convolution
    conv1d.cu             the course skeleton (hw2-convolution.cu), completed
    run_local.sh          builds, runs and profiles everything used in the report
    hw2_q1_colab.ipynb    the same steps as a Colab notebook (Tesla T4)
    results/              output of run_local.sh, including the report screenshots
Q2/                       Tiled matrix multiplication
    gemm_tiled.cu         CPU reference, naive gemm and runtime-sized tiled GEMM
    benchmark.py          ten-shape timing sweep and grouped bar chart
    results/              Colab output, Nsight Compute reports and screenshots
Bonus/                    GPU particle mover
    iPIC3D-mini/          pinned upstream miniPIC source with CPU/GPU movers
    run_compare.sh        full GEM_2D CPU/GPU output and timing comparison
    compare_vtk.py        element-wise ASCII VTK validation with NaN/Inf rejection
    results/              correctness, simulation output and screenshots
Q3/                       Llama2 inference on the GPU
    Makefile              downloads llama2.c run.c, applies run.patch, builds the CPU and CUDA versions
    run.patch             changes to run.c: matmul counter and calls to the CUDA matmul
    matmul_cuda.cu/.h     CUDA matmul, naive and optimized (weights kept on the GPU)
    LICENSE               llama2.c license
    hw2_Q3.ipynb          Colab notebook used for all Q3 runs and profiles
    run_records/          program output of the four runs in the report
    report_figures/       notebook screenshots used in the report
    output/pdf/           the earlier stand-alone Q3 report and source ZIP
```

---

## Question 1: 1D Convolution

### What the code does

`Q1/conv1d.cu` keeps the structure of the skeleton. The mask width is 5, the mask is
`[-0.25, 0.5, 1.0, 0.5, 0.25]` and the input is random. Inputs outside the array count as 0.
We added a `width` argument to both kernels, since the skeleton signatures have no length parameter.

`convolution_1D_basic` computes one output per thread and reads everything from global memory.
In `convolution_1D_tiled`, a block of `TILE_WIDTH` threads first copies its part of the input,
2 halo elements on each side and the mask into shared memory, and then computes from there.

The program runs each kernel once as a warm-up, then once with the skeleton's CPU timers, and checks
both results against each other and against a CPU version. At the end it launches each kernel 100 times
and prints the average time from CUDA events.

`TILE_WIDTH` is set at compile time (default 128). The optional second argument sets the shared memory
carveout of the tiled kernel in percent.

### Environment

We ran Q1 on an RTX 3060 Laptop GPU (compute capability 8.6) under WSL2 Ubuntu 24.04 on
Windows 11, driver 566.36, with CUDA 12.6.85, Nsight Compute 2024.3.2 and Nsight Systems 2026.6.1.

Two things were needed on this setup:

1. Nsight Compute needs access to the GPU performance counters. We allowed it in the NVIDIA Control Panel
   (Desktop > Enable Developer Settings, then Developer > Manage GPU Performance Counters > allow all users)
   and restarted WSL with `wsl --shutdown`.
2. The Nsight Systems 2024.5 that comes with CUDA 12.6 recorded no GPU kernels under WSL2 with this driver.
   We used the Linux command line tool of Nsight Systems 2026.6.1 instead. It is the `target-linux-x64`
   folder of the Windows installation, copied to `~/nsys2026/target-linux-x64` in WSL.

### How to reproduce

```bash
bash Q1/run_local.sh
```

This writes to `Q1/results/`:

- `run_<N>.txt`: program output for N = 1024 to 16384 with TILE_WIDTH = 128 (Figure 1 of the report)
- `sweep_nsys.txt`: kernel time for every N and tile size, measured with Nsight Systems (Table 1)
- `q1_nsys.nsys-rep`, `nsys_stats.txt`: Nsight Systems profile of N = 16384 (Figures 2 and 3)
- `ncu_convolution_1D_*.txt`, `q1_tiled_full.ncu-rep`: Nsight Compute results (Table 2)
- `carveout.txt`: the carveout test (Table 3)
- `sweep.txt`: the same sweep timed with CUDA events. These numbers are noisy under WSL2 and were not used.

Paths and the GPU architecture can be changed with environment variables, for example
`ARCH=sm_75 CUDA_BIN=/usr/local/cuda/bin NSYS=/path/to/nsys bash Q1/run_local.sh`.

A single run by hand:

```bash
cd Q1
nvcc -O3 -arch=sm_86 -lineinfo -DTILE_WIDTH=128 conv1d.cu -o conv1d_T128
./conv1d_T128 16384        # N = 16384
./conv1d_T128 16384 0      # N = 16384, carveout 0 %
```

To use Colab instead, open `Q1/hw2_q1_colab.ipynb`, choose a T4 GPU runtime and run the cells from the top.

---

## Question 2: Tiled Matrix Multiplication

### What the code does

`Q2/gemm_tiled.cu` computes row-major `C(MxN) = A(MxK) * B(KxN)`. It runs the
single-threaded CPU reference once, then naive `gemm()` and `tiled_gemm()` with
8x8, 16x16 and 32x32 tiles. The tiled kernel uses runtime `tileX`/`tileY` and
dynamic shared memory; out-of-range tile elements are zero-filled. Every output
element from every CUDA kernel is compared with the full CPU reference.

Each CUDA kernel has one warm-up launch followed by 10 measured launches. The
reported kernel time is the median CUDA-event duration. Set `GEMM_REPS=1` for a
single launch without warm-up when profiling.

### Environment

Google Colab, Tesla T4 (compute capability 7.5), CUDA 13.0.88 and Nsight
Compute. Final measurements were executed through the Colab CLI session `a2`;
raw outputs and output figures are saved under `Q2/results/`.

### How to reproduce

```bash
cd Q2
make ARCH=sm_75
./gemm_tiled 1024 1024 1024 float
./gemm_tiled 513 8192 1023 float
python3 benchmark.py ./gemm_tiled --type float --outdir results

GEMM_REPS=1 ncu --section Occupancy --section MemoryWorkloadAnalysis \
  ./gemm_tiled 1024 1024 1024 float > results/ncu_1024.txt 2>&1
GEMM_REPS=1 ncu --section Occupancy --section MemoryWorkloadAnalysis \
  ./gemm_tiled 513 8192 1023 float > results/ncu_513x8192x1023.txt 2>&1
```

The benchmark writes long and wide CSV tables plus `timing_float.png`, a grouped
bar chart of the CPU, naive and three tiled implementations over ten shapes.

---

## Question 3: Accelerate Llama2 Inference

### What the code does

The Makefile downloads `run.c` of llama2.c at commit `350e04f` and applies `Q3/run.patch`. From the
patched file it builds four programs:

- `run`: the original FP32 CPU version
- `run_count`: the CPU version that counts the calls to `matmul`
- `run_naive`: `matmul` on the GPU. Every call allocates device buffers, copies the input and the weights,
  runs the kernel, copies the result back and frees the buffers.
- `run_optimized`: the same kernel, but the checkpoint is copied to the GPU once at startup and the input
  and output buffers are reused. Each call only copies the input vector and the result.

In the kernel one thread computes one row of `w @ x`, with 128 threads per block. With the environment
variable `CHECK_MATMUL=1`, both CUDA versions compare every output element with the CPU result.

### Environment

Google Colab, Tesla T4, CUDA 13.0.88, Nsight Systems 2025.3.1.

### How to reproduce

Open `Q3/hw2_Q3.ipynb` in Colab with a T4 GPU runtime and run the cells from the top. The notebook
writes the sources, downloads the model and builds everything. Results are written to
`/content/llama2_q3/results/`.

By hand on a Linux machine with CUDA:

```bash
cd Q3
curl -fL -o stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories15M.bin
curl -fsSL -o tokenizer.bin https://raw.githubusercontent.com/karpathy/llama2.c/350e04fe35433e6d2941dce5a1f53308f87058eb/tokenizer.bin
make                                   # CUDA_ARCH=sm_75 for a T4 if "native" does not work

./run stories15M.bin                                    # CPU, README command
./run_count stories15M.bin -t 0 -s 42 -n 256            # prints "matmul calls" on stderr
./run_naive stories15M.bin -t 0 -s 42 -n 256
./run_optimized stories15M.bin -t 0 -s 42 -n 256

# correctness: compare every matmul output with the CPU for 8 steps
CHECK_MATMUL=1 ./run_naive stories15M.bin -t 0 -s 42 -n 8 > /dev/null
CHECK_MATMUL=1 ./run_optimized stories15M.bin -t 0 -s 42 -n 8 > /dev/null

# profiling
nsys profile --trace=cuda --sample=none --cpuctxsw=none -o naive ./run_naive stories15M.bin -t 0 -s 42 -n 256
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum naive.nsys-rep
```

The optimized version is profiled the same way with `./run_optimized`.

---

## Bonus: GPU Particle Mover

`Bonus/iPIC3D-mini` is based on KTH-ScaLab/DD2360-HT25 at commit
`53e7ca59d1904f586b2940d2cfc3f30351873e7b`. The original CPU `mover_PC()` is
kept alongside `mover_PC_gpu()`. The GPU wrapper copies the particle arrays and
contiguous field/grid arrays to the device; one CUDA thread moves one particle.
Separate object directories and binaries prevent CPU/GPU build contamination.

Both versions use the unmodified `GEM_2D.inp`. Build and run on a CUDA machine:

```bash
cd Bonus/iPIC3D-mini
make -j2 cpu
make -j2 gpu compare ARCH=sm_75

# Compare every x/y/z/u/v/w element for every species after one mover call.
./bin/mover_compare.out inputfiles/GEM_2D.inp

# Run the complete ten-cycle CPU and GPU simulations and compare their VTK output.
cd ..
./run_compare.sh
```

The full-simulation script writes separate `run_cpu/` and `run_gpu/` directories.
Report artifacts, console logs, numerical output comparison and Colab screenshots
are collected in `Bonus/results/`. The GPU program reports both end-to-end mover
time (allocation, transfers and kernel) and kernel-only CUDA-event time.
