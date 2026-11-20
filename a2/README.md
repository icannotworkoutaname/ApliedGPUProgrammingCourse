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
