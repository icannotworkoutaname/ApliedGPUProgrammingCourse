# Question 2 — 2D Dense Matrix Multiplication (naive CUDA)

Row-major `C(M x P) = A(M x N) * B(N x P)` with a templated naive `__global__`
kernel: one thread per `C` element, a serial `k`-loop over the shared
dimension, no tiling / shared-memory optimization. `float` and `double` share
the same template implementation, selected by the last CLI argument.

## Environment

Google Colab with an NVIDIA T4 (compute capability 7.5), driver 580.82.07,
CUDA 13.0.88 and Nsight Compute 2025.3.1. The saved
`results/environment.txt` records these versions for the submitted run.

## Reproduce in Colab

Open `hw1_q2_colab.ipynb` in Colab, select a T4 GPU runtime and run all cells.
The notebook writes the two source files, compiles the program, runs all
validation and timing cases, collects both occupancy measurements and downloads
`hw1_q2_results.zip`.

## Build

The notebook uses:

```bash
nvcc -O3 -std=c++17 -arch=sm_75 matMul.cu -o matMul
```

## Run

```bash
./matMul                          # defaults: 128 256 32 double
./matMul 128 256 32 double        # required small shape (double)
./matMul 128 256 32 float         # required small shape (float)
./matMul 1024 8191 8197 double    # required large shape (double)
./matMul 1024 8191 8197 float     # required large shape (float)
```

The program prints exactly one CSV header line and one data row to stdout;
all diagnostics go to stderr. Exit code is nonzero on bad arguments, CUDA
errors, or validation failure.

## Benchmark sweep (stacked bar charts)

```bash
python3 benchmark.py ./matMul --outdir results
```

Produces `results/timing_double.csv`, `results/timing_float.csv` and stacked
PNG charts `results/timing_double.png`, `results/timing_float.png`. The sweep
covers 128x256x32 up to 2048x2048x2048 for both types (the huge
1024x8191x8197 occupancy case is run separately with the commands above).

## Profiling with Nsight Compute (achieved occupancy)

```bash
ncu --section Occupancy ./matMul 128 256 32 double
ncu --section Occupancy ./matMul 1024 8191 8197 double
```

## Output fields

| Field | Meaning |
|---|---|
| `type` | `double` or `float` |
| `M,N,P` | matrix dimensions |
| `threads_per_block` | block size, always `256` (16x16) |
| `blocks_x` | `ceil(P/16)` |
| `blocks_y` | `ceil(M/16)` |
| `total_threads` | `blocks_x * blocks_y * 256` (launched threads) |
| `active_threads` | `M * P` (threads that compute a result) |
| `H2D_ms` | CPU wall time to copy A and B host→device |
| `kernel_ms` | CPU wall time for kernel launch + completion |
| `D2H_ms` | CPU wall time to copy C device→host |
| `max_abs_error` | max absolute error vs CPU reference |
| `max_rel_error` | max relative error vs CPU reference |
| `validation` | `full`, or `sampled:<count>` |

## Timing and validation boundaries

- Timers are `std::chrono::steady_clock` CPU wall times around the three
  regions, each bracketed by `cudaDeviceSynchronize()`: H2D copy of A+B,
  kernel launch + sync, D2H copy of C. Allocation, fixed-pattern host
  initialization, and validation are **not** timed.
- Validation compares against a full sequential CPU reference **in the same
  precision** when `M*N*P <= 2^26`; for larger cases a deterministic sample
  is checked instead (first/last, quarter/middle/three-quarter rows and
  columns), reported as `sampled:<count>`. This avoids an O(M*N*P) CPU
  reference for the 1024x8191x8197 case. Combined tolerance
  `|err| <= atol + rtol*|ref|` with `atol = rtol = 1e-9` (double) and
  `1e-3` (float); failure exits nonzero.

## Thread/block counts (deterministic, from the launch config)

- `128x256x32`, 16x16 blocks: grid = (2, 8) = 16 blocks, `total_threads =
  16*256 = 4096`, `active_threads = 128*32 = 4096`.
- `1024x8191x8197`: grid = (`ceil(8197/16)`=513, `ceil(1024/16)`=64) =
  32832 blocks, `total_threads = 32832*256 = 8404992`, `active_threads =
  1024*8197 = 8393728`; the difference is padding threads in edge blocks.
  All index arithmetic uses `size_t` and checked multiplication, so these
  dimensions (C alone is ~64 MiB in double) are safe.
