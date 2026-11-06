# HW1 - Assignment Group 40

Members: Yuanyu Zhang, Hongen Wang, Xinyang Li

Code and reproduction steps for the three programming questions of HW1.
Question 1 is a CUDA vector addition, Question 2 is a 2D matrix multiplication, and Question 3 profiles a quantum Fourier transform on the QuEST CUDA backend.

## Repository layout

```
README.md                   reproduction steps for all questions
HW1_Group40.pdf             complete report
report.tex                  report source
kth-logo.png                KTH cover logo
Q1/
    vecAdd.cu               CPU and GPU vector addition
    assignment1.ipynb       build, run, plots and Nsight Compute profiling
    timing_breakdown.png    saved timing plot
Q2/
    matMul.cu               float/double 2D matrix multiplication
    benchmark.py            timing sweep and stacked bar charts
    hw1_q2_colab.ipynb      self-contained Colab execution and saved outputs
    README.md               build, run and profiling commands
    results/                measured CSV data, plots and Nsight Compute logs
Q3/
    qft_bench_naive.c       benchmark provided by the course, unmodified
    hw1_q3_colab.ipynb      commands used for Question 3
    assets/                 screenshots used in the report
```

## Environment

- Questions 1 and 3: Google Colab, Tesla T4 (compute capability 7.5), Ubuntu 24.04, NVIDIA driver 580.82.07. Question 1 used CUDA Toolkit 13.0.88 and Nsight Compute 2025.3.1.
- Question 2: Google Colab, Tesla T4 (compute capability 7.5), NVIDIA driver 580.82.07, CUDA Toolkit 13.0.88 and Nsight Compute 2025.3.1. The notebook records the complete run.

The submitted plots and occupancy values are the saved runs described in the report. Timing values vary between runs; profiler-instrumented timings are not used for the plots.

---

## Question 1: Reproducing CUDA Vector Addition Results

This project runs in a Google Colab GPU runtime and uses the same execution workflow as the course tutorial.

### 1. Prepare the files and environment

Two files are needed:

- [assignment1.ipynb](Q1/assignment1.ipynb): compilation, execution, plotting, and Nsight profiling.
- [vecAdd.cu](Q1/vecAdd.cu): CPU/GPU vector addition source code.

The runtime environment recorded in the current notebook is **Tesla T4 and CUDA Toolkit 13.0.88**; the source code uses **32 threads per block**.

1. Open `assignment1.ipynb` in Colab.
2. Set the runtime type to GPU and select T4 (if available).
3. Manually upload `vecAdd.cu` from this directory to `/content` in Colab.
4. Run the notebook cells in order from the beginning to regenerate the results.

Run the commands below that contain `!` or `%` in **Colab code cells**.

### 2. Check the environment and compile

```python
!nvidia-smi
!nvcc --version
```

```python
%cd /content
!ls -l vecAdd.cu
!nvcc -arch=sm_75 vecAdd.cu -o vecAdd
```

`sm_75` corresponds to T4. After modifying the source code, upload it again and recompile.

### 3. Run the two specified lengths

```python
!./vecAdd 512
!./vecAdd 263149
```

The program takes only the vector length `N`; if no argument is provided, it defaults to `N=512`. Each run launches the kernel once.

The output columns are:

```text
N,threads,blocks,H2D_ms,kernel_ms,D2H_ms,max_error
```

- `threads`: the number of threads per block.
- `blocks`: the number of blocks launched; the total number of threads is `threads × blocks`.
- `H2D_ms`: the time to copy the two input vectors from the CPU to the GPU.
- `kernel_ms`: the kernel launch, execution, and synchronization time observed by the CPU.
- `D2H_ms`: the time to copy the output vector from the GPU back to the CPU.
- `max_error`: the maximum absolute CPU/GPU error across all elements; 0 means that the results are numerically identical.

Both test results saved in the current notebook have `max_error=0`:

| N | threads | blocks |
| ---: | ---: | ---: |
| 512 | 32 | 16 |
| 263149 | 32 | 8224 |

After rerunning, check whether your output matches the configuration above and inspect `max_error`.

### 4. Reproduce the timing table and stacked bar chart

Continue running the notebook cells titled "Import plotting libraries", "Measure different lengths", "Stacked bar chart", and "View CPU/GPU errors".

The program measures the 16 vector lengths configured in the notebook (512 to 16777216, including 263149) and generates the following files under `/content` in Colab:

| File | Contents |
| --- | --- |
| `timing_results.csv` | Thread configurations, three-stage timings, and maximum errors for each length |
| `timing_breakdown.png` | Stacked bar chart of H2D, kernel, and D2H times |

The current implementation measures each length only once, without warmup or averaging over repeated runs. CPU reference computation, memory allocation, and result validation are excluded from the three-stage timings; the timing boundaries wait for the GPU to finish. Timings fluctuate between runs and do not need to match the values already saved in the notebook exactly.

### 5. Reproduce Achieved Occupancy

First check the Nsight Compute analysis sections:

```python
!ncu --list-sections
```

Then run each command separately:

```python
!ncu --section Occupancy ./vecAdd 512
```

```python
!ncu --section Occupancy ./vecAdd 263149
```

Read the **Achieved Occupancy** row in the output, not **Theoretical Occupancy**.

The results of one T4 run saved in the current notebook are:

| N | Achieved Occupancy |
| ---: | ---: |
| 512 | 3.12% |
| 263149 | 33.67% |

These values are references from the saved run and may change when collected again. Use the normal execution times from step 4 for plotting, not the program timings printed under `ncu`, because the profiler adds overhead.

### 6. Save results and common issues

After execution, manually download `timing_results.csv` and `timing_breakdown.png`, and save the notebook with the latest outputs. If the Colab runtime is reset, upload the source code again, recompile, and rerun.

- `vecAdd.cu` not found: confirm that the file has been uploaded to `/content` and that its name matches.
- `ncu` not found: configure Nsight Compute in the current runtime first.
- `ERR_NVGPUCTRPERM` appears: the current runtime does not allow access to performance counters. Switch to a GPU environment that permits this access to collect occupancy; normal timing and validation can still run separately.

---

## Question 2 - 2D Dense Matrix Multiplication

### Source files

- `Q2/matMul.cu`: naive row-major matrix multiplication with a templated `float`/`double` kernel, 2D grid, CPU validation and three-stage timing.
- `Q2/benchmark.py`: runs both data types and saves CSV files and stacked bar charts under `Q2/results/`.
- `Q2/README.md`: complete build, run, benchmark and Nsight Compute commands.

### Build

```bash
cd Q2
nvcc -O3 -std=c++17 -arch=sm_75 matMul.cu -o matMul
```

### Run

```bash
./matMul 128 256 32 double
./matMul 1024 8191 8197 double
./matMul 128 256 32 float
python benchmark.py
```

### Reproducing the answers

Open `Q2/hw1_q2_colab.ipynb` in Colab, select a T4 runtime and run all cells to reproduce every Q2 result. Alternatively, run `python benchmark.py` after building to regenerate `timing_double.csv`, `timing_float.csv`, `timing_double.png` and `timing_float.png`. Use the `ncu --section Occupancy` commands in `Q2/README.md` to reproduce the two occupancy measurements. The normal timing runs, not profiler-instrumented runs, generate the report plots.

---

## Question 3 - Quantum Fourier Transform (QuEST CUDA backend)

### Source files

- `Q3/qft_bench_naive.c` is the benchmark from the course page. It was not edited.
- `Q3/hw1_q3_colab.ipynb` holds all commands used for this question: environment check, build, the 24-qubit run, the 28-qubit memory measurement, the Nsight Systems profiles and the result extraction.

QuEST itself is not part of the submission. It is cloned from GitHub in the build step.

### Setup and build

A GPU runtime is needed. From a QuEST checkout with `qft_bench_naive.c` copied into its root:

```bash
git clone https://github.com/QuEST-Kit/QuEST.git
cd QuEST
cp /path/to/qft_bench_naive.c .

# compute capability: 75 on a T4
cmake -B build -DQUEST_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_FLAGS="-lineinfo" \
      -DUSER_SOURCE_NAMES=qft_bench_naive.c -DUSER_OUTPUT_EXE_NAME=qft_naive
cmake --build build -j4
```

The compute capability can be read with `nvidia-smi --query-gpu=name,compute_cap --format=csv`. The notebook does this automatically.

### Run

```bash
./build/qft_naive 24 5          # 24 qubits, 5 timed repetitions
./build/qft_naive 28 5          # 28 qubits

# Nsight Systems timeline and statistics (repeat for 16, 24 and 28 qubits)
nsys profile --trace=cuda,osrt --stats=true -o qft_24q ./build/qft_naive 24 5
```

The memory measurement for 28 qubits uses `/usr/bin/time -v`, which is not installed on Colab. Install it first with `apt-get install -y time`.

### Reproducing the answers

The simplest way is the notebook: open `Q3/hw1_q3_colab.ipynb` in Colab, select a T4 runtime and run the cells in order. The benchmark file is uploaded when the "Benchmark source" cell asks for it. The last cells pack all outputs into `hw1_q3_results.zip`.

| Question | Notebook section | Output |
|---|---|---|
| 3.1 24-qubit run | Q1: 24-qubit run | `run_24q.txt` |
| 3.2 memory at 28 qubits | Q2: Memory usage, 28 qubits | `run_28q.txt`, `time_28q.txt`, `mem_samples_28q.csv` |
| 3.3 top three kernels | Q3: Kernel summary | `nsys_24q_stats.txt`, `qft_24q.nsys-rep` |
| 3.4 kernel to gate mapping | read from the QuEST sources (`quest/src/gpu/gpu_kernels.cuh`, `quest/src/api/operations.cpp`) | - |
| 3.5 CUDA API calls | open `qft_24q.nsys-rep` in the Nsight Systems GUI | - |
| 3.6 16 and 28 qubits | Q6: Benchmark timing and kernel summaries | `nsys_16q_stats.txt`, `nsys_28q_stats.txt`, `qft_16q.nsys-rep`, `qft_28q.nsys-rep` |

The `.nsys-rep` files are opened with the Nsight Systems GUI on a local machine.
