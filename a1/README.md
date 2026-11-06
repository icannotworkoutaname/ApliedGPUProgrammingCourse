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

## Question 1：复现 CUDA 向量加法结果

本项目在 Google Colab GPU 运行时中执行，调用方式与课程教程一致。

### 1. 准备文件和环境

需要两个文件：

- [assignment1.ipynb](Q1/assignment1.ipynb)：编译、运行、绘图和 Nsight 分析。
- [vecAdd.cu](Q1/vecAdd.cu)：CPU/GPU 向量加法源码。

当前 notebook 保存的运行环境为 **Tesla T4、CUDA Toolkit 13.0.88**；源码每个 block 使用 **32 个线程**。

1. 在 Colab 打开 `assignment1.ipynb`。
2. 将运行时类型设为 GPU，选择 T4（如果可选）。
3. 手动把本目录中的 `vecAdd.cu` 上传到 Colab 的 `/content`。
4. 从头依次运行 notebook 单元格，重新生成结果。

下面带 `!`、`%` 的命令都放在 **Colab 代码单元格**中运行。

### 2. 检查环境并编译

```python
!nvidia-smi
!nvcc --version
```

```python
%cd /content
!ls -l vecAdd.cu
!nvcc -arch=sm_75 vecAdd.cu -o vecAdd
```

`sm_75` 对应 T4。修改源码后，需要重新上传并编译。

### 3. 运行两个指定长度

```python
!./vecAdd 512
!./vecAdd 263149
```

程序只接收向量长度 `N`；不传参数时默认 `N=512`。每次运行只启动一次 kernel。

输出列为：

```text
N,threads,blocks,H2D_ms,kernel_ms,D2H_ms,max_error
```

- `threads`：每个 block 的线程数。
- `blocks`：启动的 block 数；总线程数为 `threads × blocks`。
- `H2D_ms`：两个输入向量从 CPU 复制到 GPU 的时间。
- `kernel_ms`：CPU 观察到的 kernel 启动、执行和同步时间。
- `D2H_ms`：输出向量从 GPU 复制回 CPU 的时间。
- `max_error`：所有元素的 CPU/GPU 最大绝对误差；为 0 表示数值完全相等。

当前 notebook 保存的两个测试结果均为 `max_error=0`：

| N | threads | blocks |
| ---: | ---: | ---: |
| 512 | 32 | 16 |
| 263149 | 32 | 8224 |

重新运行后，检查自己的输出是否符合上述配置，并查看 `max_error`。

### 4. 复现计时表和堆叠柱状图

继续运行 notebook 中的“导入绘图库”“测量不同长度”“堆叠柱状图”和“查看 CPU/GPU 误差”单元格。

程序会测量 notebook 中配置的 16 组长度（512 至 16777216，包含 263149），并在 Colab `/content` 生成：

| 文件 | 内容 |
| --- | --- |
| `timing_results.csv` | 每个长度的线程配置、三段耗时和最大误差 |
| `timing_breakdown.png` | H2D、kernel、D2H 的堆叠柱状图 |

当前实现每个长度只测量一次，不预热、不重复取平均。CPU 参考计算、内存分配和结果校验不计入三段时间；计时边界会等待 GPU 完成。重新运行时耗时会波动，不要求与 notebook 已保存的数值完全一致。

### 5. 复现 Achieved Occupancy

先检查 Nsight Compute 的分析模块：

```python
!ncu --list-sections
```

再分别运行：

```python
!ncu --section Occupancy ./vecAdd 512
```

```python
!ncu --section Occupancy ./vecAdd 263149
```

读取输出中 **Achieved Occupancy** 一行，不要读取 **Theoretical Occupancy**。

当前 notebook 保存的一次 T4 运行结果为：

| N | Achieved Occupancy |
| ---: | ---: |
| 512 | 3.12% |
| 263149 | 33.67% |

这些数值是已保存运行的参考值，重新采集可能变化。绘图使用第 4 步的普通运行时间，不能使用 `ncu` 输出中的程序计时，因为 profiler 会增加开销。

### 6. 保存结果与常见问题

运行完成后，手动下载 `timing_results.csv`、`timing_breakdown.png`，并保存带最新输出的 notebook。Colab 运行时重置后需要重新上传源码、编译和运行。

- 找不到 `vecAdd.cu`：确认文件已上传到 `/content`，且名称一致。
- 找不到 `ncu`：当前运行时需要先配置 Nsight Compute。
- 出现 `ERR_NVGPUCTRPERM`：当前运行时不允许访问性能计数器，需要换到允许该功能的 GPU 环境采集占用率；普通计时和校验仍可单独运行。

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
