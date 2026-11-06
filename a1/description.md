# Assignment 1

> You as a student are fully responsible for all submitted materials and must be able to defend and explain them without the support of generative AI.

## Submission

- Submit a PDF file fulfilling the following requirements:
  - Named as `HW1_GroupN.pdf`.
  - Include your answers to each question.
  - By submitting the report, all members acknowledge that they understand all answers and can explain them when asked.
- Submit a zip file consisting of the code used in each programming question:
  - Source codes.
  - A simple `README` with steps to reproduce your answers.

---

## Question 1 — Your First CUDA Program and GPU Performance Metrics

This exercise will get you familiar with the basic structure of a simple CUDA code. You will learn how to compile and run CUDA-based programs. You will practice memory allocation in host memory and device memory, the data movement between CPU and GPU, and the distribution of the computational threads in CUDA. You will also use Nvidia Nsight (see the tutorials) to profile your program.

Please implement a simple `vectorAdd` program that sums two vectors and stores the results into a third vector. The code should also include a CPU version of computation to create a reference result so that GPU results can be validated. You can create your own code, or modify based on the naive code (here).

### Questions to answer in the report

1. For a vector length of `N`, how many floating-point operations are being performed in your vector-add kernel? How many global memory reads are being performed by your kernel?
2. For `N = 512`, how many CUDA threads and thread blocks are launched in your code?
3. For `N = 512`, profile your program with Nvidia Nsight. What Achieved Occupancy did you get?
4. Now increase `N` to `263149`. Does your program still work with correct results? If not, what changes did you make to fix it?
5. For `N = 263149`, how many CUDA threads and thread blocks are launched in your code?
6. For `N = 263149`, profile your program again with Nvidia Nsight. What Achieved Occupancy do you get now?
7. Further increase the vector length (try 10–16 different vector lengths), plot a stacked bar chart showing the breakdown of time including:
   1. data copy from host to device,
   2. the CUDA kernel,
   3. data copy from device to host.

   For this, you will need to add simple CPU timers to your code regions (see tutorial).
8. When comparing the output between CPU and GPU implementation, do you see an exact match or is there a slight error margin? Explain the cause in your own words.

---

## Question 2 — 2D Dense Matrix Multiplication

You will implement a basic matrix multiplication program that multiplies a 2D matrix `A` of `(M, N)` by matrix `B` of `(N, P)` and stores the results into a matrix `C` of `(M, P)`. You should modify your code based on the code in Question 1. Your implementation should use 2D computational grids for the GPU kernel.

### Questions to answer in the report

1. Derive the number of floating-point operations that are performed in your matrix-multiply kernel, as a function of `M`, `N`, `P`. Derive the number of global memory reads performed in your implementation. Explain your answer.
2. For a matrix `A` of `(128 × 256)` and `B` of `(256 × 32)`, explain how many CUDA threads and thread blocks are used in your implementation. Profile your program with Nvidia Nsight. What Achieved Occupancy did you get?
3. For a matrix `A` of `(1024 × 8191)` and `B` of `(8191 × 8197)`, did your program still work? If not, what changes did you make? What Achieved Occupancy do you get now? How many CUDA threads and thread blocks are used, and how many CUDA threads actually perform the computation of the results? Explain your answer.
4. Further increase the size of matrices `A` and `B`, plot a stacked bar chart showing the breakdown of time including:
   1. data copy from host to device,
   2. the CUDA kernel,
   3. data copy from device to host.

   Explain your main observations.
5. Now, change data type from `double` to `float`, re-plot a stacked bar chart showing the time breakdown. Briefly explain why or why not an ideal speedup is observed.

---

## Question 3 — Quantum Fourier Transformation

Quantum computers need to use high-performance simulators to study and co-design quantum algorithms. In this exercise, you will profile a naïve Quantum Fourier Transform (QFT) (`qft_bench_naive.c`) running on the CUDA backend of QuEST v4. In particular, the QFT benchmark uses a simulator of quantum statevectors and you will use Nsight Systems to find hotspots in the runtime.

### Setup: QuEST CUDA backend

1. Query your GPU's compute capability (use `75` on Colab's T4 or `90` on the school cluster):

   ```bash
   nvidia-smi --query-gpu=name,compute_cap
   ```

2. Clone QuEST and copy the native QFT benchmark into it:

   ```bash
   git clone https://github.com/QuEST-Kit/QuEST.git
   cd QuEST

   # Copy the native QFT benchmark here
   cp /path/to/qft_bench_naive.c qft_bench_naive.c
   ```

3. Compile QuEST (replace `YOUR_GPU_COMPUTE_CAP` with the value from step 1):

   ```bash
   cmake -B build -DQUEST_ENABLE_CUDA=ON \
     -DCMAKE_CUDA_ARCHITECTURES=YOUR_GPU_COMPUTE_CAP \
     -DCMAKE_CUDA_FLAGS="-lineinfo" \
     -DUSER_SOURCE_NAMES=qft_bench_naive.c \
     -DUSER_OUTPUT_EXE_NAME=qft_naive

   cmake --build build -j4
   ```

4. Run the QFT benchmark on 24 qubits:

   ```bash
   ./build/qft_naive 24 5
   ```

### Questions to answer in the report

1. Do you need any troubleshooting to get the QFT benchmark running on GPU? Include a screenshot of your run output of the 24-qubit QFT.
2. Run the benchmark again using 28 qubits. How much GPU memory and system memory are used during the execution? Explain how you obtained the memory usage. *Hint: there are multiple ways to get the memory usage information.*
3. Use only Nsight Systems and the unmodified benchmark. Now record a timeline at 24 qubits with 5 timed repetitions:

   ```bash
   nsys profile --trace=cuda,osrt --stats=true -o qft_24q ./build/qft_naive 24 5
   ```

   1. From the `--stats` output, find the `cuda_gpu_kern_sum` table (GPU kernel summary). Report the top 3 kernels by total time, with their instance counts and average durations.
   2. Map each of the 3 top kernels to the quantum gate using the kernel names and `quest/src/gpu/gpu_kernels.cuh`. List the kernel name that implements these quantum gates.
   3. Open `nsys-rep` in the Nsight Systems GUI (you need to install this on your own computer). Zoom into one QFT repetition on the timeline. Find the CUDA API calls on the CPU thread that launch them.
4. Repeat the recording at 16 and 28 qubits. For each, what is the total QFT time printed by the benchmark? What is the typical kernel duration as seen on the timeline? Include two screenshots of your two timelines.
