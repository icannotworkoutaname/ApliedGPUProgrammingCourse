## Question 1 - 1D Convolution

In the lecture, you have learned convolution in its basic implementation and tiled optimization. This exercise will develop both implementations and compare their performance. You will leverage tiling and the shared memory in the optimized version to reduce global memory accesses. The input array N has (N) floats, where N is given by the user. The mask M has a fixed size of 5. The output array P also has (N) floats. You can assume out-of-boundary elements to have value 0. You can modify the code skeleton hereDownload code skeleton here.

### Questions to answer in the report

1. Name 3 application domains where convolutions are used?
2. How many global memory reads and writes are performed in the basic implementation and the tiled implementation, respectively?
3. Run with your program with N of 1024, 2048, 4096, 8192, 16384. Include the screenshot of your output. Explain how you choose the optimal tile size to use.
4. Profile your tiled implementation with Nsight Systems and Nsight Compute. What is the performance hotspot now? How is shared memory utilization and occupancy? When you adjust the shared memory capacity, does it help the performance?

## Question 2 - Tiled Matrix Multiplication

In Assignment 1, you have implemented a basic version of 2D Dense Matrix Multiplication. This exercise will develop a tiled version based on your previous implementation. Now, keep most of your basic implementation and add a new CUDA kernel called `tiled_gemm()`. It accepts two additional parameters, namely: `tileX` and `tileY`, for determining the tile size.

For each input, your code will invoke the `gemm()` kernel first and then invoke the `tiled_gemm()` kernel for multiple times to sweep different tile sizes. Your code should compare the results from the CUDA kernels with the reference result created from the CPU implementation to ensure correctness. Finally, time the gemm kernels and report their timing. Your program output will look like the following

### Questions to answer in the report

1. How many global memory reads are performed by the `tiled_gemm()` kernel? Compare it with that of the `gemm()` version.
2. Run with your program with a matrix A of (1024x1024) and B of (1024x1024), select 3 tile sizes for testing. Include the screenshot of your output. Explain the results.
3. Run with your program with a matrix A of (513x8192) and B of (8192x1023), select 3 tile sizes for testing. Include the screenshot of your output. Explain the results.
4. Profile your program in 2 and 3. Report Achieved Occupancy and Shared Memory usage.
5. Run your program with 8-10 different matrix sizes that increases from small size to larger sizes. Plot a bar chart comparing the runtime of the CPU version, the `gemm()`, and the three tiled `gemm()` kernels of these problems. Does the profiling result reveal anything unexpected?

## Question 3 – Accelerate Llama2 Inference

In this exercise, you will accelerate a CPU implementation of LLM inference at `github.com/karpathy/llama2.c` provided by Andrej Karpathy. You will use `run.c` (FP32) as the main engine to generate tokens using a small 15M parameter Llama 2 model. Follow README to compile and run `./run stories15M.bin`, it will use CPU backend. A hotspot of its CPU version is identified to be the matmul function. In this exercise, you will port it onto GPU and profile the code.

### Questions to answer in the report

1. What is the output “achieved tok/s” from running the original CPU backend. Include a screenshot. How many times is matmul triggered in one run?
2. Now port `void matmul (float* xout, float* x, float* w, int n, int d)` to GPU. You can start with a naïve implementation that include memory allocation and data transfer inside each matmul. This is not the most efficient design as the data movement cost will be high, but it is good as a starting point that ensures correctness.
3. Include your modified codes and Makefile in the submission (you are free to add new files). How did you verify the correctness of your implementation?
4. What is the output “achieved tok/s” now using the GPU-ported version. Include a screenshot.
5. Profile the code using Nsight Systems and identify the performance bottlenecks.
6. Implement one optimization to address any one of those identified bottlenecks. Briefly describe your optimization. What is the output “achieved tok/s”. Include a screenshot.
7. Profile your optimized version and compare it with the naive version. What profiling metrics have changed?

## Bonus (Optional) - A Particle Simulation Application

This exercise will look into a simplified version of a particle-in-cell simulation code and focus on its particle mover that calculates and updates the position of particles. You can access the code at `https://github.com/KTH-ScaLab/DD2360-HT25` -> folder `iPIC3D-mini`. The particle mover is contained in `/src/Particles.cu` file, which contains two main computation functions: `mover_PC()` and `interpP2G()` and two main memory management functions: `particle_allocate()` and `particle_deallocate()`.

In this exercise, you will compile and run the current version of the code, which still runs on CPU, and get familiar with the `particle mover_PC()` function in `Particles.cu`. Some background information about the code and how to run a simple simulation can be found in `Introduction of spunitGPU.pdf`. For this exercise, use the `GEM_2D.inp` input file.

You will then implement a GPU version of `mover_PC()` called `mover_PC_gpu()`. For this exercise, it is ok to include GPU memory allocation routines inside `mover_PC_gpu()` so that you allocate GPU memory for particles, copy particles to GPU memory, invoke computation, and deallocate GPU memory. Of course, you can create a more optimized implementation that will modify a larger scope of the code. Finish the following steps:

### Questions to answer in the report

1. Describe the environment you used, what changes you made to the Makefile, and how you ran the simulation.
2. Describe your design of the GPU implementation of `mover_PC()` briefly.
3. Compare the output of both CPU and GPU implementation to guarantee that your GPU implementations produce correct answers.
4. Compare the execution time of your GPU implementation with its CPU version.
