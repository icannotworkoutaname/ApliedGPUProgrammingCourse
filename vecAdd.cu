#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)

/* CUDA vector addition */
__global__ void add(float *a, float *b, float *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

double cpuSecond() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    int N = 512;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (N <= 0) return 1;
    int size = N * sizeof(float);

    // CPU memory and input data
    float *a, *b, *c, *cpu;
    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);
    cpu = (float *)malloc(size);
    for (int i = 0; i < N; i++) {
        a[i] = i * 0.1f;
        b[i] = i * 0.2f;
    }

    // CPU calculation
    for (int i = 0; i < N; i++) {
        cpu[i] = a[i] + b[i];
    }

    // GPU memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    dim3 tpb(32, 1, 1);
    dim3 grid((N + tpb.x - 1) / tpb.x, 1, 1);

    // Host to device
    CHECK(cudaDeviceSynchronize());
    double start = cpuSecond();
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    CHECK(cudaDeviceSynchronize());
    double h2d = (cpuSecond() - start) * 1000;

    // GPU calculation
    start = cpuSecond();
    add<<<grid, tpb>>>(d_a, d_b, d_c, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double kernel = (cpuSecond() - start) * 1000;

    // Device to host
    start = cpuSecond();
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    CHECK(cudaDeviceSynchronize());
    double d2h = (cpuSecond() - start) * 1000;

    // Compare CPU and GPU results
    float max_error = 0;
    for (int i = 0; i < N; i++) {
        float error = fabs(c[i] - cpu[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    // Output for the notebook
    printf("N,threads,blocks,H2D_ms,kernel_ms,D2H_ms,max_error\n");
    printf("%d,%u,%u,%.6f,%.6f,%.6f,%.9g\n",
           N, tpb.x, grid.x, h2d, kernel, d2h, max_error);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
    free(cpu);
    return 0;
}
