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

/* Change the original scalar addition to vector addition. */
__global__ void add(float *a, float *b, float *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

double cpuSecond() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    int N = 512;
    if (argc > 1) N = atoi(argv[1]);
    if (N <= 0) return 1;
    int size = N * sizeof(float);

    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);
    float *cpu = (float *)malloc(size);
    for (int i = 0; i < N; i++) {
        a[i] = i * 0.1f;
        b[i] = i * 0.2f;
        cpu[i] = a[i] + b[i];
    }

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **)&d_a, size));
    CHECK(cudaMalloc((void **)&d_b, size));
    CHECK(cudaMalloc((void **)&d_c, size));

    dim3 tpb(32, 1, 1);
    dim3 grid((N + tpb.x - 1) / tpb.x, 1, 1);

    // CPU timer: copy both input vectors to the GPU.
    CHECK(cudaDeviceSynchronize());
    double start = cpuSecond();
    CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    double h2d = (cpuSecond() - start) * 1000;

    // CPU timer: run one kernel and wait until it finishes.
    start = cpuSecond();
    add<<<grid, tpb>>>(d_a, d_b, d_c, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double kernel = (cpuSecond() - start) * 1000;

    // CPU timer: copy the result back to the CPU.
    start = cpuSecond();
    CHECK(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
    double d2h = (cpuSecond() - start) * 1000;

    float max_error = 0;
    for (int i = 0; i < N; i++) {
        float error = fabs(c[i] - cpu[i]);
        if (error > max_error) max_error = error;
    }
    // CSV output for the notebook's plot; max_error = 0 means an exact match.
    printf("N,threads,blocks,H2D_ms,kernel_ms,D2H_ms,max_error\n");
    printf("%d,%u,%u,%.6f,%.6f,%.6f,%.9g\n",
           N, tpb.x, grid.x, h2d, kernel, d2h, max_error);

    // Cleanup, as in the original example.
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    free(a);
    free(b);
    free(c);
    free(cpu);
    return 0;
}
