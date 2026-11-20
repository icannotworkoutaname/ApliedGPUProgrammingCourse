#include "matmul_cuda.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// Used only by the optimized version.
static const float *host_checkpoint;
static float *device_checkpoint;
static float *cached_x;
static float *cached_out;

static bool verify = std::getenv("CHECK_MATMUL") != nullptr;

static void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void matvec_kernel(float *out, const float *x, const float *w, int n, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= d) {
        return;
    }

    float sum = 0.0f;
    for (int col = 0; col < n; col++) {
        sum += w[(size_t)row * n + col] * x[col];
    }
    out[row] = sum;
}

static void launch_matvec(float *out, const float *x, const float *w, int n, int d) {
    int threads = 128;
    int blocks = (d + threads - 1) / threads;
    matvec_kernel<<<blocks, threads>>>(out, x, w, n, d);
    check_cuda(cudaGetLastError());
}

static void compare_with_cpu(const float *out, const float *x, const float *w, int n, int d) {
    if (!verify) {
        return;
    }

    for (int row = 0; row < d; row++) {
        float expected = 0.0f;
        for (int col = 0; col < n; col++) {
            expected += w[(size_t)row * n + col] * x[col];
        }
        float error = std::fabs(out[row] - expected);
        float tolerance = 1e-5f + 1e-5f * std::fabs(expected);
        if (!(error <= tolerance)) {
            std::fprintf(stderr, "matmul row %d: CPU %g, GPU %g\n", row, expected, out[row]);
            std::exit(EXIT_FAILURE);
        }
    }
}

void cuda_matmul_naive(float *out, const float *x, const float *w, int n, int d) {
    size_t input_bytes = n * sizeof(float);
    size_t output_bytes = d * sizeof(float);
    size_t weight_bytes = input_bytes * d;
    float *device_x;
    float *device_w;
    float *device_out;

    check_cuda(cudaMalloc(&device_w, weight_bytes));
    check_cuda(cudaMalloc(&device_x, input_bytes));
    check_cuda(cudaMalloc(&device_out, output_bytes));
    check_cuda(cudaMemcpy(device_w, w, weight_bytes, cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(device_x, x, input_bytes, cudaMemcpyHostToDevice));
    launch_matvec(device_out, device_x, device_w, n, d);
    check_cuda(cudaMemcpy(out, device_out, output_bytes, cudaMemcpyDeviceToHost));
    check_cuda(cudaFree(device_out));
    check_cuda(cudaFree(device_x));
    check_cuda(cudaFree(device_w));
    compare_with_cpu(out, x, w, n, d);
}

void cuda_matmul_init(const void *checkpoint, size_t bytes, int max_n, int max_d) {
    host_checkpoint = static_cast<const float *>(checkpoint);
    check_cuda(cudaMalloc(&device_checkpoint, bytes));
    check_cuda(cudaMalloc(&cached_x, max_n * sizeof(float)));
    check_cuda(cudaMalloc(&cached_out, max_d * sizeof(float)));
    check_cuda(cudaMemcpy(device_checkpoint, checkpoint, bytes, cudaMemcpyHostToDevice));
}

void cuda_matmul_cached(float *out, const float *x, const float *w, int n, int d) {
    // The weights have the same offset in the CPU and GPU checkpoint copies.
    const float *device_w = device_checkpoint + (w - host_checkpoint);
    check_cuda(cudaMemcpy(cached_x, x, n * sizeof(float), cudaMemcpyHostToDevice));
    launch_matvec(cached_out, cached_x, device_w, n, d);
    check_cuda(cudaMemcpy(out, cached_out, d * sizeof(float), cudaMemcpyDeviceToHost));
    compare_with_cpu(out, x, w, n, d);
}

void cuda_matmul_free(void) {
    check_cuda(cudaFree(cached_out));
    check_cuda(cudaFree(cached_x));
    check_cuda(cudaFree(device_checkpoint));
}
