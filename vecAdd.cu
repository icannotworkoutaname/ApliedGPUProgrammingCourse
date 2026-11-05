#include <stdio.h>
#include <stdlib.h>
#include <cstdio>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)

/* Our over-simplified CUDA kernel */
__global__ void add(int *a, int *b, int *c) {
     c[0] = a[0] + b[0];
}

int main() {

    int a=11, b=22, c=0;

    int *d_a, *d_b, *d_c;
    int size = sizeof(int);
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    printf("result before GPU computation is %d\n",c);

    dim3 grid(1,1,1);
    dim3 tpb(32,1,1);
    add<<<grid,tpb>>>(d_a, d_b, d_c);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    printf("result after GPU computation is %d\n",c);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
