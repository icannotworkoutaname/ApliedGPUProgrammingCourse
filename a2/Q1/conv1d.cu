#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

struct timeval t_start, t_end;
void cputimer_start(){
  gettimeofday(&t_start, 0);
}

void cputimer_stop(const char* info){
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
}

#define MASK_WIDTH 5
#define MASK_RADIUS (MASK_WIDTH / 2)
/* Tile size = threads per block = outputs per block. */
#ifndef TILE_WIDTH
#define TILE_WIDTH 128
#endif
#if TILE_WIDTH < MASK_WIDTH || TILE_WIDTH > 1024
#error "TILE_WIDTH must be in [MASK_WIDTH, 1024]"
#endif
#define REPS 100


/* Basic version: every thread reads its 5 input neighbours and the 5 mask
 * values directly from global memory. Out-of-boundary inputs count as 0. */
__global__ void convolution_1D_basic(float *N, float *M, float *P, int width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;

    float acc = 0.0f;
    int start = i - MASK_RADIUS;
    for (int j = 0; j < MASK_WIDTH; j++) {
        int k = start + j;
        if (k >= 0 && k < width) {
            acc += N[k] * M[j];
        }
    }
    P[i] = acc;
}

/* Tiled version: one block of TILE_WIDTH threads computes TILE_WIDTH outputs.
 * The block loads its tile plus MASK_RADIUS halo elements on each side into
 * shared memory, and the mask once per block, so every input element is read
 * from global memory once per block instead of 5 times. */
__global__ void convolution_1D_tiled(float *N, float *M, float *P, int width)
{
  __shared__ float input_tile[TILE_WIDTH + MASK_WIDTH - 1];
  __shared__ float mask[MASK_WIDTH];

  int t = threadIdx.x;
  int i = blockIdx.x * TILE_WIDTH + t;

  /* mask: loaded by the first MASK_WIDTH threads */
  if (t < MASK_WIDTH) {
    mask[t] = M[t];
  }
  /* left halo: loaded by the last MASK_RADIUS threads */
  if (t >= TILE_WIDTH - MASK_RADIUS) {
    int k = i - TILE_WIDTH;
    input_tile[t - (TILE_WIDTH - MASK_RADIUS)] = (k >= 0) ? N[k] : 0.0f;
  }
  /* centre */
  input_tile[MASK_RADIUS + t] = (i < width) ? N[i] : 0.0f;
  /* right halo: loaded by the first MASK_RADIUS threads */
  if (t < MASK_RADIUS) {
    int k = i + TILE_WIDTH;
    input_tile[MASK_RADIUS + TILE_WIDTH + t] = (k < width) ? N[k] : 0.0f;
  }
  __syncthreads();

  if (i < width) {
    float acc = 0.0f;
    for (int j = 0; j < MASK_WIDTH; j++) {
      acc += input_tile[t + j] * mask[j];
    }
    P[i] = acc;
  }
}

/* CPU reference */
void convolution_1D_cpu(float *N, float *M, float *P, int width)
{
  for (int i = 0; i < width; i++) {
    float acc = 0.0f;
    for (int j = 0; j < MASK_WIDTH; j++) {
      int k = i - MASK_RADIUS + j;
      if (k >= 0 && k < width) acc += N[k] * M[j];
    }
    P[i] = acc;
  }
}

float max_abs_diff(float *a, float *b, int n)
{
  float e = 0.0f;
  for (int i = 0; i < n; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > e) e = d;
  }
  return e;
}

void print_head(const char *name, float *P, int n)
{
  printf("  %s P[0..7]:", name);
  for (int i = 0; i < n && i < 8; i++) printf(" %.4f", P[i]);
  printf("\n");
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    printf("Usage: %s N [carveout%%]\n", argv[0]);
    return 1;
  }
  // Read the arguments from the command line
  int N = atoi(argv[1]);
  int carveout = argc > 2 ? atoi(argv[2]) : -1;   // optional: shared memory carveout for the tiled kernel
  if (N <= 0) {
    printf("N must be positive\n");
    return 1;
  }
  size_t size = (size_t)N * sizeof(float);
  int blocks = (N + TILE_WIDTH - 1) / TILE_WIDTH;
  printf("N = %d, MASK_WIDTH = %d, TILE_WIDTH = %d, blocks = %d\n", N, MASK_WIDTH, TILE_WIDTH, blocks);

  float *hostN; // The input array N of length N
  float *hostM; // The 1D mask M of length MASK_WIDTH
  float *hostP; // The output array P of length N
  float *hostP_tiled, *hostP_ref;

  cputimer_start();
  //@@ Allocate the host memory
  hostN = (float *)malloc(size);
  hostM = (float *)malloc(MASK_WIDTH * sizeof(float));
  hostP = (float *)malloc(size);
  hostP_tiled = (float *)malloc(size);
  hostP_ref = (float *)malloc(size);
  cputimer_stop("Allocated host memory");


  float *deviceN;
  float *deviceM;
  float *deviceP;

  cputimer_start();
  //@@ Allocate the device memory
  gpuCheck(cudaMalloc(&deviceN, size));
  gpuCheck(cudaMalloc(&deviceM, MASK_WIDTH * sizeof(float)));
  gpuCheck(cudaMalloc(&deviceP, size));
  cputimer_stop("Allocated device memory");


  cputimer_start();
  //@@ Initialize N with random values
  //@@ Initialize M with [-0.25, 0.5, 1.0, 0.5, 0.25]
  //@@ Initialize P with 0.0
  srand(2360);
  for (int i = 0; i < N; i++) hostN[i] = (float)rand() / RAND_MAX;
  float mask_init[MASK_WIDTH] = {-0.25f, 0.5f, 1.0f, 0.5f, 0.25f};
  for (int j = 0; j < MASK_WIDTH; j++) hostM[j] = mask_init[j];
  for (int i = 0; i < N; i++) {
    hostP[i] = 0.0f;
    hostP_tiled[i] = 0.0f;
  }
  cputimer_stop("Initialized host data");


  cputimer_start();
  //@@ INSERT CODE HERE
  gpuCheck(cudaMemcpy(deviceN, hostN, size, cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(deviceM, hostM, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(deviceP, hostP, size, cudaMemcpyHostToDevice));
  cputimer_stop("Copying data to the GPU.");

  //@@  Define the execution configuration
  dim3 block(TILE_WIDTH);
  dim3 grid(blocks);
  if (carveout >= 0) {
    gpuCheck(cudaFuncSetAttribute(convolution_1D_tiled,
                                  cudaFuncAttributePreferredSharedMemoryCarveout, carveout));
    printf("Shared memory carveout for the tiled kernel: %d%%\n", carveout);
  }

  /* Warm-up: one launch of each kernel, so the timed launches below do not
   * include one-time module loading. */
  convolution_1D_basic<<<grid, block>>>(deviceN, deviceM, deviceP, N);
  convolution_1D_tiled<<<grid, block>>>(deviceN, deviceM, deviceP, N);
  gpuCheck(cudaGetLastError());
  gpuCheck(cudaDeviceSynchronize());
  gpuCheck(cudaMemset(deviceP, 0, size));

  /* Call the basic kernel */
  cputimer_start();
  //@@  Run the 1D convolution kernel (basic)
  convolution_1D_basic<<<grid, block>>>(deviceN, deviceM, deviceP, N);
  gpuCheck(cudaGetLastError());
  gpuCheck(cudaDeviceSynchronize());
  cputimer_stop("Finished 1D convolution(basic)");

  cputimer_start();
  //@@ INSERT CODE HERE
  gpuCheck(cudaMemcpy(hostP, deviceP, size, cudaMemcpyDeviceToHost));
  print_head("basic", hostP, N);
  cputimer_stop("Copying output P to the CPU and print out the results");

  /* Call the tiled kernel */
  gpuCheck(cudaMemset(deviceP, 0, size));
  cputimer_start();
  //@@  Run the 1D convolution kernel (tiled)
  convolution_1D_tiled<<<grid, block>>>(deviceN, deviceM, deviceP, N);
  gpuCheck(cudaGetLastError());
  gpuCheck(cudaDeviceSynchronize());
  cputimer_stop("Finished 1D convolution(tiled)");

  cputimer_start();
  //@@ INSERT CODE HERE
  gpuCheck(cudaMemcpy(hostP_tiled, deviceP, size, cudaMemcpyDeviceToHost));
  print_head("tiled", hostP_tiled, N);
  cputimer_stop("Copying output P to the CPU and print out the results");


  //@@ Validate the results from the two implementations
  convolution_1D_cpu(hostN, hostM, hostP_ref, N);
  float diff_bt = max_abs_diff(hostP, hostP_tiled, N);
  float err_b = max_abs_diff(hostP, hostP_ref, N);
  float err_t = max_abs_diff(hostP_tiled, hostP_ref, N);
  int ok = diff_bt < 1e-5f && err_b < 1e-5f && err_t < 1e-5f;
  printf("Validation: max|basic-tiled| = %.3e, max|basic-CPU| = %.3e, max|tiled-CPU| = %.3e -> %s\n",
         diff_bt, err_b, err_t, ok ? "PASS" : "FAIL");

  /* Precise kernel times: CUDA events, average over REPS launches.
   * (The single-launch CPU timers above have 1 us resolution and include launch overhead.) */
  cudaEvent_t ev0, ev1;
  float ms_basic, ms_tiled;
  gpuCheck(cudaEventCreate(&ev0));
  gpuCheck(cudaEventCreate(&ev1));
  gpuCheck(cudaEventRecord(ev0));
  for (int r = 0; r < REPS; r++) convolution_1D_basic<<<grid, block>>>(deviceN, deviceM, deviceP, N);
  gpuCheck(cudaEventRecord(ev1));
  gpuCheck(cudaEventSynchronize(ev1));
  gpuCheck(cudaEventElapsedTime(&ms_basic, ev0, ev1));
  gpuCheck(cudaEventRecord(ev0));
  for (int r = 0; r < REPS; r++) convolution_1D_tiled<<<grid, block>>>(deviceN, deviceM, deviceP, N);
  gpuCheck(cudaEventRecord(ev1));
  gpuCheck(cudaEventSynchronize(ev1));
  gpuCheck(cudaEventElapsedTime(&ms_tiled, ev0, ev1));
  printf("Kernel time (CUDA events, avg of %d launches): basic %.3f us, tiled %.3f us\n",
         REPS, ms_basic * 1000.0f / REPS, ms_tiled * 1000.0f / REPS);
  gpuCheck(cudaEventDestroy(ev0));
  gpuCheck(cudaEventDestroy(ev1));


  cputimer_start();
  //@@ INSERT CODE HERE
  gpuCheck(cudaFree(deviceN));
  gpuCheck(cudaFree(deviceM));
  gpuCheck(cudaFree(deviceP));
  free(hostN);
  free(hostM);
  free(hostP);
  free(hostP_tiled);
  free(hostP_ref);
  cputimer_stop("Free memory resources");

  return ok ? 0 : 1;
}
