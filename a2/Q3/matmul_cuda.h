#ifndef MATMUL_CUDA_H
#define MATMUL_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_matmul_naive(float *out, const float *x, const float *w, int n, int d);
void cuda_matmul_init(const void *checkpoint, size_t bytes, int max_n, int max_d);
void cuda_matmul_cached(float *out, const float *x, const float *w, int n, int d);
void cuda_matmul_free(void);

#ifdef __cplusplus
}
#endif

#endif
