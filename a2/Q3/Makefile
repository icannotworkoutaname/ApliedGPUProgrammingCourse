CC = gcc
NVCC ?= nvcc
CUDA_ARCH ?= native
UPSTREAM = 350e04fe35433e6d2941dce5a1f53308f87058eb
SOURCE_URL = https://raw.githubusercontent.com/karpathy/llama2.c/$(UPSTREAM)

.DELETE_ON_ERROR:
.PHONY: all

all: run run_count run_naive run_optimized

run_upstream.c:
	curl -fsSL --retry 3 -o $@ $(SOURCE_URL)/run.c

run.c: run_upstream.c run.patch
	patch --batch --output=$@ run_upstream.c < run.patch

run: run_upstream.c
	$(CC) -O3 $< -o $@ -lm

run_count: run.c
	$(CC) -O3 -DQ3_COUNT $< -o $@ -lm

matmul_cuda.o: matmul_cuda.cu matmul_cuda.h
	$(NVCC) -O3 --fmad=false -arch=$(CUDA_ARCH) -c $< -o $@

run_naive.o: run.c matmul_cuda.h
	$(CC) -O3 -DQ3_CUDA_NAIVE -c $< -o $@

run_optimized.o: run.c matmul_cuda.h
	$(CC) -O3 -DQ3_CUDA_CACHED -c $< -o $@

run_naive: run_naive.o matmul_cuda.o
	$(NVCC) -o $@ $^ -lm

run_optimized: run_optimized.o matmul_cuda.o
	$(NVCC) -o $@ $^ -lm
