# Build for Q2 — tiled matrix multiplication.
# Adjust ARCH for your GPU (e.g. sm_75 for T4, sm_80 for A100, sm_86 for RTX 30xx,
# sm_89 for RTX 40xx, sm_90 for H100). Default matches the A1 Colab T4 run.

NVCC      ?= nvcc
ARCH      ?= sm_75
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH)

gemm_tiled: gemm_tiled.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f gemm_tiled

.PHONY: clean
