# iPIC3D-mini Makefile -- DD2360 a2 Bonus
# Upstream pinned version: KTH-ScaLab/DD2360-HT25, commit
# 53e7ca59d1904f586b2940d2cfc3f30351873e7b (iPIC3D-mini/ subfolder).
#
# Targets:
#   make / make gpu    GPU binary            -> bin/miniPIC_gpu.out   (nvcc)
#   make cpu           pure CPU baseline     -> bin/miniPIC_cpu.out   (g++ only)
#   make compare       mover CPU-vs-GPU checker -> bin/mover_compare.out
#   make clean
#
# The mover back end is chosen at COMPILE time (-DUSE_GPU_MOVER), so the two
# binaries can never share stale objects: CPU and GPU objects live in separate
# directories (obj/cpu, obj/gpu) and link to different binaries.
#
# ARCH: override explicitly, e.g.  make ARCH=sm_80
# If ARCH is unset it is auto-detected from the present GPU via nvidia-smi
# (compute capability, e.g. 8.0 -> sm_80); fallback is sm_70 (not the
# upstream default sm_30, which is too old for modern toolchains).

CXX      ?= g++
NVCC     ?= nvcc
STD      := -std=c++11
INCLUDES := -I./include
OPT      := -O3 -g -Xcompiler -Wall

ARCH ?=
ifeq ($(strip $(ARCH)),)
  DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. \r\n')
  ifneq ($(DETECTED_CC),)
    ARCH := sm_$(DETECTED_CC)
  else
    ARCH := sm_70
  endif
endif

NVCCFLAGS := $(INCLUDES) -arch=$(ARCH) $(STD) $(OPT)
CXXFLAGS  := $(INCLUDES) $(STD) -O3 -g -Wall

SRCDIR := src
BIN    := bin

DRIVER      := $(SRCDIR)/sputniPIC.cpp
COMPARE_SRC := $(SRCDIR)/mover_compare.cu
# support sources: everything except the two programs that contain main()
COMMON_SRCS := $(filter-out $(DRIVER) $(COMPARE_SRC),$(sort $(wildcard $(SRCDIR)/*.cpp) $(wildcard $(SRCDIR)/*.cu)))

CPU_OBJDIR := obj/cpu
GPU_OBJDIR := obj/gpu
CMP_OBJDIR := obj/cmp

CPU_COMMON_OBJS := $(foreach s,$(COMMON_SRCS),$(CPU_OBJDIR)/$(basename $(notdir $(s))).o)
GPU_COMMON_OBJS := $(foreach s,$(COMMON_SRCS),$(GPU_OBJDIR)/$(basename $(notdir $(s))).o)

.DEFAULT_GOAL := all
.PHONY: all gpu cpu compare clean
all: gpu compare

gpu:  $(BIN)/miniPIC_gpu.out
cpu:  $(BIN)/miniPIC_cpu.out
compare: $(BIN)/mover_compare.out

$(BIN):
	mkdir -p $(BIN)
$(CPU_OBJDIR):
	mkdir -p $(CPU_OBJDIR)
$(GPU_OBJDIR):
	mkdir -p $(GPU_OBJDIR)
$(CMP_OBJDIR):
	mkdir -p $(CMP_OBJDIR)

# ---- link -------------------------------------------------------------------
$(BIN)/miniPIC_cpu.out: $(CPU_COMMON_OBJS) $(CPU_OBJDIR)/sputniPIC.o | $(BIN)
	$(CXX) $(CXXFLAGS) $+ -o $@

$(BIN)/miniPIC_gpu.out: $(GPU_COMMON_OBJS) $(GPU_OBJDIR)/sputniPIC.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

$(BIN)/mover_compare.out: $(GPU_COMMON_OBJS) $(CMP_OBJDIR)/mover_compare.o | $(BIN)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

# ---- pure CPU build (g++ only, no CUDA toolkit required) --------------------
$(CPU_OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(CPU_OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Particles.cu compiles as plain C++ here: CUDA code is #ifdef __CUDACC__ guarded
$(CPU_OBJDIR)/%.o: $(SRCDIR)/%.cu | $(CPU_OBJDIR)
	$(CXX) $(CXXFLAGS) -x c++ -c $< -o $@

# ---- GPU build (nvcc) --------------------------------------------------------
# all objects carry -DUSE_GPU_MOVER; only the driver acts on it
$(GPU_OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(GPU_OBJDIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_MOVER -c $< -o $@

$(GPU_OBJDIR)/%.o: $(SRCDIR)/%.cu | $(GPU_OBJDIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_MOVER -c $< -o $@

$(CMP_OBJDIR)/mover_compare.o: $(COMPARE_SRC) | $(CMP_OBJDIR)
	$(NVCC) $(NVCCFLAGS) -DUSE_GPU_MOVER -c $< -o $@

clean:
	rm -rf obj $(BIN)
