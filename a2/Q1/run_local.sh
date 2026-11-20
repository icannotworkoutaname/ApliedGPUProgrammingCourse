#!/usr/bin/env bash
# Build, run and profile Q1 on a local Linux / WSL2 machine with an NVIDIA GPU.
# Usage: bash run_local.sh            (env overrides: CUDA_BIN, ARCH, OUT, WORK)
set -uo pipefail

CUDA_BIN=${CUDA_BIN:-/usr/local/cuda-12.6/bin}
ARCH=${ARCH:-sm_86}                      # RTX 3060 Laptop = sm_86, Tesla T4 = sm_75
NVCC=$CUDA_BIN/nvcc
NCU=$CUDA_BIN/ncu
# nsys 2024.5 (bundled with CUDA 12.6) records no GPU kernels under WSL2 with driver 12.7;
# use a newer Nsight Systems Linux CLI when available (copied from the Windows install's target-linux-x64).
NSYS=${NSYS:-$HOME/nsys2026/target-linux-x64/nsys}
[ -x "$NSYS" ] || NSYS=$CUDA_BIN/nsys
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/results}
WORK=${WORK:-$HOME/hw2q1}                # build on the Linux filesystem (faster than /mnt)
SIZES="1024 2048 4096 8192 16384"
TILES="32 64 128 256 512 1024"
PROFILE_TILE=128
PROFILE_N=16384

mkdir -p "$WORK" "$OUT"
cp "$HERE/conv1d.cu" "$WORK/"
cd "$WORK" || exit 1

echo "### environment"
{
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
  "$NVCC" --version | tail -2
  "$NSYS" --version
  "$NCU" --version | tail -1
} | tee "$OUT/environment.txt"

echo "### build (TILE_WIDTH is a compile-time constant)"
for t in $TILES; do
  "$NVCC" -O3 -arch="$ARCH" -lineinfo -DTILE_WIDTH="$t" conv1d.cu -o "conv1d_T$t" || exit 1
done

echo "### Q1.3 runs with TILE_WIDTH = $PROFILE_TILE"
for n in $SIZES; do
  "./conv1d_T$PROFILE_TILE" "$n" | tee "$OUT/run_$n.txt"
  echo
done

echo "### Q1.3 tile sweep (two repetitions to show run-to-run noise)"
{
  echo "rep N TILE basic_us tiled_us"
  for rep in 1 2; do
    for n in $SIZES; do
      for t in $TILES; do
        "./conv1d_T$t" "$n" | awk -v r="$rep" -v n="$n" -v t="$t" '/^Kernel time/ {print r, n, t, $(NF-4), $(NF-1)}'
      done
    done
  done
} | column -t | tee "$OUT/sweep.txt"

echo "### Q1.3 tile sweep with GPU-side kernel durations from Nsight Systems (median of 102 launches, ns)"
{
  echo "N TILE basic_med_ns tiled_med_ns"
  for n in $SIZES; do
    for t in $TILES; do
      "$NSYS" profile --trace=cuda --force-overwrite true -o "$WORK/sweep_tmp" "./conv1d_T$t" "$n" > /dev/null 2>&1
      "$NSYS" stats --force-export=true --report cuda_gpu_kern_sum --format csv "$WORK/sweep_tmp.nsys-rep" 2>/dev/null \
        | awk -F, -v n="$n" -v t="$t" '{gsub(/"/, "")} /convolution_1D_basic/ {b=$5} /convolution_1D_tiled/ {k=$5} END {print n, t, b, k}'
    done
  done
} | column -t | tee "$OUT/sweep_nsys.txt"

echo "### Q1.4 Nsight Systems"
"$NSYS" profile --force-overwrite true -o "$OUT/q1_nsys" "./conv1d_T$PROFILE_TILE" "$PROFILE_N" > /dev/null
"$NSYS" stats --force-export=true --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  "$OUT/q1_nsys.nsys-rep" | tee "$OUT/nsys_stats.txt"

echo "### Q1.4 Nsight Compute (launch-skip 1 skips the warm-up launch)"
SECTIONS="--section SpeedOfLight --section Occupancy --section LaunchStats --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats"
for k in convolution_1D_tiled convolution_1D_basic; do
  "$NCU" --kernel-name "$k" --launch-skip 1 --launch-count 1 $SECTIONS \
    "./conv1d_T$PROFILE_TILE" "$PROFILE_N" 2>&1 | tee "$OUT/ncu_$k.txt"
done
"$NCU" --kernel-name convolution_1D_tiled --launch-skip 1 --launch-count 1 --set full -f \
  -o "$OUT/q1_tiled_full" "./conv1d_T$PROFILE_TILE" "$PROFILE_N" > /dev/null

echo "### Q1.4 shared memory carveout"
KEYS="Shared Memory Configuration Size|Achieved Occupancy|Theoretical Occupancy|Block Limit Shared Mem|Duration"
for c in 0 25 50 100; do
  echo "== carveout $c%"
  "./conv1d_T$PROFILE_TILE" "$PROFILE_N" "$c" | grep -E "Kernel time|Validation"
  "$NCU" --kernel-name convolution_1D_tiled --launch-skip 1 --launch-count 1 \
    --section SpeedOfLight --section Occupancy --section LaunchStats \
    "./conv1d_T$PROFILE_TILE" "$PROFILE_N" "$c" 2>&1 | grep -E "$KEYS"
done | tee "$OUT/carveout.txt"

echo "### done, results in $OUT"
