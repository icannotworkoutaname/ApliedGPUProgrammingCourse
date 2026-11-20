#!/usr/bin/env bash
# Full GEM_2D simulation comparison: CPU binary vs GPU binary output files.
#
# Runs bin/miniPIC_cpu.out and bin/miniPIC_gpu.out on inputfiles/GEM_2D.inp,
# keeps the VTK/parameter output of each run in separate directories and diffs
# them. Requires both binaries to be built:
#   make cpu && make gpu
#
# The mover is the only step that differs between the two binaries; the VTK
# field outputs are written every FieldOutputCycle cycles. Small float
# differences from FMA contraction are expected in the interpolated density
# files; field files (E, B are NOT updated in this mini app: only particles
# move and densities are interpolated) should be byte-identical.
set -euo pipefail
cd "$(dirname "$0")/iPIC3D-mini"

INP=${1:-inputfiles/GEM_2D.inp}

for b in cpu gpu; do
    if [ ! -x "bin/miniPIC_${b}.out" ]; then
        echo "missing bin/miniPIC_${b}.out -- run: make ${b}" >&2
        exit 2
    fi
done

rm -rf run_cpu run_gpu
mkdir -p run_cpu/data run_gpu/data

echo ">>> running CPU simulation ($INP)"
( cd run_cpu && ../bin/miniPIC_cpu.out "../$INP" | tee console.log )

echo ">>> running GPU simulation ($INP)"
( cd run_gpu && ../bin/miniPIC_gpu.out "../$INP" | tee console.log )

echo
echo ">>> diff of output files (run_cpu vs run_gpu)"
if diff -r run_cpu/data run_gpu/data > data.diff; then
    echo "data/ outputs are IDENTICAL"
else
    echo "data/ outputs differ -- see data.diff"
    echo "(expected: small float differences in interpolated density files only)"
fi

echo
echo ">>> numerical comparison of every VTK value"
if ! python3 ../compare_vtk.py run_cpu/data run_gpu/data | tee numeric_compare.txt; then
    echo "VTK numerical comparison FAILED" >&2
    exit 1
fi

echo
echo ">>> timing summary from console logs"
grep -h -E "Mover backend|Mover Time / Cycle|Mover Kernel / Cycle|Tot. Simulation Time" \
    run_cpu/console.log run_gpu/console.log || true
