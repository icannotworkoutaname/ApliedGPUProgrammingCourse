#!/usr/bin/env python3
"""Compare every value in two legacy ASCII VTK output directories."""

import argparse
import math
import sys
from pathlib import Path


def read_vtk(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("VECTORS "):
            header_end = i + 1
            break
        if line.startswith("SCALARS "):
            if i + 1 >= len(lines) or not lines[i + 1].startswith("LOOKUP_TABLE "):
                raise ValueError(f"{path}: SCALARS has no LOOKUP_TABLE")
            header_end = i + 2
            break
    else:
        raise ValueError(f"{path}: no SCALARS or VECTORS payload")

    values = [float(token) for line in lines[header_end:]
              for token in line.split()]
    return lines[:header_end], values


def compare_file(cpu_path, gpu_path, atol, rtol):
    cpu_header, cpu = read_vtk(cpu_path)
    gpu_header, gpu = read_vtk(gpu_path)
    if cpu_header != gpu_header:
        raise ValueError(f"{cpu_path}: CPU/GPU VTK headers differ")
    if len(cpu) != len(gpu):
        raise ValueError(f"{cpu_path}: value counts differ")

    max_abs = 0.0
    max_rel = 0.0
    passed = True
    for a, b in zip(cpu, gpu):
        if not math.isfinite(a) or not math.isfinite(b):
            max_abs = max_rel = math.inf
            passed = False
            continue
        error = abs(a - b)
        relative = error / max(abs(a), sys.float_info.min)
        max_abs = max(max_abs, error)
        max_rel = max(max_rel, relative)
        if error > atol + rtol * abs(a):
            passed = False
    return len(cpu), max_abs, max_rel, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_dir", type=Path)
    parser.add_argument("gpu_dir", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    cpu_files = {p.relative_to(args.cpu_dir): p
                 for p in args.cpu_dir.rglob("*.vtk")}
    gpu_files = {p.relative_to(args.gpu_dir): p
                 for p in args.gpu_dir.rglob("*.vtk")}
    if not cpu_files:
        print(f"error: no VTK files in {args.cpu_dir}", file=sys.stderr)
        return 2
    if cpu_files.keys() != gpu_files.keys():
        print("error: CPU/GPU VTK file sets differ", file=sys.stderr)
        return 1

    all_passed = True
    identical = 0
    try:
        for relative in sorted(cpu_files):
            cpu_path = cpu_files[relative]
            gpu_path = gpu_files[relative]
            if cpu_path.read_bytes() == gpu_path.read_bytes():
                identical += 1
            count, max_abs, max_rel, passed = compare_file(
                cpu_path, gpu_path, args.atol, args.rtol)
            all_passed &= passed
            print(f"{relative}: values={count} max_abs_err={max_abs:.9g} "
                  f"max_rel_err={max_rel:.9g} {'PASS' if passed else 'FAIL'}")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"VTK files: {len(cpu_files)}; byte-identical: {identical}")
    print(f"RESULT: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
