#!/usr/bin/env python3
"""Benchmark driver for the Q2 naive CUDA matMul.

Runs the compiled matMul executable over an increasing size sweep for both
double and float, stores the raw CSV output per type and renders stacked bar
charts of the H2D / kernel / D2H time breakdown.

Requires: matplotlib (provided by `module load scicomp-python-env` on the
remote cluster). Standard library only otherwise; no Colab dependency.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

HEADER = [
    "type", "M", "N", "P", "threads_per_block", "blocks_x", "blocks_y",
    "total_threads", "active_threads", "H2D_ms", "kernel_ms", "D2H_ms",
    "max_abs_error", "max_rel_error", "validation",
]
TYPES = ("double", "float")
# Increasing sweep; starts at the required 128x256x32 shape and reaches
# 2048^3 cubic work. Deliberately excludes the huge 1024x8191x8197 occupancy
# case, which is run separately (see README).
SWEEP = [
    (128, 256, 32),
    (256, 256, 256),
    (384, 384, 384),
    (512, 512, 512),
    (768, 768, 768),
    (1024, 1024, 1024),
    (1280, 1280, 1280),
    (1536, 1536, 1536),
    (2048, 2048, 2048),
]


def run_case(exe, m, n, p, type_name):
    cmd = [str(exe), str(m), str(n), str(p), type_name]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"matMul failed ({' '.join(cmd)}), exit code {proc.returncode}:\n"
            f"{proc.stderr.strip()}"
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) != 2:
        raise RuntimeError(
            f"unexpected stdout from {' '.join(cmd)}: "
            f"expected 2 lines, got {len(lines)}"
        )
    rows = list(csv.reader(lines))
    if rows[0] != HEADER:
        raise RuntimeError(
            f"CSV header mismatch from {' '.join(cmd)}:\n{rows[0]}"
        )
    if len(rows[1]) != len(HEADER):
        raise RuntimeError(
            f"CSV row has {len(rows[1])} fields, expected {len(HEADER)}"
        )
    val = rows[1][14]
    if val != "full" and not val.startswith("sampled:"):
        raise RuntimeError(f"bad validation field: {val!r}")
    return rows[1]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)


def plot(path, rows, type_name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r[1]}x{r[2]}x{r[3]}" for r in rows]
    h2d = [float(r[9]) for r in rows]
    kernel = [float(r[10]) for r in rows]
    d2h = [float(r[11]) for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(labels, h2d, label="H2D copy (A+B)")
    ax.bar(labels, kernel, bottom=h2d, label="kernel")
    ax.bar(labels, d2h, bottom=[a + b for a, b in zip(h2d, kernel)],
           label="D2H copy (C)")
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Matrix shape M x N x P")
    ax.set_title(f"matMul timing breakdown ({type_name}) — 16x16 blocks, "
                 "one thread per C element")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark matMul over an increasing size sweep.")
    parser.add_argument("exe", nargs="?", default="./matMul",
                        help="path to the compiled matMul executable")
    parser.add_argument("--outdir", default="results",
                        help="output directory for CSV/PNG files")
    args = parser.parse_args()

    exe = Path(args.exe).resolve()
    if not exe.exists():
        print(f"error: executable not found: {exe}", file=sys.stderr)
        return 2
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for type_name in TYPES:
        rows = [run_case(exe, m, n, p, type_name) for m, n, p in SWEEP]
        csv_path = outdir / f"timing_{type_name}.csv"
        png_path = outdir / f"timing_{type_name}.png"
        write_csv(csv_path, rows)
        plot(png_path, rows, type_name)
        print(f"{type_name}: wrote {csv_path} and {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
