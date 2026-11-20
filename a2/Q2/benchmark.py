#!/usr/bin/env python3
"""Benchmark driver for Q2 — tiled matrix multiplication.

Runs the compiled `gemm_tiled` executable over an increasing size sweep,
stores the raw per-invocation CSV rows (cpu, naive gemm, tiled 8x8 / 16x16 /
32x32) in long format, a pivoted wide table, and renders a grouped bar chart
comparing CPU, naive gemm and the three tiled kernels per matrix shape.

Requires: matplotlib. Standard library only otherwise. Any failed run,
missing kernel row, or validation failure aborts the benchmark with a clear
error instead of writing empty or partial results.

Usage:
    python3 benchmark.py ./gemm_tiled --outdir results
    python3 benchmark.py ./gemm_tiled --type float --sizes 256 512 1024
"""

import argparse
import csv
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

# Long-format CSV header emitted by gemm_tiled.cu.
HEADER = [
    "type", "M", "K", "N", "kernel", "tile_x", "tile_y", "threads_per_block",
    "blocks_x", "blocks_y", "total_threads", "active_threads",
    "smem_per_block_B", "time_ms", "max_abs_error", "max_rel_error",
    "validation",
]

# One bar per kernel, in chart order.
SERIES = ["cpu", "gemm", "tiled_8x8", "tiled_16x16", "tiled_32x32"]
SERIES_LABELS = {
    "cpu": "CPU reference",
    "gemm": "gemm (naive 16x16)",
    "tiled_8x8": "tiled 8x8",
    "tiled_16x16": "tiled 16x16",
    "tiled_32x32": "tiled 32x32",
}

# Increasing sweep from the required small shape up to 2048^3, ordered by
# total work M*K*N (513*8192*1023 ~= 4.3e9 sits between 1536^3 and 2048^3);
# the non-divisible 513x8192x1023 case from report question 3 is included.
DEFAULT_SWEEP = [
    (128, 256, 32),
    (256, 256, 256),
    (384, 384, 384),
    (512, 512, 512),
    (768, 768, 768),
    (1024, 1024, 1024),
    (1280, 1280, 1280),
    (1536, 1536, 1536),
    (513, 8192, 1023),
    (2048, 2048, 2048),
]


def run_case(exe, m, k, n, type_name):
    cmd = [str(exe), str(m), str(k), str(n), type_name]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"gemm_tiled failed ({' '.join(cmd)}), exit code "
            f"{proc.returncode}:\n{proc.stderr.strip()}"
        )
    rows = []
    for ln in proc.stdout.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith("type,"):
            continue
        fields = next(csv.reader([ln]))
        if len(fields) != len(HEADER):
            raise RuntimeError(
                f"CSV row from {' '.join(cmd)} has {len(fields)} fields, "
                f"expected {len(HEADER)}: {ln}"
            )
        rows.append(dict(zip(HEADER, fields)))
    if len(rows) != 5:
        raise RuntimeError(
            f"expected 5 timing rows (cpu + gemm + 3 tiled) from "
            f"{' '.join(cmd)}, got {len(rows)}"
        )
    for row in rows:
        if row["validation"] != "full":
            raise RuntimeError(f"bad validation field: {row['validation']!r}")
        if not row["time_ms"]:
            raise RuntimeError(f"empty time_ms in row: {row}")
    return rows


def series_key(row):
    if row["kernel"] == "cpu":
        return "cpu"
    if row["kernel"] == "gemm":
        return "gemm"
    return f"tiled_{row['tile_x']}x{row['tile_y']}"


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")


def write_wide_csv(path, per_case):
    cols = ["type", "M", "K", "N"] + [f"{s}_ms" for s in SERIES]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for (type_name, m, k, n), times in per_case.items():
            writer.writerow([type_name, m, k, n] +
                            [f"{times[s]:.4f}" for s in SERIES])


def plot(path, per_case, type_name, log_scale):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{m}x{k}x{n}" for (_, m, k, n) in per_case.keys()]
    n_groups = len(labels)
    n_series = len(SERIES)
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = range(n_groups)
    for i, s in enumerate(SERIES):
        heights = [per_case[key][s] for key in per_case]
        ax.bar([xi + i * width for xi in x], heights, width=width,
               label=SERIES_LABELS[s])
    ax.set_yscale("log" if log_scale else "linear")
    ax.set_ylabel("Time (ms, log scale)" if log_scale else "Time (ms)")
    ax.set_xlabel("Matrix shape M x K x N")
    ax.set_title(
        f"Q2 tiled GEMM — CPU vs naive vs tiled ({type_name}), "
        f"tile sweep 8x8 / 16x16 / 32x32")
    ax.set_xticks([xi + width * (n_series - 1) / 2 for xi in x])
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark gemm_tiled over an increasing size sweep.")
    parser.add_argument("exe", nargs="?", default="./gemm_tiled",
                        help="path to the compiled gemm_tiled executable")
    parser.add_argument("--outdir", default="results",
                        help="output directory for CSV/PNG files")
    parser.add_argument("--type", default="double", choices=["double", "float"],
                        help="precision passed to gemm_tiled (default double)")
    parser.add_argument("--sizes", nargs=3, type=int, action="append",
                        metavar=("M", "K", "N"),
                        help="one M K N shape; repeat to override the sweep")
    parser.add_argument("--linear", action="store_true",
                        help="use a linear y-axis (default: log scale)")
    args = parser.parse_args()

    exe = Path(args.exe).resolve()
    if not exe.exists():
        print(f"error: executable not found: {exe}", file=sys.stderr)
        return 2
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sweep = args.sizes if args.sizes else DEFAULT_SWEEP

    all_rows = []
    per_case = OrderedDict()
    for m, k, n in sweep:
        rows = run_case(exe, m, k, n, args.type)
        all_rows.extend(rows)
        key = (args.type, m, k, n)
        times = {}
        for row in rows:
            s = series_key(row)
            if s in times:
                raise RuntimeError(f"duplicate series {s!r} for case {key}")
            times[s] = float(row["time_ms"])
        missing = [s for s in SERIES if s not in times]
        if missing:
            raise RuntimeError(f"case {key} missing series: {missing}")
        per_case[key] = times
        print(f"{args.type} {m}x{k}x{n}: cpu={times['cpu']:.2f} ms, "
              f"gemm={times['gemm']:.3f} ms, "
              f"tiled16={times['tiled_16x16']:.3f} ms")

    csv_path = outdir / f"timing_{args.type}.csv"
    wide_path = outdir / f"timing_{args.type}_wide.csv"
    png_path = outdir / f"timing_{args.type}.png"
    write_csv(csv_path, [[r[c] for c in HEADER] for r in all_rows])
    write_wide_csv(wide_path, per_case)
    plot(png_path, per_case, args.type, log_scale=not args.linear)
    print(f"wrote {csv_path}")
    print(f"wrote {wide_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
