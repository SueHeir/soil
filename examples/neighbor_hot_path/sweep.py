#!/usr/bin/env python3
"""Run the SOIL neighbor distance-kernel validation and plot timings."""

import csv
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BIN = os.path.join(ROOT, "target", "release", "examples", "neighbor_hot_path")
CFG = os.path.join(HERE, "config.toml")
DATA = os.path.join(HERE, "data")
PLOTS = os.path.join(HERE, "plots")
CSV = os.path.join(DATA, "neighbor_hot_path.csv")
PLOT = os.path.join(PLOTS, "neighbor_hot_path.png")

ROW_RE = re.compile(r"^SIMD_DISTANCE (?P<body>.*)$")


def run(cmd):
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)
    return result.stdout


def parse_rows(stdout):
    rows = []
    for line in stdout.splitlines():
        match = ROW_RE.match(line)
        if not match:
            continue
        row = {}
        for item in match.group("body").split():
            key, value = item.split("=", 1)
            row[key] = value
        rows.append(row)
    if not rows:
        sys.exit("no SIMD_DISTANCE rows parsed")
    for row in rows:
        if row.get("pairs_match") != "true":
            sys.exit(f"pair mismatch in {row.get('case')}")
    return rows


def write_csv(rows):
    os.makedirs(DATA, exist_ok=True)
    fields = [
        "case",
        "layout",
        "natoms",
        "pairs",
        "simd_available",
        "pairs_match",
        "scalar_ns_per_build",
        "simd_ns_per_build",
        "speedup",
    ]
    with open(CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[data] {os.path.relpath(CSV, ROOT)}")


def plot(rows):
    os.makedirs(PLOTS, exist_ok=True)
    labels = [row["layout"] for row in rows]
    scalar = [float(row["scalar_ns_per_build"]) / 1.0e6 for row in rows]
    simd = [float(row["simd_ns_per_build"]) / 1.0e6 for row in rows]
    speedups = [float(row["speedup"]) for row in rows]

    x = list(range(len(rows)))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), constrained_layout=True)

    ax = axes[0]
    ax.bar([v - width / 2 for v in x], scalar, width, label="scalar")
    ax.bar([v + width / 2 for v in x], simd, width, label="simd requested")
    ax.set_ylabel("ns/build (millions)")
    ax.set_xticks(x, labels)
    ax.set_title("Neighbor build timing")
    ax.legend()

    ax = axes[1]
    ax.axhline(1.0, color="0.3", linewidth=1)
    ax.bar(x, speedups, width=0.5, color="#4c78a8")
    ax.set_ylabel("scalar / simd")
    ax.set_xticks(x, labels)
    ax.set_title("PASS: SIMD-requested pairs match scalar pairs in every case")

    for idx, row in enumerate(rows):
        axes[0].text(idx, max(scalar[idx], simd[idx]) * 1.02, f"{row['pairs']} pairs", ha="center", fontsize=8)
        axes[1].text(idx, speedups[idx] * 1.02, f"{speedups[idx]:.2f}x", ha="center", fontsize=8)

    fig.suptitle("SOIL neighbor distance kernel")
    fig.savefig(PLOT, dpi=170)
    print(f"[plot] {os.path.relpath(PLOT, ROOT)}")


def main():
    print("[build] cargo build --release --example neighbor_hot_path")
    run(["cargo", "build", "--release", "--example", "neighbor_hot_path"])
    print(f"[run] {os.path.relpath(BIN, ROOT)} {os.path.relpath(CFG, ROOT)}")
    out = run([BIN, CFG])
    print(out)
    rows = parse_rows(out)
    write_csv(rows)
    plot(rows)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
