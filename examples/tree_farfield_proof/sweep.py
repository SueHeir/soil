#!/usr/bin/env python3
"""Run the SOIL tree far-field proof and emit the validation plot."""

import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PLOTS = os.path.join(HERE, "plots")


def run(cmd):
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)
    return result.stdout


def parse(output):
    theta_rows = []
    size_rows = []
    max_at_02 = None
    conv_ratio = None
    slope = None
    for line in output.splitlines():
        m = re.search(
            r"theta\s+([0-9.]+): max rel err\s+([0-9.eE+-]+)\s+mean rel err\s+([0-9.eE+-]+)\s+interactions\s+([0-9]+)",
            line,
        )
        if m:
            theta_rows.append(
                (float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4)))
            )
            continue
        m = re.search(
            r"n=\s*([0-9]+): treecode interactions=\s*([0-9]+)\s+direct n\^2/2=\s*([0-9]+)",
            line,
        )
        if m:
            size_rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
            continue
        m = re.search(r"max rel err @theta=0.2 = ([0-9.eE+-]+)", line)
        if m:
            max_at_02 = float(m.group(1))
            continue
        m = re.search(r"mean-err convergence 0.4->0.2 = ([0-9.eE+-]+)x", line)
        if m:
            conv_ratio = float(m.group(1))
            continue
        m = re.search(r"log-log growth exponent = ([0-9.eE+-]+)", line)
        if m:
            slope = float(m.group(1))
    if not theta_rows or not size_rows or max_at_02 is None or conv_ratio is None or slope is None:
        sys.exit("could not parse tree_farfield_proof output")
    return theta_rows, size_rows, max_at_02, conv_ratio, slope


def plot(theta_rows, size_rows, max_at_02, conv_ratio, slope):
    os.makedirs(PLOTS, exist_ok=True)
    thetas = [r[0] for r in theta_rows]
    means = [r[2] for r in theta_rows]
    theta_ref = [means[0] * (t / thetas[0]) ** 2 for t in thetas]
    sizes = [r[0] for r in size_rows]
    tree_counts = [r[1] for r in size_rows]
    direct_counts = [r[2] for r in size_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax = axes[0]
    ax.loglog(thetas, means, marker="o", label="measured mean rel error")
    ax.loglog(thetas, theta_ref, linestyle="--", label="theta^2 reference")
    ax.scatter([0.2], [max_at_02], color="tab:red", label="max error @ theta=0.2")
    ax.axhline(5e-2, color="tab:red", linestyle=":", label="max-error tolerance")
    ax.invert_xaxis()
    ax.set_xlabel("opening angle theta")
    ax.set_ylabel("relative error")
    ax.set_title(f"Accuracy PASS, convergence {conv_ratio:.2f}x")
    ax.legend(fontsize="small")

    ax = axes[1]
    ax.loglog(sizes, tree_counts, marker="o", label=f"tree interactions, slope {slope:.3f}")
    ax.loglog(sizes, direct_counts, marker="s", label="direct n^2/2 reference")
    ax.set_xlabel("particles")
    ax.set_ylabel("interaction count")
    ax.set_title("Scaling PASS: slope < 1.6")
    ax.legend(fontsize="small")

    fig.suptitle("PASS: tree far-field primitive matches direct references")
    out = os.path.join(PLOTS, "tree_farfield_proof.png")
    fig.savefig(out, dpi=160)
    print(f"[plot] wrote {out}")


def main():
    print("[run] cargo run --release --example tree_farfield_proof")
    output = run(["cargo", "run", "--release", "--example", "tree_farfield_proof"])
    print(output)
    if "ALL CHECKS PASSED" not in output:
        sys.exit("missing validation verdict")
    plot(*parse(output))


if __name__ == "__main__":
    main()
