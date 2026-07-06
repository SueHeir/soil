#!/usr/bin/env python3
"""Validate that too-far exchange migration fails hard in release builds."""

import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PLOTS = os.path.join(HERE, "plots")
BIN = os.path.join(ROOT, "target", "release", "examples", "exchange_multihop_gate")
CONFIG = os.path.join(HERE, "config.toml")
NEEDLE = "SOIL exchange left"


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def write_plot(passed):
    os.makedirs(PLOTS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
    ax.bar(
        ["release MPI run"],
        [1 if passed else 0],
        color=["tab:green" if passed else "tab:red"],
    )
    ax.axhline(1, color="black", linestyle="--", linewidth=1, label="hard error observed")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("gate result")
    ax.set_title("PASS: multi-hop exchange stops before mis-binning" if passed else "FAIL")
    ax.legend(loc="upper right")
    out = os.path.join(PLOTS, "exchange_multihop_gate.png")
    fig.savefig(out, dpi=160)
    print(f"[plot] wrote {out}")


def main():
    print("[build] cargo build --release --example exchange_multihop_gate")
    build = run(["cargo", "build", "--release", "--example", "exchange_multihop_gate"])
    if build.returncode != 0:
        print(build.stdout)
        print(build.stderr)
        sys.exit("build failed")

    cmd = ["mpiexec", "-n", "4", BIN, CONFIG]
    print("[run] " + " ".join(cmd))
    result = run(cmd)
    combined = result.stdout + result.stderr
    passed = result.returncode != 0 and NEEDLE in combined
    write_plot(passed)

    print(f"  [{'PASS' if passed else 'FAIL'}] release run returns hard actionable error")
    print(f"      exit_code={result.returncode} contains_message={NEEDLE in combined}")
    if not passed:
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
