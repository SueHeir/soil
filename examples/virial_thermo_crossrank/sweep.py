#!/usr/bin/env python3
"""
SOIL virial thermo cross-rank validation.

Runs the same declarative pair-force virial case on 1, 2, and 4 MPI ranks,
parses the rank-0 thermo columns virial_xx..virial_yz, and checks them against
the analytic tensor in data/reference.csv.
"""

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
PLOTS = os.path.join(HERE, "plots")
BIN = os.path.join(ROOT, "target", "release", "examples", "virial_thermo_crossrank")
LAYOUTS = [(1, "cfg_n1.toml"), (2, "cfg_n2.toml"), (4, "cfg_n4.toml")]
COMPONENTS = ["virial_xx", "virial_yy", "virial_zz", "virial_xy", "virial_xz", "virial_yz"]
TOL = 1.0e-12


def sh(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def build():
    print("[build] cargo build --release --example virial_thermo_crossrank")
    result = sh(["cargo", "build", "--release", "--example", "virial_thermo_crossrank"])
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit("build failed")


def reference():
    path = os.path.join(HERE, "data", "reference.csv")
    with open(path, newline="") as f:
        return {row["component"]: float(row["analytic"]) for row in csv.DictReader(f)}


def run(ranks, cfg):
    cmd = ["mpiexec", "-n", str(ranks), BIN, os.path.join(HERE, cfg)]
    result = sh(cmd)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(f"run failed: n{ranks}")
    return result.stdout


def parse_thermo(stdout):
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("Step") and "Virial_xx" in line:
            header_idx = idx
            break
    if header_idx is None:
        raise RuntimeError(f"thermo header not found in output:\n{stdout}")
    headers = lines[header_idx].split()
    for line in lines[header_idx + 1:]:
        if re.fullmatch(r"-+", line.replace(" ", "")):
            continue
        parts = line.split()
        if len(parts) == len(headers) and parts[0].isdigit():
            values = dict(zip(headers, parts))
            return {
                name: float(values[name.capitalize()])
                for name in COMPONENTS
            }
    raise RuntimeError(f"thermo row not found in output:\n{stdout}")


def write_plot(measured, expected):
    os.makedirs(PLOTS, exist_ok=True)
    xs = range(len(COMPONENTS))
    labels = [c.replace("virial_", "") for c in COMPONENTS]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax = axes[0]
    ax.plot(xs, [expected[c] for c in COMPONENTS], "ko", label="analytic")
    for ranks, vals in measured.items():
        ax.plot(xs, [vals[c] for c in COMPONENTS], marker="o", label=f"{ranks} ranks")
    ax.set_xticks(list(xs), labels)
    ax.set_ylabel("virial tensor component")
    ax.set_title("Thermo virial vs analytic tensor")
    ax.legend()

    ax = axes[1]
    for ranks, vals in measured.items():
        ax.plot(xs, [abs(vals[c] - expected[c]) for c in COMPONENTS], marker="o", label=f"{ranks} ranks")
    ax.axhline(TOL, color="tab:red", linestyle="--", label=f"tolerance {TOL:.0e}")
    ax.set_yscale("symlog", linthresh=1e-16)
    ax.set_xticks(list(xs), labels)
    ax.set_ylabel("absolute error")
    ax.set_title("Reduction error")
    ax.legend()

    fig.suptitle("PASS: virial_xx..virial_yz are globally reduced")
    out = os.path.join(PLOTS, "virial_thermo_crossrank.png")
    fig.savefig(out, dpi=160)
    print(f"[plot] wrote {out}")


def main():
    build()
    expected = reference()
    measured = {}
    failures = []

    print("\nSOIL virial thermo reduction:")
    for ranks, cfg in LAYOUTS:
        vals = parse_thermo(run(ranks, cfg))
        measured[ranks] = vals
        max_err = max(abs(vals[c] - expected[c]) for c in COMPONENTS)
        ok = max_err <= TOL
        print(f"  [{'PASS' if ok else 'FAIL'}] n{ranks}: max|thermo-analytic|={max_err:.3e} (tol {TOL:.0e})")
        if not ok:
            failures.append((ranks, max_err))

    cross = max(
        abs(measured[ranks][c] - measured[1][c])
        for ranks, _ in LAYOUTS
        for c in COMPONENTS
    )
    ok = cross <= TOL
    print(f"  [{'PASS' if ok else 'FAIL'}] cross-rank consistency vs n1: max|delta|={cross:.3e} (tol {TOL:.0e})")
    if not ok:
        failures.append(("cross-rank", cross))

    write_plot(measured, expected)
    if failures:
        sys.exit("virial thermo validation failed")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
