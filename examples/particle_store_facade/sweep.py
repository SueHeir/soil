#!/usr/bin/env python3
"""Run ParticleStore's framed communication validation and plot its result."""
import csv
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BIN = os.path.join(ROOT, "target", "release", "examples", "particle_store_facade")
CONFIG = os.path.join(HERE, "config.toml")
DATA = os.path.join(HERE, "data", "particle_store_facade.csv")
PLOT = os.path.join(HERE, "plots", "particle_store_facade.png")

def run(command):
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)
    return completed.stdout

def main():
    print("[build] cargo build --release --no-default-features --features precision-double --example particle_store_facade")
    run(["cargo", "build", "--release", "--no-default-features", "--features", "precision-double", "--example", "particle_store_facade"])
    output = run([BIN, CONFIG]); print(output, end="")
    rows = []
    for line in output.splitlines():
        if line.startswith("PARTICLE_STORE "):
            row = dict(item.split("=", 1) for item in line.split()[1:])
            if row["pass"] != "true": raise SystemExit(f"failed case: {row['case']}")
            rows.append(row)
    if len(rows) != 3: raise SystemExit("expected three facade result rows")
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w", newline="") as handle:
        fields = ["case", "expected", "observed", "elapsed_ns", "pass"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    os.makedirs(os.path.dirname(PLOT), exist_ok=True)
    labels = ["framed multi-record\nparity", "trailing payload\nrollback", "facade comm\nrows measured"]
    expected = [int(r["expected"]) for r in rows]
    observed = [int(r["observed"]) for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.bar([v - .18 for v in x], expected, .36, label="required rows", color="#4c78a8")
    ax.bar([v + .18 for v in x], observed, .36, label="observed rows", color="#59a14f")
    ax.set_xticks(list(x), labels); ax.set_ylabel("rows")
    ax.set_title("PASS: ParticleStore preserves framed migration and rollback")
    ax.legend(loc="upper left")
    fig.savefig(PLOT, dpi=170)
    print(f"[data] {os.path.relpath(DATA, ROOT)}")
    print(f"[plot] {os.path.relpath(PLOT, ROOT)}")
    print("ALL CHECKS PASSED")

if __name__ == "__main__": main()
