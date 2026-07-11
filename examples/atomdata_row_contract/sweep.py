#!/usr/bin/env python3
"""Run the AtomData registration contract checks and plot observed row counts."""
import csv
import os
import subprocess
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BIN = os.path.join(ROOT, "target", "release", "examples", "atomdata_row_contract")
CONFIG = os.path.join(HERE, "config.toml")
DATA = os.path.join(HERE, "data", "atomdata_row_contract.csv")
PLOT = os.path.join(HERE, "plots", "atomdata_row_contract.png")

def run(cmd):
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)
    return completed.stdout

def main():
    print("[build] cargo build --release --example atomdata_row_contract")
    run(["cargo", "build", "--release", "--example", "atomdata_row_contract"])
    output = run([BIN, CONFIG]); print(output, end="")
    rows = []
    for line in output.splitlines():
        if line.startswith("ATOMDATA_ROW "):
            row = dict(item.split("=", 1) for item in line.split()[1:])
            if row["pass"] != "true": raise SystemExit(f"failed case: {row['case']}")
            rows.append(row)
    if len(rows) != 2: raise SystemExit("expected two AtomData contract rows")
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "expected", "observed", "accepted", "pass"],
            lineterminator="\n",
        )
        writer.writeheader(); writer.writerows(rows)
    os.makedirs(os.path.dirname(PLOT), exist_ok=True)
    labels = ["late derived\n(accepted)", "manual multi-column\n(rejected)"]
    expected = [int(row["expected"]) for row in rows]; observed = [int(row["observed"]) for row in rows]; x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.bar([v - 0.18 for v in x], expected, 0.36, label="reference atom rows", color="#4c78a8")
    ax.bar([v + 0.18 for v in x], observed, 0.36, label="measured extension rows", color="#f58518")
    ax.axhline(3, color="#333", linestyle="--", linewidth=1, label="exact pass boundary (3)")
    ax.set_xticks(x, labels); ax.set_ylabel("rows"); ax.set_ylim(0, 4)
    ax.set_title("PASS: aligned store accepted; hidden short column rejected"); ax.legend(loc="lower left")
    fig.savefig(PLOT, dpi=170)
    print(f"[data] {os.path.relpath(DATA, ROOT)}"); print(f"[plot] {os.path.relpath(PLOT, ROOT)}"); print("ALL CHECKS PASSED")

if __name__ == "__main__": main()
