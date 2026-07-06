#!/usr/bin/env python3
"""Run SOIL neighbor/communication profiling layouts and plot the counters."""

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
BIN = os.path.join(ROOT, "target", "release", "examples", "neighbor_comm_profile")
DATA = os.path.join(HERE, "data")
PLOTS = os.path.join(HERE, "plots")
CSV = os.path.join(DATA, "profile_summary.csv")
PLOT = os.path.join(PLOTS, "neighbor_comm_profile.png")

LAYOUTS = [
    ("serial", 1, os.path.join(HERE, "config_serial.toml")),
    ("mpi_x2", 2, os.path.join(HERE, "config_mpi_x2.toml")),
]

PROFILE_RE = re.compile(r"PROFILE (?P<body>.*)")


def run(cmd):
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)
    return result.stdout


def parse_profiles(layout, stdout):
    rows = []
    for line in stdout.splitlines():
        match = PROFILE_RE.search(line)
        if not match:
            continue
        row = {"layout_name": layout}
        for item in match.group("body").split():
            key, value = item.split("=", 1)
            row[key] = value
        rows.append(row)
    if not rows:
        sys.exit(f"no PROFILE rows parsed for {layout}")
    return rows


def write_csv(rows):
    os.makedirs(DATA, exist_ok=True)
    fields = [
        "layout_name",
        "layout",
        "size",
        "rank",
        "phase",
        "calls",
        "mean_ns",
        "total_ns",
        "min_ns",
        "max_ns",
        "alloc_calls",
        "alloc_bytes",
        "p2p_messages",
        "p2p_f64",
        "collectives",
        "nlocal",
        "nghost",
        "pairs",
        "iterations",
        "warmup",
    ]
    with open(CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[data] {os.path.relpath(CSV, ROOT)}")


def plot(rows):
    os.makedirs(PLOTS, exist_ok=True)
    phases = [
        "sort_atoms_by_bin",
        "borders",
        "forward_comm",
        "reverse_comm",
        "neighbor_build",
    ]
    layouts = [name for name, _, _ in LAYOUTS]
    means = {}
    messages = {}
    allocs = {}
    for layout in layouts:
        means[layout] = []
        messages[layout] = []
        allocs[layout] = []
        for phase in phases:
            subset = [r for r in rows if r["layout_name"] == layout and r["phase"] == phase]
            means[layout].append(sum(float(r["mean_ns"]) for r in subset) / len(subset) / 1.0e6)
            messages[layout].append(sum(int(r["p2p_messages"]) for r in subset))
            allocs[layout].append(sum(int(r["alloc_calls"]) for r in subset))

    x = range(len(phases))
    width = 0.35
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.2), constrained_layout=True)

    ax = axes[0]
    for idx, layout in enumerate(layouts):
        offset = (idx - 0.5) * width
        ax.bar([v + offset for v in x], means[layout], width, label=layout)
    ax.set_ylabel("mean time / call (ms)")
    ax.set_title("SOIL neighbor/communication phase timing")
    ax.set_xticks(list(x), phases, rotation=20, ha="right")
    ax.legend()

    ax = axes[1]
    for idx, layout in enumerate(layouts):
        offset = (idx - 0.5) * width
        ax.bar([v + offset for v in x], messages[layout], width, label=layout)
    ax.set_ylabel("p2p messages")
    ax.set_title("Recorded point-to-point messages")
    ax.set_xticks(list(x), phases, rotation=20, ha="right")

    ax = axes[2]
    for idx, layout in enumerate(layouts):
        offset = (idx - 0.5) * width
        ax.bar([v + offset for v in x], allocs[layout], width, label=layout)
    ax.set_ylabel("allocation calls")
    ax.set_title("Recorded allocation calls")
    ax.set_xticks(list(x), phases, rotation=20, ha="right")

    fig.suptitle("PASS: serial and MPI neighbor/comm profile captured")
    fig.savefig(PLOT, dpi=170)
    print(f"[plot] {os.path.relpath(PLOT, ROOT)}")


def main():
    print("[build] cargo build --release --example neighbor_comm_profile")
    run(["cargo", "build", "--release", "--example", "neighbor_comm_profile"])
    rows = []
    for layout, ranks, cfg in LAYOUTS:
        cmd = [BIN, cfg] if ranks == 1 else ["mpiexec", "-n", str(ranks), BIN, cfg]
        print("[run] " + " ".join(os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd))
        out = run(cmd)
        print(out)
        rows.extend(parse_profiles(layout, out))
    write_csv(rows)
    plot(rows)
    expected = {"sort_atoms_by_bin", "borders", "forward_comm", "reverse_comm", "neighbor_build"}
    seen = {r["phase"] for r in rows}
    if not expected.issubset(seen):
        sys.exit(f"missing phases: {sorted(expected - seen)}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
