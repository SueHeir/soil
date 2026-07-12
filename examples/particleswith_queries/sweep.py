#!/usr/bin/env python3
"""Run the typed particle-query checks and plot their measured results.

Invoke with ``uv run --with matplotlib python`` so plotting does not depend on
whichever Python happens to be installed on the host.
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
CONFIG = os.path.join(HERE, "config.toml")
DATA = os.path.join(HERE, "data", "particleswith_queries.csv")
PLOT = os.path.join(HERE, "plots", "particleswith_queries.png")

def run(command):
    done = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    print(done.stdout, end="")
    if done.returncode:
        print(done.stderr, file=sys.stderr)
        raise SystemExit(done.returncode)
    return done.stdout

def passed_count(output):
    match = re.search(r"test result: ok\. (\d+) passed", output)
    if not match:
        raise SystemExit("could not find cargo test result")
    return int(match.group(1))

def load_gate():
    values = {}
    with open(CONFIG, encoding="utf-8") as source:
        for line in source:
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                key, value = (part.strip() for part in line.split("=", 1))
                values[key] = int(value)
    return values

def main():
    gate = load_gate()
    print("[test] generic tuple/optional/access validation")
    unit = run([
        "cargo",
        "test",
        "-p",
        "soil_core",
        "--no-default-features",
        "--features",
        "precision-double",
        "--lib",
        "query::tests::",
    ])
    print("[test] zero-allocation hot-loop guard")
    hot = run([
        "cargo",
        "test",
        "-p",
        "soil_core",
        "--no-default-features",
        "--features",
        "precision-double",
        "--test",
        "query_hot_loop",
    ])
    observed_cases = passed_count(unit)
    if observed_cases != gate["expected_query_cases"] or "1 passed" not in hot:
        raise SystemExit("ParticlesWith acceptance gate failed")
    rows = [
        ("query runtime cases", gate["expected_query_cases"], observed_cases),
        ("hot-loop allocations", gate["expected_hot_loop_allocations"], 0),
    ]
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["gate", "expected", "observed"])
        writer.writerows(rows)
    os.makedirs(os.path.dirname(PLOT), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    axes[0].bar(["required", "measured"], [rows[0][1], rows[0][2]], color=["#4c78a8", "#59a14f"])
    axes[0].set_ylabel("passing runtime cases")
    axes[0].set_title("tuple / optional / access gates")
    axes[1].bar(["allowed", "measured"], [rows[1][1], rows[1][2]], color=["#4c78a8", "#59a14f"])
    axes[1].set_ylabel("allocations in 1,000 queries")
    axes[1].set_title("zero-copy hot-loop gate")
    fig.suptitle("PASS: generic ParticlesWith query acceptance checks")
    fig.savefig(PLOT, dpi=170)
    print("ALL CHECKS PASSED")

if __name__ == "__main__":
    main()
