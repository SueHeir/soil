#!/usr/bin/env python3
"""Run and graph the declared typed-label compatibility contracts."""

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
PLOT = os.path.join(HERE, "plots", "typed_system_labels_matrix.png")
CASE = re.compile(r"^CASE (\S+) expected=(\S+) observed=(\S+) status=(PASS|FAIL)$")


def main():
    cmd = [
        "cargo", "run", "--quiet", "--no-default-features", "--features",
        "precision-double", "--example", "typed_system_labels", "--", CONFIG,
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr, file=sys.stderr, end="")
        raise SystemExit(result.returncode)

    rows = [CASE.match(line) for line in result.stdout.splitlines()]
    rows = [match.groups() for match in rows if match]
    if not rows or any(status != "PASS" for _, _, _, status in rows):
        raise SystemExit("typed-label driver did not report an all-PASS matrix")

    names = [name.replace("-", "\n") for name, _, _, _ in rows]
    measured = [1 if expected == observed else 0 for _, expected, observed, _ in rows]
    os.makedirs(os.path.dirname(PLOT), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    x = range(len(rows))
    ax.bar(x, measured, color="tab:green", label="measured contract match")
    ax.plot(x, [1] * len(rows), "k--", marker="_", label="reference: declared expectation")
    ax.set_xticks(list(x), names, fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 1], ["mismatch", "match"])
    ax.set_ylabel("observed vs. expected outcome")
    ax.set_title("SOIL typed scheduler-label compatibility matrix")
    ax.text(0.02, 1.07, "PASS criterion: every measured value equals 1", transform=ax.transAxes)
    ax.legend(loc="lower right")
    fig.savefig(PLOT, dpi=160)
    print(f"[plot] wrote {PLOT}")
    print(f"VALIDATION: PASS ({len(rows)}/{len(rows)} contracts)")


if __name__ == "__main__":
    main()
