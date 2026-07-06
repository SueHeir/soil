#!/usr/bin/env python3
"""Run and plot the SOIL ORBIT two-body gravity validation case."""

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
PLOT = os.path.join(PLOTS, "orbit_nbody_gravity.png")

PERIOD_TOL = 1.0e-3
ENERGY_TOL = 1.0e-6
NEGATIVE_DRIFT_MIN = 1.0e-2


def run(cmd):
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)
    return result.stdout


def parse_metrics(out):
    period_match = re.search(
        r"analytic two-body circular orbit: period = 2\*pi = ([0-9.eE+-]+)", out
    )
    if not period_match:
        sys.exit("missing analytic period")

    blocks = re.split(r"\n\[[12]\] ", out)
    if len(blocks) < 3:
        sys.exit("missing theta validation blocks")

    metrics = {"reference_period": float(period_match.group(1))}
    for name, block in (("theta_0_5", blocks[1]), ("theta_4_0", blocks[2])):
        period = re.search(r"measured period\s+=\s+([0-9.eE+-]+) \(rel err ([0-9.eE+-]+)\)", block)
        energy = re.search(
            r"max energy rel drift\s+=\s+([0-9.eE+-]+); final drift = ([0-9.eE+-]+)",
            block,
        )
        if not period or not energy:
            sys.exit(f"missing validation metrics for {name}")
        metrics[name] = {
            "measured_period": float(period.group(1)),
            "period_rel_err": float(period.group(2)),
            "max_energy_rel_drift": float(energy.group(1)),
            "final_energy_rel_drift": float(energy.group(2)),
        }
    return metrics


def plot(metrics):
    os.makedirs(PLOTS, exist_ok=True)
    labels = ["theta=0.5", "theta=4.0\nnegative"]
    keys = ["theta_0_5", "theta_4_0"]
    period_values = [metrics[k]["measured_period"] for k in keys]
    period_err = [metrics[k]["period_rel_err"] for k in keys]
    drift = [metrics[k]["max_energy_rel_drift"] for k in keys]
    reference = metrics["reference_period"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 3.9),
        constrained_layout=True,
    )

    ax = axes[0]
    ax.axhspan(
        reference * (1.0 - PERIOD_TOL),
        reference * (1.0 + PERIOD_TOL),
        color="#4b8b3b",
        alpha=0.18,
        label="+/-1e-3 period gate",
    )
    ax.axhline(reference, color="#2f6f9f", linewidth=2.0, label="analytic 2*pi")
    ax.plot(labels, period_values, "o", color="#c23b22", markersize=7, label="measured")
    for i, err in enumerate(period_err):
        ax.text(i, period_values[i], f"rel err {err:.2e}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("orbital period")
    ax.set_title("Two-body period vs Kepler reference")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    ax.axhline(ENERGY_TOL, color="#4b8b3b", linewidth=2.0, label="theta=0.5 max drift gate")
    ax.axhline(
        NEGATIVE_DRIFT_MIN,
        color="#8a4f2a",
        linewidth=2.0,
        linestyle="--",
        label="negative-control minimum",
    )
    ax.scatter(labels, drift, color=["#4b8b3b", "#c23b22"], s=55, zorder=3)
    for i, value in enumerate(drift):
        ax.text(i, value, f"{value:.2e}", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(5.0e-13, 1.0e3)
    ax.set_ylabel("max relative energy drift")
    ax.set_title("Energy drift gates")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    good = metrics["theta_0_5"]
    bad = metrics["theta_4_0"]
    verdict = (
        good["period_rel_err"] < PERIOD_TOL
        and good["max_energy_rel_drift"] < ENERGY_TOL
        and bad["max_energy_rel_drift"] > NEGATIVE_DRIFT_MIN
    )
    fig.suptitle(
        "PASS: ORBIT gravity validation" if verdict else "FAIL: ORBIT gravity validation",
        fontsize=12,
    )
    fig.savefig(PLOT, dpi=180)
    plt.close(fig)
    print(f"[plot] {os.path.relpath(PLOT, ROOT)}")


def main():
    print("[run] cargo run --release --example orbit_nbody_gravity")
    out = run(["cargo", "run", "--release", "--example", "orbit_nbody_gravity"])
    print(out)
    if "ALL CHECKS PASSED" not in out:
        sys.exit("missing validation verdict")
    plot(parse_metrics(out))


if __name__ == "__main__":
    main()
