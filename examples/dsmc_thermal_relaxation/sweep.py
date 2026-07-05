#!/usr/bin/env python3
"""Run and plot the SOIL DSMC thermal-relaxation validation case."""

import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CFG = os.path.join(HERE, "cfg.toml")
PLOTS = os.path.join(HERE, "plots")
PLOT = os.path.join(PLOTS, "dsmc_thermal_relaxation.png")


PATTERNS = {
    "tau": r"collision time tau\s+:\s+([0-9.eE+-]+)",
    "measured": r"DSMC A/A0\s+:\s+([0-9.eE+-]+)",
    "bgk": r"BGK\s+exp\(-tau\)\s+:\s+([0-9.eE+-]+)",
    "rel_err": r"relative error\s+:\s+([0-9.eE+-]+)",
    "rate": r"fitted decay rate\s+:\s+([0-9.eE+-]+)",
}


def run(cmd):
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)
    return result.stdout


def parse_metrics(out):
    metrics = {}
    for name, pattern in PATTERNS.items():
        match = re.search(pattern, out)
        if not match:
            sys.exit(f"missing validation metric: {name}")
        metrics[name] = float(match.group(1))
    return metrics


def plot(metrics):
    os.makedirs(PLOTS, exist_ok=True)
    tau_final = metrics["tau"]
    tau = np.linspace(0.0, tau_final, 200)
    bgk = np.exp(-tau)
    rel_tol = 0.20
    rate_tol = 0.08

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.0, 3.8),
        gridspec_kw={"width_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )

    ax = axes[0]
    ax.plot(tau, bgk, color="#2f6f9f", linewidth=2.0, label="BGK exp(-tau)")
    ax.fill_between(
        tau,
        bgk * (1.0 - rel_tol),
        bgk * (1.0 + rel_tol),
        color="#2f6f9f",
        alpha=0.16,
        label="+/-20% pass band",
    )
    ax.plot(
        [0.0, tau_final],
        [1.0, metrics["measured"]],
        "o",
        color="#c23b22",
        markersize=6,
        label="DSMC A/A0",
    )
    ax.plot(
        tau,
        np.exp(-metrics["rate"] * tau),
        color="#555555",
        linestyle="--",
        linewidth=1.5,
        label=f"fit exp(-{metrics['rate']:.3f} tau)",
    )
    ax.set_xlabel("collision time tau")
    ax.set_ylabel("thermal anisotropy A/A0")
    ax.set_title("Thermal anisotropy decay")
    ax.set_xlim(0.0, tau_final)
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    ax.axhspan(1.0 - rate_tol, 1.0 + rate_tol, color="#4b8b3b", alpha=0.18)
    ax.axhline(1.0, color="#4b8b3b", linewidth=2.0, label="BGK target")
    ax.plot([0], [metrics["rate"]], "o", color="#c23b22", markersize=7)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0.88, 1.12)
    ax.set_xticks([0])
    ax.set_xticklabels(["fit"])
    ax.set_ylabel("decay rate")
    ax.set_title("+/-0.08 rate gate")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.0,
        0.895,
        f"rel. err. {metrics['rel_err']:.3e}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    fig.suptitle("SOIL DSMC thermal relaxation vs BGK reference", fontsize=12)
    fig.savefig(PLOT, dpi=180)
    plt.close(fig)
    print(f"[plot] {os.path.relpath(PLOT, ROOT)}")


def main():
    print("[run] cargo run --release --example dsmc_thermal_relaxation")
    out = run([
        "cargo",
        "run",
        "--release",
        "--example",
        "dsmc_thermal_relaxation",
        "--",
        CFG,
    ])
    print(out)
    if "ALL CHECKS PASSED" not in out:
        sys.exit("missing validation verdict")
    plot(parse_metrics(out))


if __name__ == "__main__":
    main()
