#!/usr/bin/env python3
"""Run the SOIL DSMC thermal-relaxation validation case."""

import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CFG = os.path.join(HERE, "cfg.toml")


def run(cmd):
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)
    return result.stdout


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


if __name__ == "__main__":
    main()
