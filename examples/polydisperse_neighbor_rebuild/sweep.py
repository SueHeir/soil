#!/usr/bin/env python3
"""Validate mixed-radius neighbor pairs against an O(N^2) reference."""

import csv
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "data")
PLOTS = os.path.join(HERE, "plots")
BIN = os.path.join(ROOT, "target", "release", "examples", "polydisperse_neighbor_rebuild")
LAYOUTS = [(1, "cfg_n1.toml"), (2, "cfg_n2.toml"), (4, "cfg_n4.toml")]

NX, NY, NZ = 8, 4, 3
SPACING = 1.40
JITTER = 0.09
RADII = [0.18, 0.28, 0.42, 0.56, 0.70]
SEED = 0x5011202607064242
SKIN_FRACTION = 1.40


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


def signed_unit(state):
    state, bits = splitmix64(state)
    u = (bits >> 11) / float(1 << 53)
    return state, 2.0 * u - 1.0


def global_atoms():
    atoms = []
    state = SEED
    for ix in range(NX):
        for iy in range(NY):
            for iz in range(NZ):
                tag = ix * NY * NZ + iy * NZ + iz
                state, jx = signed_unit(state)
                state, jy = signed_unit(state)
                state, jz = signed_unit(state)
                pos = (
                    (ix + 0.5) * SPACING + JITTER * jx,
                    (iy + 0.5) * SPACING + JITTER * jy,
                    (iz + 0.5) * SPACING + JITTER * jz,
                )
                radius = RADII[(ix + 2 * iy + 3 * iz) % len(RADII)]
                atoms.append((tag, pos, radius))
    return atoms


def brute_pairs():
    atoms = global_atoms()
    pairs = set()
    for a, (ti, pi, ri) in enumerate(atoms):
        for tj, pj, rj in atoms[a + 1 :]:
            dx = pj[0] - pi[0]
            dy = pj[1] - pi[1]
            dz = pj[2] - pi[2]
            cutoff = (ri + rj) * SKIN_FRACTION
            if dx * dx + dy * dy + dz * dz < cutoff * cutoff:
                pairs.add((ti, tj))
    return pairs


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def build():
    print("[build] cargo build --release --example polydisperse_neighbor_rebuild")
    result = run(["cargo", "build", "--release", "--example", "polydisperse_neighbor_rebuild"])
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit("build failed")


def run_layout(ranks, cfg):
    cmd = ["mpiexec", "-n", str(ranks), BIN, os.path.join(HERE, cfg)]
    print("[run] " + " ".join(cmd))
    result = run(cmd)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(f"run failed for n={ranks}")
    return result.stdout


def load_pairs(ranks):
    pairs = set()
    for rank in range(ranks):
        path = os.path.join(DATA, f"pairs_n{ranks}_rank{rank}.csv")
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                pairs.add((int(row["tag_i"]), int(row["tag_j"])))
    return pairs


def write_plot(results, ref_count):
    os.makedirs(PLOTS, exist_ok=True)
    ranks = [r for r, _ in LAYOUTS]
    missing = [results[r]["missing"] for r in ranks]
    extra = [results[r]["extra"] for r in ranks]

    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    width = 0.35
    xs = list(range(len(ranks)))
    ax.bar([x - width / 2 for x in xs], missing, width, label="missing pairs")
    ax.bar([x + width / 2 for x in xs], extra, width, label="extra pairs")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(xs, [str(r) for r in ranks])
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("pair-count mismatch vs brute force")
    ax.set_title(f"PASS: mixed-radius neighbor pairs match brute force ({ref_count} pairs)")
    ax.legend()
    out = os.path.join(PLOTS, "polydisperse_neighbor_rebuild.png")
    fig.savefig(out, dpi=160)
    print(f"[plot] wrote {out}")


def main():
    os.makedirs(DATA, exist_ok=True)
    build()
    want = brute_pairs()
    results = {}
    ok = True
    for ranks, cfg in LAYOUTS:
        run_layout(ranks, cfg)
        got = load_pairs(ranks)
        missing = want - got
        extra = got - want
        results[ranks] = {"missing": len(missing), "extra": len(extra), "got": len(got)}
        passed = not missing and not extra
        ok = ok and passed
        print(
            f"  [{'PASS' if passed else 'FAIL'}] n={ranks}: "
            f"pairs={len(got)} reference={len(want)} missing={len(missing)} extra={len(extra)}"
        )
        if not passed:
            print("    sample_missing=", sorted(missing)[:10])
            print("    sample_extra=", sorted(extra)[:10])
    write_plot(results, len(want))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
