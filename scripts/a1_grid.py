"""Write the A1 train/eval manifests for scripts/slurm/a1_{train,eval}.sbatch (stdlib only, runs anywhere).

Stages:
- archcheck: one seed, blind, LIBERO-Spatial, conditions none / oracle-fresh, sizes 1M / 10M,
  core gru|tf x head l1|flow. Picks the architecture that is then fixed for the grid (EXP-0001 change log).
- grid: the Phase A A1 grid (README §7.1): per suite, vision none|full, sizes 10k..10M, conditions none / task-id /
  oracle-delayed / oracle-fresh, 3 seeds; oracle-delayed is evaluated at k = 1 and 10. Needs --core and --head.

Usage: python3 scripts/a1_grid.py archcheck
       python3 scripts/a1_grid.py grid --core gru --head l1
"""

import argparse
import itertools
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "experiments/EXP-0001-latent-taskid-vs-state/manifests"
ROOT = "$ECVLA_SCRATCH/a1"
SUITES = ("libero_spatial", "libero_object", "libero_goal")
SIZES = {"10k": 10_000, "100k": 100_000, "1M": 1_000_000, "10M": 10_000_000}
K_EVAL = (1, 10)


def run(stage, suite, vision, cond, size, core, head, seed):
    name = f"{vision}_{cond}_{size}_{core}-{head}_s{seed}"
    rdir = f"{ROOT}/runs/{stage}/{suite}/{name}"
    args = [f"--data {ROOT}/prepared", f"--suites {suite}", f"--condition {cond}", f"--size {SIZES[size]}",
            f"--seed {seed}", f"--core {core}", f"--head {head}", f"--vision {vision}", f"--out {rdir}"]
    if vision == "full":
        args.append(f"--vision-dir {ROOT}/vision")
    evals = [f"{rdir} {suite} {t} {k}" for t in range(10) for k in (K_EVAL if cond == "oracle-delayed" else ["-"])]
    return " ".join(args), evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["archcheck", "grid"])
    ap.add_argument("--core", choices=["gru", "tf"])
    ap.add_argument("--head", choices=["l1", "flow"])
    a = ap.parse_args()
    if a.stage == "archcheck":
        cells = itertools.product(["libero_spatial"], ["none"], ["none", "oracle-fresh"], ["1M", "10M"],
                                  ["gru", "tf"], ["l1", "flow"], [0])
    else:
        assert a.core and a.head, "grid needs the architecture chosen by the archcheck"
        cells = itertools.product(SUITES, ["none", "full"], ["none", "task-id", "oracle-delayed", "oracle-fresh"],
                                  list(SIZES), [a.core], [a.head], [0, 1, 2])
    train, evals = [], []
    for c in cells:
        t, e = run(a.stage, *c)
        train.append(t)
        evals += e
    OUT.mkdir(exist_ok=True)
    (OUT / f"a1_{a.stage}_train.txt").write_text("\n".join(train) + "\n")
    (OUT / f"a1_{a.stage}_eval.txt").write_text("\n".join(evals) + "\n")
    print(f"{a.stage}: {len(train)} train runs, {len(evals)} eval jobs -> {OUT}")


if __name__ == "__main__":
    main()
