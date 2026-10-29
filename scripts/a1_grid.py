"""Write the A1 train/eval manifests for scripts/slurm/a1_{train,eval}.sbatch (stdlib only, runs anywhere).

Stages:
- archcheck: one seed, blind, LIBERO-Spatial, conditions none / oracle-fresh, sizes 1M / 10M,
  core gru|tf x head l1|flow. Picks the architecture that is then fixed for the grid (EXP-0001 change log).
- teacher: A1-teacher (H4): blind 10k / 100k students conditioned on z from the 10M `task-id` GRU teacher of the
  same suite and seed; z = its action chunk (primary) or hidden state; evaluated at k = 1, 4, 10, 20 (one eval job
  per run and k). Also writes the teacher-latent manifest (one line per suite and seed).
- grid: the Phase A A1 grid (README §7.1): per suite, vision none|full, sizes 10k..10M, conditions none / task-id /
  oracle-delayed / oracle-fresh, 3 seeds; oracle-delayed is evaluated at k = 1 and 10. Needs --core and --head.

Usage: python3 scripts/a1_grid.py archcheck
       python3 scripts/a1_grid.py grid --core gru --head l1
       python3 scripts/a1_grid.py teacher
"""

import argparse
import itertools
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "experiments/EXP-0001-latent-taskid-vs-state/manifests"
ROOT = "$ECVLA_SCRATCH/a1"
SUITES = ("libero_spatial", "libero_object", "libero_goal")
SIZES = {"10k": 10_000, "100k": 100_000, "1M": 1_000_000, "10M": 10_000_000}
K_EVAL = (1, 10)
K_TEACHER = (1, 4, 10, 20)


def teacher_dir(suite, seed):
    return f"{ROOT}/runs/grid/{suite}/none_task-id_10M_gru-l1_s{seed}"


def teacher_run(suite, kind, size, seed):
    rdir = f"{ROOT}/runs/teacher/{suite}/none_z-teacher-{kind}_{size}_gru-l1_s{seed}"
    args = [f"--data {ROOT}/prepared_v2", f"--suites {suite}", "--condition z-teacher", f"--size {SIZES[size]}",
            f"--seed {seed}", "--core gru", "--head l1", "--vision none", f"--latent-dir {ROOT}/teacher_z/s{seed}",
            f"--latent-kind {kind}", f"--teacher {teacher_dir(suite, seed)}", f"--out {rdir}"]
    return " ".join(args), [f"{rdir} {suite} {k}" for k in K_TEACHER]


def run(stage, suite, vision, cond, size, core, head, seed):
    name = f"{vision}_{cond}_{size}_{core}-{head}_s{seed}"
    rdir = f"{ROOT}/runs/{stage}/{suite}/{name}"
    data = "prepared" if stage == "archcheck" else "prepared_v2"  # v2 adds fixture-joint state to the oracle
    args = [f"--data {ROOT}/{data}", f"--suites {suite}", f"--condition {cond}", f"--size {SIZES[size]}",
            f"--seed {seed}", f"--core {core}", f"--head {head}", f"--vision {vision}", f"--out {rdir}"]
    if vision == "full":
        args.append(f"--vision-dir {ROOT}/vision")
    ks = ",".join(map(str, K_EVAL)) if cond == "oracle-delayed" else "-"
    return " ".join(args), [f"{rdir} {suite} {ks}"]  # one eval job per run: all 10 tasks (x k) in one process


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["archcheck", "grid", "teacher"])
    ap.add_argument("--core", choices=["gru", "tf"])
    ap.add_argument("--head", choices=["l1", "flow"])
    a = ap.parse_args()
    if a.stage == "teacher":
        OUT.mkdir(exist_ok=True)
        train, evals = [], []
        for c in itertools.product(SUITES, ["chunk", "hidden"], ["10k", "100k"], [0, 1, 2]):
            t, e = teacher_run(*c)
            train.append(t)
            evals += e
        z = [f"{teacher_dir(su, sd)} {ROOT}/prepared_v2/{su} {ROOT}/teacher_z/s{sd}/{su}" for su in SUITES for sd in (0, 1, 2)]
        (OUT / "a1_teacher_latents.txt").write_text("\n".join(z) + "\n")
        (OUT / "a1_teacher_train.txt").write_text("\n".join(train) + "\n")
        (OUT / "a1_teacher_eval.txt").write_text("\n".join(evals) + "\n")
        print(f"teacher: {len(z)} latent jobs, {len(train)} train runs, {len(evals)} eval jobs -> {OUT}")
        return
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
    print(f"{a.stage}: {len(train)} train runs, {len(evals)} eval jobs (one per run) -> {OUT}")


if __name__ == "__main__":
    main()
