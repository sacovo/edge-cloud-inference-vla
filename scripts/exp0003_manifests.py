"""Write the EXP-0003 (tracking structure) manifests for the existing Slurm scripts (stdlib only).

Reuses EXP-0002's H = 40 teachers and teacher latents. Replay (reference arm): each teacher executes its own delayed
plan, all k in one job (a1_eval.sbatch; extra manifest fields pass --replay and the output dir). Students: 10k / 100k
`track` z-teacher students (a1_train_cpu.sbatch), evaluated per k (a1_eval.sbatch). Outputs under $ECVLA_SCRATCH/exp0003.

Usage: python3 scripts/exp0003_manifests.py
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "experiments/EXP-0003-tracking-structure/manifests"
ROOT = "$ECVLA_SCRATCH/exp0003"
E2 = "$ECVLA_SCRATCH/exp0002"
DATA = "$ECVLA_SCRATCH/a1/prepared_v2"
SUITES = ("libero_spatial", "libero_object")
SEEDS = (0, 1, 2)
SIZES = {"10k": 10_000, "100k": 100_000}
KS = (1, 4, 10, 20)


def teacher_dir(suite, seed):
    return f"{E2}/runs/teachers/{suite}/task-id_10M_h40_s{seed}"


def main():
    common = "--core gru --head l1 --vision none"
    replay = [f"{teacher_dir(su, sd)} {su} {','.join(map(str, KS))} --replay --out-dir {teacher_dir(su, sd)}/eval_replay"
              for su in SUITES for sd in SEEDS]
    students, student_eval = [], []
    for su in SUITES:
        for size, n in SIZES.items():
            for sd in SEEDS:
                rdir = f"{ROOT}/runs/students/{su}/z-teacher-h40-track_{size}_s{sd}"
                students.append(f"--data {DATA} --suites {su} --condition z-teacher --size {n} --seed {sd} {common} "
                                f"--latent-dir {E2}/teacher_z/s{sd} --latent-kind chunk --teacher {teacher_dir(su, sd)} "
                                f"--track --track-bound 0.5 --out {rdir}")
                student_eval += [f"{rdir} {su} {k}" for k in KS]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, lines in [("replay_eval", replay), ("student_train", students), ("student_eval", student_eval)]:
        (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)}")


if __name__ == "__main__":
    main()
