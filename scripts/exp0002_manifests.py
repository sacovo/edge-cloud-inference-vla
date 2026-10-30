"""Write the EXP-0002 (plan horizon) manifests for the existing Slurm scripts (stdlib only).

Teachers: 10M blind task-id GRU with a 40-step chunk (a1_train.sbatch, GPU) and their evals (a1_eval.sbatch).
Latents: teacher chunk on the demo frames (a1_teacher_latents.sbatch). Students: 10k / 100k z-teacher on that latent
(a1_train_cpu.sbatch), evaluated per k (a1_eval.sbatch). Everything lives under $ECVLA_SCRATCH/exp0002.

Usage: python3 scripts/exp0002_manifests.py
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "experiments/EXP-0002-plan-horizon/manifests"
ROOT = "$ECVLA_SCRATCH/exp0002"
DATA = "$ECVLA_SCRATCH/a1/prepared_v2"
SUITES = ("libero_spatial", "libero_object")
SEEDS = (0, 1, 2)
H = 40
SIZES = {"10k": 10_000, "100k": 100_000}
KS = (1, 4, 10, 20)


def teacher_dir(suite, seed):
    return f"{ROOT}/runs/teachers/{suite}/task-id_10M_h{H}_s{seed}"


def main():
    common = "--core gru --head l1 --vision none"
    teachers = [f"--data {DATA} --suites {su} --condition task-id --size 10000000 --seed {sd} --chunk {H} {common} "
                f"--out {teacher_dir(su, sd)}" for su in SUITES for sd in SEEDS]
    teacher_eval = [f"{teacher_dir(su, sd)} {su} -" for su in SUITES for sd in SEEDS]
    latents = [f"{teacher_dir(su, sd)} {DATA}/{su} {ROOT}/teacher_z/s{sd}/{su}" for su in SUITES for sd in SEEDS]
    students, student_eval = [], []
    for su in SUITES:
        for size, n in SIZES.items():
            for sd in SEEDS:
                rdir = f"{ROOT}/runs/students/{su}/z-teacher-h{H}_{size}_s{sd}"
                students.append(f"--data {DATA} --suites {su} --condition z-teacher --size {n} --seed {sd} {common} "
                                f"--latent-dir {ROOT}/teacher_z/s{sd} --latent-kind chunk --teacher {teacher_dir(su, sd)} "
                                f"--out {rdir}")
                student_eval += [f"{rdir} {su} {k}" for k in KS]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, lines in [("teacher_train", teachers), ("teacher_eval", teacher_eval), ("teacher_latents", latents),
                        ("student_train", students), ("student_eval", student_eval)]:
        (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)}")


if __name__ == "__main__":
    main()
