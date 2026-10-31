"""Write the EXP-0004 (hold start) manifests for a1_eval.sbatch (stdlib only).

No training: the EXP-0003 `track` students and the EXP-0002 H40 teachers are reused. Extra manifest fields pass
--hold-start / --replay and the output dir (the later --out-dir wins over the script's default <run>/eval).
  student_hold: students with --hold-start -> <run>/eval_hold (one job per run and k)
  student_log:  students without hold, rerun for the residual log -> <run>/eval_log (one job per run and k)
  replay_hold:  teachers, --replay --hold-start -> <teacher>/eval_replay_hold (one job per teacher, both k)

Usage: python3 scripts/exp0004_manifests.py
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "experiments/EXP-0004-hold-start/manifests"
E2 = "$ECVLA_SCRATCH/exp0002"
E3 = "$ECVLA_SCRATCH/exp0003"
SUITES = ("libero_spatial", "libero_object")
SEEDS = (0, 1, 2)
SIZES = ("10k", "100k")
KS = (10, 20)


def main():
    hold, log = [], []
    for su in SUITES:
        for size in SIZES:
            for sd in SEEDS:
                rdir = f"{E3}/runs/students/{su}/z-teacher-h40-track_{size}_s{sd}"
                hold += [f"{rdir} {su} {k} --hold-start --out-dir {rdir}/eval_hold" for k in KS]
                log += [f"{rdir} {su} {k} --out-dir {rdir}/eval_log" for k in KS]
    replay = []
    for su in SUITES:
        for sd in SEEDS:
            t = f"{E2}/runs/teachers/{su}/task-id_10M_h40_s{sd}"
            replay.append(f"{t} {su} {','.join(map(str, KS))} --replay --hold-start --out-dir {t}/eval_replay_hold")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, lines in [("student_hold", hold), ("student_log", log), ("replay_hold", replay)]:
        (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)}")


if __name__ == "__main__":
    main()
