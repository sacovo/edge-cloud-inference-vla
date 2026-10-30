"""EXP-0002 (plan horizon): the pre-registered H1 result (README §5) and the reported, non-gating comparisons.

Inputs are a1_summarize.py outputs. H = 8 numbers are EXP-0001's (students: a1_teacher_summary.json, chunk latent;
teachers: the 10M blind task-id runs in a1_grid_blind_summary.json); H = 40 numbers are EXP-0002's.
Pooling: mean of Spatial and Object per seed; student seed s is paired with teacher seed s.
Ratio = mean over seeds of the student / mean over seeds of its own teacher (as EXP-0001 H4); per-seed values are
reported too. 95% CIs: bootstrap over the 3 seeds (resampled together, paired).

Usage: python3 scripts/exp0002_summarize.py <h40_teachers.json> <h40_students.json> [--out summary.json]
"""

import argparse
import json
import random
from pathlib import Path

EXP1 = Path(__file__).resolve().parents[1] / "experiments/EXP-0001-latent-taskid-vs-state/results"
SUITES = ("libero_spatial", "libero_object")
SEEDS = (0, 1, 2)
KS = (1, 4, 10, 20)
SIZES = (10_000, 100_000)


def index(rows, keep):
    """{(suite, size, seed, k): success} for complete rows that pass keep."""
    out = {}
    for r in rows:
        if r["suite"] in SUITES and keep(r):
            assert r["tasks"] == 10, f"incomplete eval: {r['suite']} {r['run']} k={r['k']} ({r['tasks']} tasks)"
            out[(r["suite"], r["size_target"], r["seed"], r["k"])] = r["success_rate"]
    return out


def pooled(sr, size, k):
    """Per-seed mean over the two suites."""
    return [sum(sr[(su, size, sd, k)] for su in SUITES) / len(SUITES) for sd in SEEDS]


def mean(x):
    return sum(x) / len(x)


def boot(fn, n=10_000, seed=0):
    """95% CI of fn(seed indices) over seed resamples."""
    rng = random.Random(seed)
    v = sorted(fn([rng.randrange(len(SEEDS)) for _ in SEEDS]) for _ in range(n))
    return [round(v[int(0.025 * n)], 4), round(v[int(0.975 * n) - 1], 4)]


def compare(s8, s40, t8, t40, size, k):
    a, b = pooled(s8, size, k), pooled(s40, size, k)
    diff = [y - x for x, y in zip(a, b)]
    pick = lambda v, i: mean([v[j] for j in i])  # noqa: E731
    return {
        "size": size, "k": k,
        "student_h8": round(mean(a), 4), "student_h40": round(mean(b), 4),
        "student_h8_per_seed": [round(x, 4) for x in a], "student_h40_per_seed": [round(x, 4) for x in b],
        "diff_pp": round(100 * mean(diff), 2), "diff_pp_per_seed": [round(100 * d, 2) for d in diff],
        "diff_pp_ci95": [round(100 * x, 2) for x in boot(lambda i: pick(diff, i))],
        "ratio_h8": round(mean(a) / mean(t8), 4), "ratio_h40": round(mean(b) / mean(t40), 4),
        "ratio_h40_per_seed": [round(x / t, 4) for x, t in zip(b, t40)],
        "ratio_h40_ci95": boot(lambda i: pick(b, i) / pick(t40, i)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teachers_h40")
    ap.add_argument("students_h40")
    ap.add_argument("--out")
    a = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    t8 = index(load(EXP1 / "a1_grid_blind_summary.json"),
               lambda r: r["condition"] == "task-id" and r["size_target"] == 10_000_000)
    s8 = index(load(EXP1 / "a1_teacher_summary.json"), lambda r: "z-teacher-chunk" in r["run"])
    t40 = index(load(a.teachers_h40), lambda r: r["condition"] == "task-id")
    s40 = index(load(a.students_h40), lambda r: r["condition"] == "z-teacher")
    tp8, tp40 = pooled(t8, 10_000_000, None), pooled(t40, 10_000_000, None)

    rows = [compare(s8, s40, tp8, tp40, size, k) for size in SIZES for k in KS]
    p = next(r for r in rows if r["size"] == 100_000 and r["k"] == 10)
    if p["ratio_h40"] >= 0.5 and p["diff_pp"] >= 15:
        verdict = "supported"
    elif p["diff_pp"] < 5:
        verdict = "refuted"
    else:
        verdict = "neither (inconclusive)"
    per_suite = {f"{su}|{size}|k{k}": {"h8": round(mean([s8[(su, size, sd, k)] for sd in SEEDS]), 4),
                                       "h40": round(mean([s40[(su, size, sd, k)] for sd in SEEDS]), 4)}
                 for su in SUITES for size in SIZES for k in KS}
    out = {
        "h1_primary": {"criterion": "100k, k=10: ratio_h40 >= 0.5 and diff >= 15 pp (refuted if diff < 5 pp)",
                       "verdict": verdict, **p},
        "teachers": {"h8": round(mean(tp8), 4), "h40": round(mean(tp40), 4),
                     "h8_per_seed": [round(x, 4) for x in tp8], "h40_per_seed": [round(x, 4) for x in tp40],
                     "per_suite": {su: {"h8": round(mean([t8[(su, 10_000_000, sd, None)] for sd in SEEDS]), 4),
                                        "h40": round(mean([t40[(su, 10_000_000, sd, None)] for sd in SEEDS]), 4)}
                                   for su in SUITES}},
        "comparisons": rows,
        "per_suite_students": per_suite,
    }
    print(f"H1 (100k, k=10): {verdict}; student H40 {p['student_h40']:.3f} vs H8 {p['student_h8']:.3f} "
          f"({p['diff_pp']:+.1f} pp, CI {p['diff_pp_ci95']}); ratio to own teacher {p['ratio_h40']:.2f}")
    print(f"teachers: H8 {out['teachers']['h8']:.3f}  H40 {out['teachers']['h40']:.3f}")
    print(f"{'size':>7} {'k':>3} {'H8':>6} {'H40':>6} {'diff pp':>8} {'ratio8':>7} {'ratio40':>7}")
    for r in rows:
        print(f"{r['size']:>7} {r['k']:>3} {r['student_h8']:6.3f} {r['student_h40']:6.3f} {r['diff_pp']:+8.1f} "
              f"{r['ratio_h8']:7.2f} {r['ratio_h40']:7.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
