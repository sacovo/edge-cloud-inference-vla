"""EXP-0003 (tracking structure): the pre-registered H1 result (README §5) and the reported, non-gating comparisons.

Inputs are a1_summarize.py outputs: the `track` students (EXP-0003 runs/students) and the replay reference arm
(EXP-0002 teachers with --eval-dir eval_replay). Baselines are EXP-0002's results: its students (full 280-d chunk +
age) and its H = 40 teachers. Pooling, ratio and bootstrap as EXP-0002 (scripts/exp0002_summarize.py).

Usage: python3 scripts/exp0003_summarize.py <track_students.json> <replay.json> [--out summary.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp0002_summarize import KS, SEEDS, SIZES, SUITES, boot, index, mean, pooled  # noqa: E402

EXP2 = Path(__file__).resolve().parents[1] / "experiments/EXP-0002-plan-horizon/results"
TEACHER = 10_000_000


def r4(x):
    return [round(v, 4) for v in x] if isinstance(x, list) else round(x, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("students_track")
    ap.add_argument("replay")
    ap.add_argument("--out")
    a = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    teach = pooled(index(load(EXP2 / "teachers_h40.json"), lambda r: r["condition"] == "task-id"), TEACHER, None)
    base = index(load(EXP2 / "students_h40.json"), lambda r: r["condition"] == "z-teacher")
    track = index(load(a.students_track), lambda r: r["condition"] == "z-teacher")
    rep = index(load(a.replay), lambda r: True)
    pick = lambda v, i: mean([v[j] for j in i])  # noqa: E731

    rows = []
    for size in SIZES:
        for k in KS:
            b, t, rp = pooled(base, size, k), pooled(track, size, k), pooled(rep, TEACHER, k)
            diff = [y - x for x, y in zip(b, t)]
            over = [y - x for x, y in zip(rp, t)]
            rows.append({
                "size": size, "k": k, "exp0002": r4(mean(b)), "track": r4(mean(t)), "replay": r4(mean(rp)),
                "exp0002_per_seed": r4(b), "track_per_seed": r4(t), "replay_per_seed": r4(rp),
                "diff_pp": round(100 * mean(diff), 2), "diff_pp_per_seed": [round(100 * d, 2) for d in diff],
                "diff_pp_ci95": [round(100 * x, 2) for x in boot(lambda i: pick(diff, i))],
                "track_minus_replay_pp": round(100 * mean(over), 2),
                "ratio_exp0002": r4(mean(b) / mean(teach)), "ratio_track": r4(mean(t) / mean(teach)),
                "ratio_track_per_seed": [round(x / y, 4) for x, y in zip(t, teach)],
                "ratio_track_ci95": boot(lambda i: pick(t, i) / pick(teach, i)),
            })
    p = next(r for r in rows if r["size"] == 100_000 and r["k"] == 20)
    if p["ratio_track"] >= 0.5 and p["diff_pp"] >= 15:
        verdict = "supported"
    elif p["diff_pp"] < 5:
        verdict = "refuted"
    else:
        verdict = "neither (inconclusive)"
    rp1, rp20 = mean(pooled(rep, TEACHER, 1)), mean(pooled(rep, TEACHER, 20))
    out = {
        "h1_primary": {"criterion": "100k, k=20: ratio_track >= 0.5 and track - exp0002 >= 15 pp (refuted if < 5 pp)",
                       "verdict": verdict, **p},
        "teacher_h40": r4(mean(teach)), "teacher_h40_per_seed": r4(teach),
        "replay_reading": {"replay_k1": r4(rp1), "replay_k20": r4(rp20), "ratio_k20_k1": r4(rp20 / rp1),
                           "plan_too_stale_open_loop": rp20 < 0.5 * rp1},
        "comparisons": rows,
        "per_suite": {f"{su}|{arm}|{size}|k{k}": r4(mean([d[(su, size, sd, k)] for sd in SEEDS]))
                      for su in SUITES for arm, d, sizes in (("exp0002", base, SIZES), ("track", track, SIZES),
                                                             ("replay", rep, (TEACHER,)))
                      for size in sizes for k in KS},
    }
    print(f"H1 (100k, k=20): {verdict}; track {p['track']:.3f} vs EXP-0002 {p['exp0002']:.3f} "
          f"({p['diff_pp']:+.1f} pp, CI {p['diff_pp_ci95']}); ratio to teacher {p['ratio_track']:.2f}")
    print(f"teacher {out['teacher_h40']:.3f}; replay k=1 {rp1:.3f}, k=20 {rp20:.3f} "
          f"(plan too stale open loop: {out['replay_reading']['plan_too_stale_open_loop']})")
    print(f"{'size':>7} {'k':>3} {'E2':>6} {'track':>6} {'replay':>6} {'diff pp':>8} {'ratio':>6}")
    for r in rows:
        print(f"{r['size']:>7} {r['k']:>3} {r['exp0002']:6.3f} {r['track']:6.3f} {r['replay']:6.3f} "
              f"{r['diff_pp']:+8.1f} {r['ratio_track']:6.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
