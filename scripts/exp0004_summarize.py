"""EXP-0004 (hold start): the pre-registered H1 result (README §5) and the reported, non-gating comparisons.

Success inputs are a1_summarize.py outputs: `track` students with --hold-start (eval_hold) and without (eval_log,
the rerun with residual logging), and the H40 teachers' replay with --hold-start (eval_replay_hold). No-hold replay,
the EXP-0003 students and the EXP-0002 baselines come from the committed EXP-0002/0003 results. Pooling, ratio and
bootstrap as EXP-0002/0003. Residuals are read from the per-episode summaries in the students' eval_hold / eval_log
JSONs (--students-dir): step-weighted over episodes, split by success.

Usage: python3 scripts/exp0004_summarize.py <track_hold.json> <track_log.json> <replay_hold.json>
           --students-dir <runs/students> [--out summary.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp0002_summarize import SEEDS, SIZES, SUITES, boot, index, mean, pooled  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXP2, EXP3 = ROOT / "EXP-0002-plan-horizon/results", ROOT / "EXP-0003-tracking-structure/results"
TEACHER = 10_000_000
KS = (10, 20)
RES_KEYS = ("pos_abs_mean", "pos_abs_p95", "rot_abs_mean", "grip_abs_mean", "pos_frac_at_95", "pos_frac_at_50",
            "any_arm_dim_at_95")


def r4(x):
    return [round(v, 4) for v in x] if isinstance(x, list) else round(x, 4)


def residuals(students_dir: Path, eval_dir: str) -> dict:
    """{(size, k, 'all'|'success'|'failure'): step-weighted means of the per-episode residual summaries}."""
    acc = {}
    for f in students_dir.glob(f"*/*/{eval_dir}/*.json"):
        r = json.loads(f.read_text())
        size = r["info"]["size_target"]
        for ep in r["per_episode"]:
            s = ep.get("residual")
            if not s:
                continue
            for split in ("all", "success" if ep["success"] else "failure"):
                a = acc.setdefault((size, r["k"], split), {"n": 0, "episodes": 0, "win": [], **{k: 0.0 for k in RES_KEYS}})
                a["n"] += s["n"]
                a["episodes"] += 1
                a["win"].append(s["pos_abs_mean_per_window"])
                for k in RES_KEYS:
                    a[k] += s[k] * s["n"]
    out = {}
    for key, a in acc.items():
        row = {k: round(a[k] / a["n"], 5) for k in RES_KEYS}
        nw = max(len(w) for w in a["win"])  # per-window mean over episodes that reach the window (unweighted)
        row["pos_abs_mean_per_window"] = [round(mean([w[i] for w in a["win"] if len(w) > i]), 5) for i in range(nw)]
        out[key] = {"steps": a["n"], "episodes": a["episodes"], **row}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track_hold")
    ap.add_argument("track_log")
    ap.add_argument("replay_hold")
    ap.add_argument("--students-dir", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    z = lambda r: r["condition"] == "z-teacher"  # noqa: E731
    teach = pooled(index(load(EXP2 / "teachers_h40.json"), lambda r: r["condition"] == "task-id"), TEACHER, None)
    base = index(load(EXP2 / "students_h40.json"), z)
    track3 = index(load(EXP3 / "students_track.json"), z)
    rep = index(load(EXP3 / "replay.json"), lambda r: True)
    hold = index(load(a.track_hold), z)
    log = index(load(a.track_log), z)
    rep_hold = index(load(a.replay_hold), lambda r: True)
    pick = lambda v, i: mean([v[j] for j in i])  # noqa: E731

    repro = {f"{su}|{size}|s{sd}|k{k}": round(log[(su, size, sd, k)] - track3[(su, size, sd, k)], 4)
             for su in SUITES for size in SIZES for sd in SEEDS for k in KS}
    rows = []
    for size in SIZES:
        for k in KS:
            th, tl, rp, rh, b = (pooled(d, s, k) for d, s in ((hold, size), (log, size), (rep, TEACHER),
                                                              (rep_hold, TEACHER), (base, size)))
            g_no = [x - y for x, y in zip(rp, tl)]
            g_hold = [x - y for x, y in zip(rh, th)]
            rows.append({
                "size": size, "k": k, "track_hold": r4(mean(th)), "track_nohold": r4(mean(tl)),
                "replay_hold": r4(mean(rh)), "replay_nohold": r4(mean(rp)), "exp0002": r4(mean(b)),
                "track_hold_per_seed": r4(th), "track_nohold_per_seed": r4(tl), "replay_hold_per_seed": r4(rh),
                "gap_nohold_pp": round(100 * mean(g_no), 2), "gap_nohold_pp_per_seed": [round(100 * g, 2) for g in g_no],
                "gap_hold_pp": round(100 * mean(g_hold), 2), "gap_hold_pp_per_seed": [round(100 * g, 2) for g in g_hold],
                "gap_hold_pp_ci95": [round(100 * x, 2) for x in boot(lambda i: pick(g_hold, i))],
                "gap_ratio": round(mean(g_hold) / mean(g_no), 4) if mean(g_no) else None,
                "hold_minus_nohold_track_pp": round(100 * (mean(th) - mean(tl)), 2),
                "hold_minus_nohold_replay_pp": round(100 * (mean(rh) - mean(rp)), 2),
                "track_hold_minus_exp0002_pp": round(100 * (mean(th) - mean(b)), 2),
                "ratio_track_hold": r4(mean(th) / mean(teach)),
            })
    p = next(r for r in rows if r["size"] == 100_000 and r["k"] == 20)
    if p["gap_hold_pp"] <= p["gap_nohold_pp"] / 3:
        verdict = "supported"
    elif p["gap_hold_pp"] >= 2 * p["gap_nohold_pp"] / 3:
        verdict = "refuted"
    else:
        verdict = "partial"

    res_hold = residuals(Path(a.students_dir), "eval_hold")
    res_log = residuals(Path(a.students_dir), "eval_log")
    tf = load(EXP3 / "residual_bound.json")
    tf_pos = mean([r["k20"]["pos_abs_mean"] for r in tf if r["size"] == "100k"])
    h = res_hold.get((100_000, 20, "all"), {})
    grows = bool(h) and (h["pos_abs_mean"] >= 2 * tf_pos or h["any_arm_dim_at_95"] >= 0.01)
    fmt = lambda d: {f"{s}|k{k}|{sp}": v for (s, k, sp), v in sorted(d.items())}  # noqa: E731
    out = {
        "h1_primary": {"criterion": "100k, k=20: gap_hold <= gap_nohold/3 supported; >= 2/3 refuted; else partial",
                       "verdict": verdict, **p},
        "teacher_h40": r4(mean(teach)),
        "reproduction_track_nohold_minus_exp0003": {"max_abs": max(abs(v) for v in repro.values()), "per_run": repro},
        "comparisons": rows,
        "per_suite": {f"{su}|{arm}|{size}|k{k}": r4(mean([d[(su, size, sd, k)] for sd in SEEDS]))
                      for su in SUITES for arm, d, sizes in (("track_hold", hold, SIZES), ("track_nohold", log, SIZES),
                                                             ("replay_hold", rep_hold, (TEACHER,)),
                                                             ("replay_nohold", rep, (TEACHER,)))
                      for size in sizes for k in KS},
        "residual_reading": {"criterion": "100k k=20 hold: pos_abs_mean >= 2x teacher-forced or >= 1% of steps at 95%",
                             "teacher_forced_pos_abs_mean": round(tf_pos, 5), "grows_in_closed_loop": grows},
        "residuals_hold": fmt(res_hold), "residuals_nohold": fmt(res_log),
    }
    print(f"H1 (100k, k=20): {verdict}; gap hold {p['gap_hold_pp']:+.1f} pp (CI {p['gap_hold_pp_ci95']}) vs "
          f"no hold {p['gap_nohold_pp']:+.1f} pp; track hold {p['track_hold']:.3f}, replay hold {p['replay_hold']:.3f}")
    print(f"reproduction of EXP-0003 by the no-hold rerun: max |diff| {out['reproduction_track_nohold_minus_exp0003']['max_abs']}")
    print(f"{'size':>7} {'k':>3} {'tr_h':>6} {'tr_n':>6} {'rp_h':>6} {'rp_n':>6} {'gap_h':>7} {'gap_n':>7}")
    for r in rows:
        print(f"{r['size']:>7} {r['k']:>3} {r['track_hold']:6.3f} {r['track_nohold']:6.3f} {r['replay_hold']:6.3f} "
              f"{r['replay_nohold']:6.3f} {r['gap_hold_pp']:+7.1f} {r['gap_nohold_pp']:+7.1f}")
    print(f"residuals 100k k=20 hold: {h.get('pos_abs_mean')} pos mean raw (teacher-forced {tf_pos:.4f}); grows: {grows}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
