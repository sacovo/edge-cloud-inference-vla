"""EXP-0005: merge per-source mixture parts (built with exp0005_build_mixture.py --stats) into one LeRobot dataset.

Each part is <parts>/<source>/{data, mixture_stats.json}; all were built from the same --check file, so dims,
quantiles and gripper signs agree. Writes <out>/data (LeRobot `aggregate_datasets`) and <out>/mixture_stats.json
(the check's stats plus each part's written episode count).

Runs in the VLM env:  uv run --project vlm python scripts/exp0005_merge_mixture.py --parts <dir> --sources ... --out <dir>
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from lerobot.datasets.aggregate import aggregate_datasets

    parts, out = Path(a.parts), Path(a.out)
    stats = None
    for name in a.sources:
        st = json.loads((parts / name / "mixture_stats.json").read_text())
        assert st["sources"][name].get("episodes_written"), f"{name}: part not finished"
        if stats is None:
            stats = {k: v for k, v in st.items() if k != "sources"} | {"sources": {}}
        assert (st["state_dim"], st["action_dim"], st["stats_from"]) == \
            (stats["state_dim"], stats["action_dim"], stats["stats_from"]), f"{name}: built from other stats"
        stats["sources"][name] = st["sources"][name]
    out.mkdir(parents=True, exist_ok=True)
    aggregate_datasets(repo_ids=[f"local/{n}" for n in a.sources], aggr_repo_id=f"local/{out.name}",
                       roots=[parts / n / "data" for n in a.sources], aggr_root=out / "data")
    (out / "mixture_stats.json").write_text(json.dumps(stats, indent=1))
    print("merged", {n: s["episodes_written"] for n, s in stats["sources"].items()}, "->", out)


if __name__ == "__main__":
    main()
