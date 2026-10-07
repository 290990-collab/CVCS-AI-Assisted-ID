"""
Multi-seed replica summary of the late fusion (CPU, read-only on its inputs).

One row per (split, replica). Replica 0 = historical run (graph gat/asymrob, damage seed 42);
replica S = seed replica run with S (graph gat/asymrob_s<S>, damage seed S, same query sample).
Per row: alpha* (`alpha_star` of the select json), auc_graph / auc_fusion / auc_vision
(`auc_means` at alpha 0 / alpha* / 1), r_vision (`robustness_auc.load_robust_auc`),
best_epoch (graph `training_summary.json`).

Sources: replica 0 from results/fusion/select_<split>.json and results/perquery/{vision_damage_valid_B | vision_test_B};
replica S from results/fusion/seeds/s<S>/select_<split>.json and results/perquery/seeds/s<S>/fusion_branches_<split>.

Mean +- sd (ddof=1) and min-max per split over all replicas and over the new ones only. A missing input
is reported (row kept, value None), never a crash.

Usage:

    python -m src.evaluation.seed_summary
    python -m src.evaluation.seed_summary --seeds 100042 200042 --splits valid
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from src.evaluation.robustness_auc import load_robust_auc

# seeds/s<digits> folders found on disk are added
DEFAULT_SEEDS = (100042, 200042, 300042)
SPLITS = ("valid", "test")
METRICS = ("alpha_star", "auc_graph", "auc_fusion", "auc_vision", "r_vision", "best_epoch")
VISION_TAG = "vision_pespatial_gem_whiten-train"
HIST_R_DIRS = {"valid": "results/perquery/vision_damage_valid_B",
               "test": "results/perquery/vision_test_B"}
GRAPH_ENCODER_DIR = "embeddings/graph/gat"


def replica_paths(seed: int, split: str, root: str | Path = ".") -> dict:
    """Input paths of one replica (seed 0 = historical)."""
    root = Path(root)
    if seed == 0:
        return {"select": root / f"results/fusion/select_{split}.json",
                "fusion_dir": f"results/perquery/fusion_{split}",
                "vision_prefix": root / HIST_R_DIRS[split] / VISION_TAG,
                "summary": root / GRAPH_ENCODER_DIR / "asymrob" / "training_summary.json",
                "variant": "asymrob"}
    variant = f"asymrob_s{seed}"
    return {"select": root / f"results/fusion/seeds/s{seed}/select_{split}.json",
            "fusion_dir": f"results/perquery/seeds/s{seed}/fusion_{split}",
            "vision_prefix": root / f"results/perquery/seeds/s{seed}/fusion_branches_{split}" / VISION_TAG,
            "summary": root / GRAPH_ENCODER_DIR / variant / "training_summary.json",
            "variant": variant}


def discover_seeds(root: str | Path = ".") -> list[int]:
    """Seeds with a results/fusion/seeds/s<S>/ folder or a gat/asymrob_s<S>/ checkpoint folder."""
    root = Path(root)
    found = set()
    for d in (root / "results/fusion/seeds").glob("s*"):
        m = re.fullmatch(r"s(\d+)", d.name)
        if m and d.is_dir():
            found.add(int(m.group(1)))
    for d in (root / GRAPH_ENCODER_DIR).glob("asymrob_s*"):
        m = re.fullmatch(r"asymrob_s(\d+)", d.name)
        if m and d.is_dir():
            found.add(int(m.group(1)))
    return sorted(found)


def collect_row(seed: int, split: str, root: str | Path = ".") -> dict:
    """Metrics of one (replica, split); missing inputs -> None + a note in `missing`."""
    p = replica_paths(seed, split, root)
    row = {"replica": seed, "split": split, "variant": p["variant"],
           **{m: None for m in METRICS}, "missing": [], "warnings": []}

    if p["select"].exists():
        sel = json.loads(p["select"].read_text())
        star = float(sel["alpha_star"])
        means = sel["auc_means"]
        row["alpha_star"] = star
        for key, k in (("auc_graph", "0"), ("auc_fusion", f"{star:g}"), ("auc_vision", "1")):
            if k in means:
                row[key] = float(means[k])
            else:
                row["missing"].append(f"{key} (alpha={k} not in {p['select']})")
        if sel.get("fusion_dir") != p["fusion_dir"]:
            row["warnings"].append(f"select fusion_dir {sel.get('fusion_dir')!r} "
                                   f"!= expected {p['fusion_dir']!r}")
        row["n_queries"] = sel.get("n_queries")
    else:
        row["missing"].append(f"select json {p['select']}")

    try:
        row["r_vision"] = float(load_robust_auc(p["vision_prefix"], split).mean)
    except (FileNotFoundError, ValueError) as exc:
        row["missing"].append(f"r_vision ({exc})")

    if p["summary"].exists():
        row["best_epoch"] = json.loads(p["summary"].read_text()).get("best_epoch")
    else:
        row["missing"].append(f"training summary {p['summary']}")
    return row


def describe(values) -> dict:
    """n, mean, sd (ddof=1; None with n<2), min, max of the non-None values."""
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if len(x) == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {"n": int(len(x)), "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
            "min": float(x.min()), "max": float(x.max())}


def summarize(rows: list[dict]) -> dict:
    """Per split: stats over all replicas and over the new ones (replica != 0)."""
    out = {}
    for split in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == split]
        out[split] = {
            group: {m: describe([r[m] for r in sub]) for m in METRICS}
            for group, sub in (("all", rs), ("new_only", [r for r in rs if r["replica"] != 0]))
        }
    return out


def _fmt(v, nd=4) -> str:
    if v is None:
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def print_report(rows: list[dict], stats: dict) -> None:
    cols = ("alpha_star", "auc_graph", "auc_fusion", "auc_vision", "r_vision", "best_epoch")
    for split, groups in stats.items():
        print(f"\n=== split {split} · AUC nowalls-random (f=0.25/0.5/0.75) · R = nowalls-random+crop+patch ===")
        print(f"{'replica':>8} {'variant':<18} " + " ".join(f"{c:>10}" for c in cols))
        for r in (r for r in rows if r["split"] == split):
            print(f"{r['replica']:>8} {r['variant']:<18} "
                  + " ".join(f"{_fmt(r[c]):>10}" for c in cols))
            for m in r["missing"]:
                print(f"{'':>8} mancante: {m}")
            for w in r["warnings"]:
                print(f"{'':>8} ⚠️ {w}")
        for group, st in groups.items():
            print(f"{group:>8} {'mean ± sd':<18} " + " ".join(
                f"{_fmt(st[c]['mean']):>10}" for c in cols))
            print(f"{'':>8} {'':<18} " + " ".join(
                f"{('±' + _fmt(st[c]['sd'])) if st[c]['sd'] is not None else '—':>10}" for c in cols))
            print(f"{'':>8} {'min–max':<18} " + " ".join(
                f"{(_fmt(st[c]['min'], 3) + '–' + _fmt(st[c]['max'], 3)) if st[c]['n'] else '—':>10}"
                for c in cols))
            print(f"{'':>8} {'n':<18} " + " ".join(f"{st[c]['n']:>10}" for c in cols))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed replica summary of the late fusion")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help=f"new replicas (default: {list(DEFAULT_SEEDS)} + seeds/s<S> found on disk); "
                        "replica 0 (historical) is always included")
    p.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    p.add_argument("--root", default=".", help="project root (default: cwd)")
    p.add_argument("--out-json", default="results/fusion/seeds/summary.json", dest="out_json")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    seeds = args.seeds if args.seeds is not None else sorted(
        set(DEFAULT_SEEDS) | set(discover_seeds(args.root)))
    seeds = [0] + [s for s in seeds if s != 0]
    rows = [collect_row(s, split, args.root) for split in args.splits for s in seeds]
    stats = summarize(rows)
    print_report(rows, stats)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "stats": stats, "seeds": seeds,
                               "metrics": list(METRICS)}, indent=2, sort_keys=True))
    print(f"\n[seed_summary] scritto {out}")


if __name__ == "__main__":
    main()
