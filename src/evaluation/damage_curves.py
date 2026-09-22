# src/evaluation/damage_curves.py

"""
Degradation curves with the REMOVED PLAN AREA on the x axis (vision damage
modes, 11 Sep 2026). CPU only, reads the per-query files.

Why: the three damage families (room removal, rectangular crop, ViT patches)
have different nominal knobs — a room fraction is not a pixel fraction — so
they are compared on what each query actually lost, `area_removed`
(`src/vision/data/vision_damage.py`).

What it computes, for one config prefix (same convention as robustness_auc:
`<prefix>_partial-<strategy>-f<frac>_<split>.npz`):

- window curves: for each area centre c and strategy, the points (query, file)
  with |area - c| <= halfwidth; a query hit by several files (room removal:
  the achieved area varies per query, so two nominal levels can fall in the
  same window) is first averaged PER QUERY, then the mean over queries gets a
  bootstrap CI over queries (`significance.bootstrap_ci`);
- paired delta between two strategies at the same nominal level, paired by query
  NAME, with the same bootstrap (`--delta A B`, default `crop patch`; for the
  walls question it is `--delta random nowalls-random`, where the two runs
  removed the SAME rooms and differ only by the walls left behind);
- achieved-area statistics per nominal level (how far the knob is from the
  area it produced, and how many fallbacks).

Metrics: self_rr (self-recovery) and nDCG@k on composition/topology/geometry.
Files without `area_removed` (written before 11 Sep 2026) cannot be placed on
the area axis: they are skipped with a message.

Usage:

    python -m src.evaluation.damage_curves \\
        --prefix results/perquery/vision_partial_valid_B/vision_dinov3_natural_head \\
        --out results/damage_curves/dinov3_natural_head_valid.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.evaluation.perquery import load_perquery
from src.evaluation.relevance import AXES
from src.evaluation.robustness_auc import _check_same_protocol
from src.evaluation.significance import bootstrap_ci

DAMAGE_STRATEGIES = ("random", "crop", "patch", "nowalls-random")
DEFAULT_CENTERS = (0.25, 0.5, 0.75)
DEFAULT_HALFWIDTH = 0.05
DEFAULT_K = 10
SELF_RR = "self_rr"
# numerical slack on the window edges: area_removed is stored as float32
_EDGE_EPS = 1e-6


def metric_names(k: int = DEFAULT_K) -> tuple[str, ...]:
    return (SELF_RR,) + tuple(f"ndcg{k}_{ax}" for ax in AXES)


@dataclass
class DamageRun:
    """One per-query file: a strategy at one nominal level."""

    strategy: str
    fraction: float
    names: np.ndarray                       # [Q]
    area: np.ndarray                        # [Q] area_removed
    values: dict = field(default_factory=dict)   # metric -> [Q] (NaN = not defined)
    meta: dict = field(default_factory=dict)


def _file_re(strategy: str) -> re.Pattern:
    return re.compile(r"^(?P<prefix>.+)_partial-" + re.escape(strategy)
                      + r"-f(?P<frac>[0-9.]+)_(?P<split>[a-z]+)\.npz$")


def load_runs(prefix, split: str = "valid", strategies=DAMAGE_STRATEGIES,
              k: int = DEFAULT_K, verbose: bool = True) -> list[DamageRun]:
    """All the damage files of `prefix` that carry `area_removed`.

    Raises:
        FileNotFoundError: no usable file.
        ValueError: files of the same prefix with a different protocol/gallery.
    """
    prefix = Path(prefix)
    runs: list[DamageRun] = []
    for strategy in strategies:
        rx = _file_re(strategy)
        for path in sorted(prefix.parent.glob(f"{prefix.name}_partial-{strategy}-f*_{split}.npz")):
            m = rx.match(path.name)
            if m is None or m["prefix"] != prefix.name or m["split"] != split:
                continue
            d = load_perquery(path)
            if d.area_removed is None or d.self_rr is None:
                if verbose:
                    print(f"[damage_curves] skip {path.name}: no area_removed/self_rr "
                          "(file written before the damage modes)")
                continue
            ki = d.k_index(k)
            values = {SELF_RR: np.asarray(d.self_rr, dtype=float)}
            for ax in AXES:
                values[f"ndcg{k}_{ax}"] = np.asarray(d.ndcg[d.axis_index(ax), ki], dtype=float)
            runs.append(DamageRun(strategy=strategy, fraction=float(m["frac"]),
                                  names=np.asarray([str(n) for n in d.names]),
                                  area=np.asarray(d.area_removed, dtype=float),
                                  values=values, meta=d.meta))
    if not runs:
        raise FileNotFoundError(f"no per-query file with area_removed for {prefix} ({split})")
    ref = runs[0]
    for r in runs[1:]:
        _check_same_protocol(ref.meta, r.meta,
                             f"{prefix.name} {ref.strategy} f={ref.fraction} vs "
                             f"{r.strategy} f={r.fraction}")
    return runs


def window_curve(runs, strategy: str, metric: str, centers=DEFAULT_CENTERS,
                 halfwidth: float = DEFAULT_HALFWIDTH) -> list[dict]:
    """Mean of `metric` over queries whose achieved area falls in each window.

    Per query first (a query contributes one value per window, the mean of its
    points inside it), then over queries, with a bootstrap CI over queries.
    """
    rows = []
    for c in centers:
        per_query: dict[str, list[tuple[float, float]]] = {}
        for r in runs:
            if r.strategy != strategy:
                continue
            vals = r.values[metric]
            inside = np.abs(r.area - c) <= halfwidth + _EDGE_EPS
            for n, v, a in zip(r.names[inside], vals[inside], r.area[inside]):
                if not np.isnan(v):
                    per_query.setdefault(str(n), []).append((float(v), float(a)))
        names = sorted(per_query)
        qv = np.asarray([np.mean([v for v, _ in per_query[n]]) for n in names], dtype=float)
        qa = np.asarray([np.mean([a for _, a in per_query[n]]) for n in names], dtype=float)
        lo, hi = bootstrap_ci(qv)
        rows.append({
            "strategy": strategy, "metric": metric, "center": float(c),
            "halfwidth": float(halfwidth), "n_queries": len(names),
            "n_points": int(sum(len(p) for p in per_query.values())),
            "area_mean": float(qa.mean()) if len(qa) else float("nan"),
            "mean": float(qv.mean()) if len(qv) else float("nan"),
            "ci_lo": lo, "ci_hi": hi,
        })
    return rows


def paired_delta(runs, metric: str, a: str = "crop", b: str = "patch") -> list[dict]:
    """Delta a - b at each nominal level both strategies have, paired by name."""
    by_a = {r.fraction: r for r in runs if r.strategy == a}
    by_b = {r.fraction: r for r in runs if r.strategy == b}
    rows = []
    for f in sorted(set(by_a) & set(by_b)):
        ra, rb = by_a[f], by_b[f]
        col_b = {str(n): i for i, n in enumerate(rb.names)}
        xa, xb, aa, ab = [], [], [], []
        for i, n in enumerate(ra.names):
            j = col_b.get(str(n))
            if j is None:
                continue
            va, vb = ra.values[metric][i], rb.values[metric][j]
            if np.isnan(va) or np.isnan(vb):
                continue
            xa.append(va)
            xb.append(vb)
            aa.append(ra.area[i])
            ab.append(rb.area[j])
        diff = np.asarray(xa, dtype=float) - np.asarray(xb, dtype=float)
        lo, hi = bootstrap_ci(diff)
        rows.append({
            "a": a, "b": b, "metric": metric, "fraction": f, "n_pairs": len(diff),
            "mean_a": float(np.mean(xa)) if xa else float("nan"),
            "mean_b": float(np.mean(xb)) if xb else float("nan"),
            "delta": float(diff.mean()) if len(diff) else float("nan"),
            "ci_lo": lo, "ci_hi": hi,
            "area_a": float(np.mean(aa)) if aa else float("nan"),
            "area_b": float(np.mean(ab)) if ab else float("nan"),
        })
    return rows


def area_stats(runs) -> list[dict]:
    """Achieved area per (strategy, nominal level)."""
    rows = []
    for r in sorted(runs, key=lambda r: (DAMAGE_STRATEGIES.index(r.strategy)
                                         if r.strategy in DAMAGE_STRATEGIES else 99, r.fraction)):
        a = r.area
        rows.append({
            "strategy": r.strategy, "fraction": r.fraction, "n": len(a),
            "mean": float(a.mean()) if len(a) else float("nan"),
            "std": float(a.std()) if len(a) else float("nan"),
            "min": float(a.min()) if len(a) else float("nan"),
            "max": float(a.max()) if len(a) else float("nan"),
            "n_fallback": (r.meta.get("damage") or {}).get("n_fallback"),
            "area_space": (r.meta.get("damage") or {}).get("area_space"),
        })
    return rows


CSV_FIELDS = ("strategy", "metric", "center", "halfwidth", "area_mean",
              "mean", "ci_lo", "ci_hi", "n_queries", "n_points")


def write_csv(rows, path) -> None:
    """Window rows -> CSV for the plot (x = area_mean, y = mean, band = CI)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="degradation curves vs removed plan area")
    p.add_argument("--prefix", required=True, help="config prefix, with its directory")
    p.add_argument("--split", default="valid")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--centers", type=float, nargs="+", default=list(DEFAULT_CENTERS))
    p.add_argument("--halfwidth", type=float, default=DEFAULT_HALFWIDTH)
    p.add_argument("--out", default=None, help="CSV of the window curves (optional)")
    p.add_argument("--delta", nargs=2, metavar=("A", "B"), default=["crop", "patch"],
                   help="paired delta A - B at each shared nominal level "
                        "(default: crop patch; walls question: random nowalls-random)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    runs = load_runs(args.prefix, args.split, k=args.k)
    metrics = metric_names(args.k)
    strategies = [s for s in DAMAGE_STRATEGIES if any(r.strategy == s for r in runs)]
    g = runs[0].meta.get("gallery", {})
    print(f"[damage_curves] {Path(args.prefix).name} · split={args.split} · "
          f"gallery n={g.get('n')} sha1={str(g.get('sha1'))[:12]} · "
          f"windows {args.centers} ±{args.halfwidth}")

    print("\nAchieved area per nominal level:")
    for s in area_stats(runs):
        print(f"  {s['strategy']:<7} f={s['fraction']:<5} n={s['n']:<5} "
              f"area {s['mean']:.3f}±{s['std']:.3f} [{s['min']:.3f}, {s['max']:.3f}] "
              f"fallback={s['n_fallback']} space={s['area_space']}")

    rows = []
    for m in metrics:
        print(f"\n{m} vs removed area (mean over queries, bootstrap 95% CI):")
        for s in strategies:
            for row in window_curve(runs, s, m, args.centers, args.halfwidth):
                rows.append(row)
                print(f"  {s:<7} c={row['center']:<5} area={row['area_mean']:.3f} "
                      f"{row['mean']:.4f} [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] "
                      f"n={row['n_queries']}")
        for d in paired_delta(runs, m, a=args.delta[0], b=args.delta[1]):
            print(f"  {d['a']} − {d['b']} f={d['fraction']}: {d['delta']:+.4f} "
                  f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] n={d['n_pairs']} "
                  f"(area {d['area_a']:.3f} vs {d['area_b']:.3f})")

    if args.out:
        write_csv(rows, args.out)
        print(f"\n[damage_curves] CSV -> {args.out}")


if __name__ == "__main__":
    main()
