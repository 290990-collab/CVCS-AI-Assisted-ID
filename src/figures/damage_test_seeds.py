# src/figures/damage_test_seeds.py

"""
Figure: self-recovery under room removal, TEST split, mean over the replicas
(status.md §57-§57.3, 1 Oct 2026). Same picture as `damage_test`, but every
line is the mean over the 4 replicas and the band is ± 1 sd ACROSS replicas.

What it shows
-------------
x: fraction of rooms removed from the query (0, 0.25, 0.5, 0.75).
y: self-recovery MRR. One line per system: fusion (alpha* of each replica,
chosen on its own valid), graph (alpha=0) and vision (alpha=1).

The band answers "would another seed change the picture?": it is the spread
between replicas (graph training + which rooms are removed), NOT the bootstrap
CI over queries of `damage_test`. The two measure different things and are not
combined (§57).

The training-free `hist` baseline exists for the historical replica only (it
has no training and was not rerun per damage seed): it is drawn as that single
run and says so in the legend.

Where the numbers come from
---------------------------
    replica 0:  results/perquery/fusion_test/fusion_a{alpha*,0,1}_partial-nowalls-random-f*_test.npz
    replica S:  results/perquery/seeds/s<S>/fusion_test/fusion_a{alpha*,0,1}_...
    alpha*:     results/fusion/select_valid.json · results/fusion/seeds/s<S>/select_valid.json

The per-replica AUCs recomputed here are checked against
`results/fusion/seeds/summary.json` (written by `seed_summary`): if one
disagrees the figure would show numbers that are not the reported ones, so it stops.

Usage (CPU, seconds):

    python -m src.figures.damage_test_seeds                 # italian labels
    python -m src.figures.damage_test_seeds --lang en --out-dir figures/english
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.robustness_auc import AUC_FRACTIONS, fraction_path, load_auc
from src.figures.style import SYSTEM_STYLE, new_figure, save_figure

CURVE_FRACTIONS = (0.0,) + AUC_FRACTIONS
SPLIT = "test"
STRATEGY = "nowalls-random"
# Replicas: every results/fusion/seeds/s<S>/ with a select_test.json (= replica finished), or --seeds.
SUMMARY = "results/fusion/seeds/summary.json"
BASELINE = "results/perquery/graph_test_B/graph_hist-baseline_base"
MEAN_TOLERANCE = 1e-6

# summary.json column of each system
SUMMARY_KEY = {"fusion": "auc_fusion", "graph": "auc_graph", "vision": "auc_vision"}

LABELS = {
    "it": {
        "x": "frazione di stanze tolte dalla query",
        "y": "MRR di auto-ritrovamento",
        "ceiling": "tetto dei dati\n(escluso dall'AUC)",
        "systems": {"vision": "vision", "graph": "graph", "fusion": "fusione α=0.6"},
        "baseline": "baseline hist",
        "auc": "AUC",
        "band": "media su {n} repliche, banda ± 1 sd fra repliche\nbaseline: una sola run",
    },
    "en": {
        "x": "fraction of rooms removed from the query",
        "y": "self-recovery MRR",
        "ceiling": "data ceiling\n(not in the AUC)",
        "systems": {"vision": "vision", "graph": "graph", "fusion": "fusion α=0.6"},
        "baseline": "hist baseline",
        "auc": "AUC",
        "band": "mean over {n} replicas, band ± 1 sd across replicas\nbaseline: a single run",
    },
}


def discover_seeds(split: str = SPLIT) -> list[int]:
    """Seeds whose replica is finished on `split` (select_<split>.json written)."""
    found = []
    for d in Path("results/fusion/seeds").glob("s*"):
        if d.name[1:].isdigit() and (d / f"select_{split}.json").exists():
            found.append(int(d.name[1:]))
    return sorted(found)


def replica_dirs(seeds) -> list[tuple[int, Path, Path]]:
    """(replica id, fused per-query dir, select_valid json) for every replica."""
    rows = [(0, Path("results/perquery/fusion_test"), Path("results/fusion/select_valid.json"))]
    rows += [(s, Path(f"results/perquery/seeds/s{s}/fusion_test"),
              Path(f"results/fusion/seeds/s{s}/select_valid.json")) for s in seeds]
    return rows


def load_replicas(seeds, split: str = SPLIT) -> tuple[dict, list[int], list[Path]]:
    """Per replica and system: mean self-recovery per fraction and the AUC.

    Returns:
        (per_replica, replica_ids, sources): `per_replica[key]` is a list (one
        entry per replica) of dicts {"curve": {f: mean}, "auc": float}.
    """
    per_replica = {key: [] for key in SUMMARY_KEY}
    ids, sources = [], []
    for rid, fusion_dir, select in replica_dirs(seeds):
        alpha_star = float(json.loads(select.read_text(encoding="utf-8"))["alpha_star"])
        sources.append(select)
        ids.append(rid)
        for key, alpha in (("fusion", alpha_star), ("graph", 0.0), ("vision", 1.0)):
            prefix = fusion_dir / f"fusion_a{alpha:g}"
            data = load_auc(prefix, split=split, fractions=CURVE_FRACTIONS, strategy=STRATEGY)
            auc = float(np.mean(np.stack([data.per_fraction[f] for f in AUC_FRACTIONS]), axis=0).mean())
            per_replica[key].append({
                "curve": {f: float(data.per_fraction[f].mean()) for f in CURVE_FRACTIONS},
                "auc": auc,
                "n": len(data.names),
            })
            sources += [fraction_path(prefix, f, split, STRATEGY) for f in CURVE_FRACTIONS]
    return per_replica, ids, sources


def check_against_summary(per_replica: dict, ids: list[int], split: str) -> None:
    """Stop if a recomputed AUC differs from `seed_summary`'s reported one."""
    rows = json.loads(Path(SUMMARY).read_text(encoding="utf-8"))["rows"]
    reported = {(r["replica"], r["split"]): r for r in rows}
    for key, column in SUMMARY_KEY.items():
        for rid, rep in zip(ids, per_replica[key]):
            ref = reported[(rid, split)][column]
            if abs(rep["auc"] - ref) > MEAN_TOLERANCE:
                raise ValueError(f"replica {rid} {key}: AUC {rep['auc']:.6f} != summary "
                                 f"{ref:.6f}; la figura mostrerebbe numeri non riportati")


def load_baseline(split: str) -> tuple[dict, float, list[Path]]:
    """Historical `hist` baseline: mean per fraction and AUC (single run)."""
    data = load_auc(BASELINE, split=split, fractions=CURVE_FRACTIONS, strategy="random")
    curve = {f: float(data.per_fraction[f].mean()) for f in CURVE_FRACTIONS}
    auc = float(np.mean(np.stack([data.per_fraction[f] for f in AUC_FRACTIONS]), axis=0).mean())
    return curve, auc, [fraction_path(BASELINE, f, split, "random") for f in CURVE_FRACTIONS]


def summarize(per_replica: dict) -> dict:
    """Mean and sd (ddof=1) across replicas, per fraction and for the AUC."""
    stats = {}
    for key, reps in per_replica.items():
        curves = np.asarray([[r["curve"][f] for f in CURVE_FRACTIONS] for r in reps])
        aucs = np.asarray([r["auc"] for r in reps])
        stats[key] = {"mean": curves.mean(0), "sd": curves.std(0, ddof=1),
                      "min": curves.min(0), "max": curves.max(0),
                      "auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std(ddof=1)),
                      "aucs": aucs}
    return stats


def draw(stats: dict, baseline: tuple, n_rep: int, lang: str, height_in: float):
    """Baseline (single run) + three replicated curves with their ± sd band."""
    text = LABELS[lang]
    fig, ax = new_figure("column", height_in=height_in)
    x = np.asarray(CURVE_FRACTIONS, dtype=float)

    curve, auc = baseline
    ax.plot(x, [curve[f] for f in CURVE_FRACTIONS],
            label=f"{text['baseline']} — {text['auc']} {auc:.3f}",
            markerfacecolor="white", markeredgewidth=1.2, **SYSTEM_STYLE["baseline"])

    for key in ("vision", "graph", "fusion"):        # fused line drawn last, on top
        s = stats[key]
        style = SYSTEM_STYLE[key]
        ax.fill_between(x, s["mean"] - s["sd"], s["mean"] + s["sd"], color=style["color"],
                        alpha=0.25, linewidth=0)
        ax.plot(x, s["mean"],
                label=f"{text['systems'][key]} — {text['auc']} {s['auc_mean']:.3f} ± {s['auc_sd']:.3f}",
                markerfacecolor="white", markeredgewidth=1.2, **style)

    ax.axvline(0.0, color="0.55", linewidth=0.6, linestyle=(0, (1, 2)))
    ax.annotate(text["ceiling"], xy=(0.0, 0.62), xytext=(0.035, 0.62),
                fontsize=6, color="0.35", ha="left", va="center", linespacing=1.15)
    ax.annotate(text["band"].format(n=n_rep), xy=(0.99, 0.99), xycoords="axes fraction",
                fontsize=6, color="0.35", ha="right", va="top")

    ax.set_xlabel(text["x"])
    ax.set_ylabel(text["y"])
    ax.set_xticks(list(CURVE_FRACTIONS))
    ax.set_xlim(-0.03, 0.78)
    ax.set_ylim(-0.02, 1.06)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.06), labelspacing=0.3, fontsize=6.3)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--lang", choices=("it", "en"), default="it")
    p.add_argument("--height", type=float, default=2.5, help="altezza in pollici (default: 2.5)")
    p.add_argument("--name", default="f_damage_test_seeds", help="nome base dei file")
    p.add_argument("--out-dir", default="figures", help="cartella di destinazione")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="repliche nuove (default: tutte quelle con select_test.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else discover_seeds(SPLIT)
    per_replica, ids, sources = load_replicas(seeds, SPLIT)
    check_against_summary(per_replica, ids, SPLIT)
    stats = summarize(per_replica)
    b_curve, b_auc, b_sources = load_baseline(SPLIT)
    fig = draw(stats, (b_curve, b_auc), len(ids), args.lang, args.height)

    notes = [f"split: {SPLIT} · repliche: {', '.join(str(i) for i in ids)} (0 = storica)",
             "banda: ± 1 sd (ddof=1) FRA repliche, non CI bootstrap sulle query",
             f"baseline hist: solo replica 0 — AUC {b_auc:.4f}",
             f"AUC verificate contro {SUMMARY} (tolleranza {MEAN_TOLERANCE:g})"]
    for key in ("fusion", "graph", "vision"):
        s = stats[key]
        per_f = "  ".join(f"f={f}: {m:.4f} ± {d:.4f} [{lo:.4f}–{hi:.4f}]"
                          for f, m, d, lo, hi in zip(CURVE_FRACTIONS, s["mean"], s["sd"],
                                                     s["min"], s["max"]))
        aucs = ", ".join(f"{a:.4f}" for a in s["aucs"])
        notes.append(f"{key:7s} AUC {s['auc_mean']:.4f} ± {s['auc_sd']:.4f} (per replica: {aucs}) | {per_f}")
    for line in notes:
        print(line)
    save_figure(fig, args.name, sources + b_sources + [Path(SUMMARY)], notes=notes,
                out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
