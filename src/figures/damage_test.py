"""
Figure: self-recovery MRR under room removal, TEST split, four systems.
x = fraction of rooms removed (0, 0.25, 0.5, 0.75); y = self-recovery MRR (1 = first position).

Systems: fusion (alpha=0.6, fixed on valid), graph (gat/asymrob), vision (pespatial/gem/whiten), training-free `hist` baseline.
Drawn on the queries present in all four systems. f=0.0 is the data ceiling (RPLAN duplicates), excluded from the AUC.

Per-query files already on disk, nothing recomputed:
    results/perquery/fusion_test/fusion_a{0.6,0,1}_partial-nowalls-random-f*_test.npz
    results/perquery/graph_test_B/graph_hist-baseline_base_partial-random-f*_test.npz
alpha=1 is the vision branch and alpha=0 the graph branch, so the fused files keep all systems on the same queries and removed rooms.

Usage (CPU, seconds):

    python -m src.figures.damage_test              # italian labels
    python -m src.figures.damage_test --lang en
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.evaluation.robustness_auc import AUC_FRACTIONS, fraction_path, load_auc
from src.figures.style import SYSTEM_STYLE, mean_ci, new_figure, save_figure

# f=0.0 is drawn but not in the AUC
CURVE_FRACTIONS = (0.0,) + AUC_FRACTIONS
SPLIT = "test"


@dataclass(frozen=True)
class System:
    """One line of the figure: where its per-query files are and how it is drawn."""

    key: str            # style key in SYSTEM_STYLE
    prefix: str         # per-query prefix, as robustness_auc expects it
    strategy: str       # damage label inside the file names
    label_it: str
    label_en: str


# plotting order: fused line last, on top
SYSTEMS = (
    System("baseline", "results/perquery/graph_test_B/graph_hist-baseline_base",
           "random", "baseline hist", "hist baseline"),
    System("vision", "results/perquery/fusion_test/fusion_a1",
           "nowalls-random", "vision", "vision"),
    System("graph", "results/perquery/fusion_test/fusion_a0",
           "nowalls-random", "graph", "graph"),
    System("fusion", "results/perquery/fusion_test/fusion_a0.6",
           "nowalls-random", "fusione α=0.6", "late fusion α=0.6"),
)

LABELS = {
    "it": {
        "x": "frazione di stanze tolte dalla query",
        "y": "MRR di auto-ritrovamento",
        "ceiling": "tetto dei dati\n(escluso dall'AUC)",
        "auc": "AUC",
    },
    "en": {
        "x": "fraction of rooms removed from the query",
        "y": "self-recovery MRR",
        "ceiling": "data ceiling\n(not in the AUC)",
        "auc": "AUC",
    },
}


def load_systems(split: str = SPLIT) -> tuple[dict, list[str], list[Path]]:
    """Per-query curves of the four systems paired by query name.

    Returns (curves, names, sources): `curves[key][f]` is the [P] vector aligned to `names`.
    Raises FileNotFoundError on a missing fraction, ValueError if no query is shared.
    """
    data, sources = {}, []
    for system in SYSTEMS:
        data[system.key] = load_auc(system.prefix, split=split,
                                    fractions=CURVE_FRACTIONS, strategy=system.strategy)
        sources += [fraction_path(system.prefix, f, split, system.strategy)
                    for f in CURVE_FRACTIONS]

    common = set.intersection(*[set(d.names.tolist()) for d in data.values()])
    if not common:
        raise ValueError("i sistemi non hanno nessuna query in comune")
    # order of the first system
    names = [n for n in data[SYSTEMS[0].key].names.tolist() if n in common]

    curves = {}
    for key, d in data.items():
        index = {n: i for i, n in enumerate(d.names.tolist())}
        rows = [index[n] for n in names]
        curves[key] = {f: d.per_fraction[f][rows] for f in CURVE_FRACTIONS}
    return curves, names, sources


def summarize(curves: dict) -> dict:
    """`stats[key]` with `mean`/`lo`/`hi` per fraction (95% CI) and `auc` (mean over f > 0)."""
    stats = {}
    for key, per_fraction in curves.items():
        points = {f: mean_ci(per_fraction[f]) for f in CURVE_FRACTIONS}
        auc_per_query = np.mean(np.stack([per_fraction[f] for f in AUC_FRACTIONS]), axis=0)
        stats[key] = {
            "mean": {f: points[f][0] for f in CURVE_FRACTIONS},
            "lo": {f: points[f][1] for f in CURVE_FRACTIONS},
            "hi": {f: points[f][2] for f in CURVE_FRACTIONS},
            "auc": float(auc_per_query.mean()),
        }
    return stats


def draw(stats: dict, lang: str, height_in: float):
    """Draw the four curves with their confidence bands."""
    text = LABELS[lang]
    fig, ax = new_figure("column", height_in=height_in)
    x = np.asarray(CURVE_FRACTIONS, dtype=float)

    for system in SYSTEMS:
        s = stats[system.key]
        label = system.label_it if lang == "it" else system.label_en
        mean = np.asarray([s["mean"][f] for f in CURVE_FRACTIONS])
        lo = np.asarray([s["lo"][f] for f in CURVE_FRACTIONS])
        hi = np.asarray([s["hi"][f] for f in CURVE_FRACTIONS])
        style = SYSTEM_STYLE[system.key]
        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.18, linewidth=0)
        ax.plot(x, mean, label=f"{label} — {text['auc']} {s['auc']:.3f}",
                markerfacecolor="white", markeredgewidth=1.2, **style)

    ax.axvline(0.0, color="0.55", linewidth=0.6, linestyle=(0, (1, 2)))
    ax.annotate(text["ceiling"], xy=(0.0, 0.62), xytext=(0.035, 0.62),
                fontsize=6, color="0.35", ha="left", va="center", linespacing=1.15)

    ax.set_xlabel(text["x"])
    ax.set_ylabel(text["y"])
    ax.set_xticks(list(CURVE_FRACTIONS))
    ax.set_xlim(-0.03, 0.78)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.06), labelspacing=0.35)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--lang", choices=("it", "en"), default="it",
                   help="lingua delle etichette (default: it)")
    p.add_argument("--split", default=SPLIT, help="split dei file per-query (default: test)")
    p.add_argument("--height", type=float, default=2.5, help="altezza in pollici (default: 2.5)")
    p.add_argument("--name", default="f_damage_test", help="nome base dei file in figures/")
    p.add_argument("--out-dir", default="figures", help="cartella di destinazione")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    curves, names, sources = load_systems(args.split)
    stats = summarize(curves)
    fig = draw(stats, args.lang, args.height)

    notes = [f"query appaiate: {len(names)}", f"split: {args.split}",
             f"frazioni: {', '.join(str(f) for f in CURVE_FRACTIONS)} "
             f"(AUC su {', '.join(str(f) for f in AUC_FRACTIONS)})"]
    for system in SYSTEMS:
        s = stats[system.key]
        per_f = "  ".join(f"f={f}: {s['mean'][f]:.4f} [{s['lo'][f]:.4f}, {s['hi'][f]:.4f}]"
                          for f in CURVE_FRACTIONS)
        notes.append(f"{system.key:8s} AUC {s['auc']:.4f} | {per_f}")
    for line in notes:
        print(line)
    save_figure(fig, args.name, sources, notes=notes, out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
