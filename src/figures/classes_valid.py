# src/figures/classes_valid.py

"""
Figure (F5): how many plans count as "relevant" for a query, on the two
discrete axes (VALID, 2000 queries against the whole gallery).

Why the figure exists
---------------------
Composition and topology are evaluated against an exact equivalence class: the
plans whose room histogram (composition) or whose typed adjacency (topology) is
IDENTICAL to the query's. The two classes have completely different sizes, and
that single fact explains two caveats of the report:

- composition classes are enormous — the median query has thousands of relevant
  plans out of 67.405, so a ranking hardly has to work to score well and the
  metric saturates;
- topology classes are tiny, and a share of the queries has NO relevant plan at
  all besides itself: those queries are singletons and are skipped by Recall and
  mAP (they would divide by zero), which is why the skipped count is declared.

Where it comes from
-------------------
`num_relevant` of a full-plan per-query file: the count the evaluation itself
used, on the same gallery, with the query excluded (so 0 = singleton). It does
not depend on the system being evaluated, and the figure checks exactly that by
reading the same field from two different systems and refusing to draw if they
disagree.

Usage (CPU, instant):

    python -m src.figures.classes_valid
    python -m src.figures.classes_valid --lang en
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.perquery import load_perquery
from src.evaluation.relevance import DISCRETE_AXES
from src.figures.style import OKABE_ITO, apply_style, new_figure, save_figure

# Stesso file per-query dei numeri del report; il secondo serve solo a verificare
# che il conteggio non dipenda dal sistema.
FULL_FILE = "results/perquery/fusion_valid/fusion_a0.6_full_valid.npz"
CHECK_FILE = "results/perquery/fusion_valid/fusion_a0_full_valid.npz"

AXIS_STYLE = {
    "composition": {"color": OKABE_ITO["sky"], "hatch": None},
    "topology": {"color": OKABE_ITO["purple"], "hatch": "///"},
}

LABELS = {
    "it": {
        "x": "piante rilevanti per query (scala log)",
        "y": "quota delle query (%)",
        "axes": {"composition": "composizione", "topology": "topologia"},
        "median": "mediana {v:.0f}",
        "singleton": ("gallery: {g} piante\n"
                      "classi da una sola pianta (query saltate da Recall/mAP):\n"
                      "composizione {c}/{n} · topologia {t}/{n}"),
    },
    "en": {
        "x": "plans relevant per query (log scale)",
        "y": "share of queries (%)",
        "axes": {"composition": "composition", "topology": "topology"},
        "median": "median {v:.0f}",
        "singleton": ("gallery: {g} plans\n"
                      "classes of one plan only (queries skipped by Recall/mAP):\n"
                      "composition {c}/{n} · topology {t}/{n}"),
    },
}


def load_counts(path: str = FULL_FILE, check: str | None = CHECK_FILE) -> dict:
    """Class size per query on the two discrete axes.

    Returns:
        dict with `counts[axis]` ([Q] ints, 0 = singleton), the number of
        queries and the gallery size.

    Raises:
        ValueError: if the two systems disagree on the counts (they must not:
            the class of a query is a property of the ground truth alone).
    """
    from src.evaluation.relevance import AXES

    data = load_perquery(path)
    counts = {ax: data.num_relevant[AXES.index(ax)].astype(int) for ax in DISCRETE_AXES}
    if check:
        other = load_perquery(check)
        for ax in DISCRETE_AXES:
            if not np.array_equal(counts[ax], other.num_relevant[AXES.index(ax)].astype(int)):
                raise ValueError(
                    f"{ax}: i due file danno classi diverse — il conteggio dei rilevanti "
                    "non può dipendere dal sistema valutato")
    return {"counts": counts, "n_queries": len(data.names),
            "gallery": int(data.meta["gallery"]["n"])}


def draw(loaded: dict, lang: str, height_in: float):
    """Two log-binned histograms, their medians, and the singleton share."""
    text = LABELS[lang]
    counts = loaded["counts"]
    n = loaded["n_queries"]
    fig, ax = new_figure("column", height_in=height_in)

    top = max(int(c.max()) for c in counts.values())
    bins = np.logspace(0, np.log10(top) + 0.05, 26)
    for axis in DISCRETE_AXES:
        values = counts[axis]
        style = AXIS_STYLE[axis]
        drawn = values[values > 0]                  # 0 = singleton: fuori dalla scala log
        weights = np.full(drawn.shape, 100.0 / n)
        ax.hist(drawn, bins=bins, weights=weights, histtype="stepfilled",
                facecolor=(*plt.matplotlib.colors.to_rgb(style["color"]), 0.30),
                edgecolor=style["color"], linewidth=1.1, hatch=style["hatch"],
                label=text["axes"][axis])
        median = float(np.median(values))
        x_med = max(median, 1.0)
        ax.axvline(x_med, color=style["color"], linestyle=(0, (3, 2)), linewidth=0.9)
        # Vicino al bordo destro l'etichetta va scritta verso l'interno, o esce.
        side = "right" if x_med > np.sqrt(top) else "left"
        ax.annotate(text["median"].format(v=median),
                    xy=(x_med * (0.85 if side == "right" else 1.18), 0.99),
                    xycoords=("data", "axes fraction"), fontsize=6,
                    color=style["color"], ha=side, va="top")

    singles = {ax_: int((counts[ax_] == 0).sum()) for ax_ in DISCRETE_AXES}
    ax.set_xscale("log")
    ax.set_xlabel(text["x"])
    ax.set_ylabel(text["y"])
    # Spazio in alto per le mediane e le note: le barre non devono toccarle.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.90), labelspacing=0.3)
    ax.annotate(text["singleton"].format(c=singles["composition"], t=singles["topology"],
                                        n=n, g=f"{loaded['gallery']:,}".replace(",", ".")),
                xy=(0.01, 0.64), xycoords="axes fraction", fontsize=6, color="0.30",
                ha="left", va="top", linespacing=1.4)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--lang", choices=("it", "en"), default="it")
    p.add_argument("--height", type=float, default=2.5)
    p.add_argument("--name", default="f_classes_valid")
    p.add_argument("--out-dir", default="figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_counts()
    fig = draw(loaded, args.lang, args.height)

    notes = [f"query: {loaded['n_queries']} · gallery: {loaded['gallery']}"]
    for axis in DISCRETE_AXES:
        v = loaded["counts"][axis]
        notes.append(f"{axis:12s} mediana {np.median(v):.0f} · media {v.mean():.0f} · "
                     f"max {v.max()} · singleton (0) {int((v == 0).sum())} · "
                     f"quantili 10/90: {np.percentile(v, 10):.0f}/{np.percentile(v, 90):.0f}")
    for line in notes:
        print(line)
    save_figure(fig, args.name, [FULL_FILE, CHECK_FILE], notes=notes, out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
