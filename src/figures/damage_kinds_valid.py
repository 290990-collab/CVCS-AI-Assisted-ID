"""
Figure: the vision branch under three damages, and the walls artefact (VALID).
Left: one line per damage (rooms removed with walls, crop, scattered ViT patches) for the frozen config `pespatial/gem/whiten`; their mean is R.
Right: on `dinov3/natural/whiten`, same rooms removed with walls left vs erased.

Nothing is recomputed: per-query files of wave 2 (`results/perquery/vision_damage_valid_B`) are read; the removed
areas go in the provenance file (a fraction of rooms is not a fraction of pixels).

Usage (CPU, seconds):

    python -m src.figures.damage_kinds_valid
    python -m src.figures.damage_kinds_valid --lang en
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.robustness_auc import (
    AUC_FRACTIONS,
    ROBUST_STRATEGIES,
    fraction_path,
    load_auc,
)
from src.figures.style import (
    OKABE_ITO,
    TEXT_WIDTH_IN,
    apply_style,
    mean_ci,
    save_figure,
)

SPLIT = "valid"
DAMAGE_DIR = "results/perquery/vision_damage_valid_B"
CONFIG = f"{DAMAGE_DIR}/vision_pespatial_gem_whiten-train"       # config of record
WALLS_CONFIG = f"{DAMAGE_DIR}/vision_dinov3_natural_whiten-train"  # only config with both curves

# left panel: three damages
DAMAGE_STYLE = {
    "nowalls-random": {"color": OKABE_ITO["purple"], "linestyle": "-", "marker": "o"},
    "crop": {"color": OKABE_ITO["sky"], "linestyle": "--", "marker": "s"},
    "patch": {"color": OKABE_ITO["orange"], "linestyle": "-.", "marker": "^"},
}
# right panel: walls left vs erased
WALLS_STYLE = {
    "random": {"color": "0.45", "linestyle": ":", "marker": "x"},
    "nowalls-random": {"color": OKABE_ITO["purple"], "linestyle": "-", "marker": "o"},
}

LABELS = {
    "it": {
        "x": "frazione di danno (f)",
        "y": "MRR di auto-ritrovamento",
        "damages": {"nowalls-random": "stanze tolte", "crop": "ritaglio", "patch": "toppe sparse"},
        "walls": {"random": "muri della stanza lasciati", "nowalls-random": "muri cancellati"},
        "title_left": "un encoder, tre danni: `pespatial/gem/whiten`",
        "title_right": "stesse stanze tolte: `dinov3/natural/whiten`",
        "auc": "AUC",
        "robust": "media dei tre danni (R) = {r:.3f}",
        "gap": "quanto valevano\ni muri rimasti",
    },
    "en": {
        "x": "damage fraction (f)",
        "y": "self-recovery MRR",
        "damages": {"nowalls-random": "rooms removed", "crop": "crop", "patch": "scattered patches"},
        "walls": {"random": "room walls left", "nowalls-random": "walls erased"},
        "title_left": "one encoder, three damages: `pespatial/gem/whiten`",
        "title_right": "same rooms removed: `dinov3/natural/whiten`",
        "auc": "AUC",
        "robust": "mean of the three damages (R) = {r:.3f}",
        "gap": "what the walls\nleft behind were worth",
    },
}


def load_panel(prefix: str, strategies, split: str = SPLIT):
    """Curve per strategy on the queries shared by all; `curves[strategy]` has `mean`/`lo`/`hi` per fraction, `auc`, `area`."""
    data = {s: load_auc(prefix, split=split, strategy=s) for s in strategies}
    common = set.intersection(*[set(d.names.tolist()) for d in data.values()])
    if not common:
        raise ValueError(f"{prefix}: i danni non hanno query in comune")
    names = [n for n in data[strategies[0]].names.tolist() if n in common]

    curves, sources = {}, []
    for strategy, d in data.items():
        index = {n: i for i, n in enumerate(d.names.tolist())}
        rows = [index[n] for n in names]
        points = {f: mean_ci(d.per_fraction[f][rows]) for f in AUC_FRACTIONS}
        auc = float(np.mean([points[f][0] for f in AUC_FRACTIONS]))
        curves[strategy] = {"mean": {f: points[f][0] for f in AUC_FRACTIONS},
                            "lo": {f: points[f][1] for f in AUC_FRACTIONS},
                            "hi": {f: points[f][2] for f in AUC_FRACTIONS},
                            "auc": auc, "area": _areas(prefix, strategy, split)}
        sources += [str(fraction_path(prefix, f, split, strategy)) for f in AUC_FRACTIONS]
    return curves, names, sources


def _areas(prefix: str, strategy: str, split: str) -> dict:
    """Mean plan area actually removed at each fraction (empty if not recorded)."""
    from src.evaluation.perquery import load_perquery

    out = {}
    for f in AUC_FRACTIONS:
        data = load_perquery(fraction_path(prefix, f, split, strategy))
        if data.area_removed is not None:
            out[f] = float(np.nanmean(data.area_removed))
    return out


def _plot(ax, curves, styles, labels, text):
    """One panel: a line with its band per curve, AUC in the legend."""
    x = np.asarray(AUC_FRACTIONS, dtype=float)
    for key, style in styles.items():
        c = curves[key]
        mean = np.asarray([c["mean"][f] for f in AUC_FRACTIONS])
        lo = np.asarray([c["lo"][f] for f in AUC_FRACTIONS])
        hi = np.asarray([c["hi"][f] for f in AUC_FRACTIONS])
        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.18, linewidth=0)
        ax.plot(x, mean, label=f"{labels[key]} — {text['auc']} {c['auc']:.3f}",
                markerfacecolor="white", markeredgewidth=1.2, **style)
    ax.set_xlabel(text["x"])
    ax.set_xticks(list(AUC_FRACTIONS))
    ax.set_xlim(0.20, 0.80)
    ax.set_ylim(-0.02, 1.02)


def draw(left: dict, right: dict, lang: str, height_in: float):
    """Two panels: the three damages, and the walls artefact."""
    text = LABELS[lang]
    apply_style()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, height_in),
                                     sharey=True)

    _plot(ax_l, left, DAMAGE_STYLE, text["damages"], text)
    ax_l.set_ylabel(text["y"])
    ax_l.set_title(text["title_left"], fontsize=7.5, pad=4)
    r = float(np.mean([left[s]["auc"] for s in ROBUST_STRATEGIES]))
    ax_l.annotate(text["robust"].format(r=r), xy=(0.5, 0.955), xycoords="axes fraction",
                  fontsize=6.5, color="0.30", ha="center", va="top")
    ax_l.legend(loc="lower left", bbox_to_anchor=(-0.01, -0.01), labelspacing=0.3)

    _plot(ax_r, right, WALLS_STYLE, text["walls"], text)
    ax_r.set_title(text["title_right"], fontsize=7.5, pad=4)
    ax_r.legend(loc="lower left", bbox_to_anchor=(-0.01, -0.01), labelspacing=0.3)
    # gap at f=0.5
    mid = 0.5
    top, bottom = right["random"]["mean"][mid], right["nowalls-random"]["mean"][mid]
    ax_r.annotate("", xy=(mid, top), xytext=(mid, bottom),
                  arrowprops={"arrowstyle": "<->", "color": "0.30", "linewidth": 0.8,
                              "shrinkA": 0, "shrinkB": 0})
    ax_r.annotate(f"{text['gap']}\n+{top - bottom:.3f}", xy=(mid - 0.02, (top + bottom) / 2),
                  fontsize=6.5, color="0.30", ha="right", va="center", linespacing=1.25)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--lang", choices=("it", "en"), default="it")
    p.add_argument("--split", default=SPLIT)
    p.add_argument("--height", type=float, default=2.6, help="altezza in pollici")
    p.add_argument("--name", default="f_damage_kinds_valid")
    p.add_argument("--out-dir", default="figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    left, left_names, sources_l = load_panel(CONFIG, ROBUST_STRATEGIES, args.split)
    right, right_names, sources_r = load_panel(
        WALLS_CONFIG, ("random", "nowalls-random"), args.split)
    fig = draw(left, right, args.lang, args.height)

    notes = [f"split: {args.split}",
             f"pannello sinistro: {Path(CONFIG).name} — {len(left_names)} query",
             f"pannello destro:  {Path(WALLS_CONFIG).name} — {len(right_names)} query"]
    for title, curves in (("tre danni", left), ("muri", right)):
        for key, c in curves.items():
            per_f = "  ".join(f"f={f}: {c['mean'][f]:.4f} [{c['lo'][f]:.4f}, {c['hi'][f]:.4f}]"
                              for f in AUC_FRACTIONS)
            area = "  ".join(f"area f={f}: {a:.3f}" for f, a in c["area"].items())
            notes.append(f"{title:9s} {key:15s} AUC {c['auc']:.4f} | {per_f} | {area}")
    notes.append("R (media dei tre danni) = "
                 f"{np.mean([left[s]['auc'] for s in ROBUST_STRATEGIES]):.4f}")
    for line in notes:
        print(line)
    save_figure(fig, args.name, sources_l + sources_r, notes=notes, out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
