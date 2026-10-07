"""
Figure: robustness AUC vs fusion weight alpha, VALID, for three fused pairs.
x = weight of the first model of the pair; y = AUC of self-recovery under room removal.

Curves: vision+graph (different models and information), vision+vision (same information),
graph+graph (same model trained twice). The annotated height is the gain of the best alpha over the best end.

Means come from `results/fusion/select*_valid.json`; bands from the per-query files of each alpha.
Means recomputed from the files must match the json, otherwise the script stops.

Usage (CPU; ~1 min with the bands, seconds with `--no-ci`):

    python -m src.figures.alpha_valid
    python -m src.figures.alpha_valid --lang en --no-ci
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.evaluation.robustness_auc import AUC_FRACTIONS, fraction_path, load_auc
from src.evaluation.significance import BOOTSTRAP_B
from src.figures.style import PAIR_STYLE, mean_ci, new_figure, save_figure

SPLIT = "valid"
MEAN_TOLERANCE = 1e-6      # json vs files


@dataclass(frozen=True)
class Pair:
    """One fused pair: its selection file, its damage label and its two ends."""

    key: str            # style key in PAIR_STYLE
    select: str         # json written by fusion_select
    strategy: str       # damage label inside the per-query file names
    label_it: str
    label_en: str


PAIRS = (
    Pair("graph-graph", "results/fusion/select_graphgraph_valid.json", "random",
         "graph + graph (stesso modello)",
         "graph + graph (same model, twice)"),
    Pair("vision-vision", "results/fusion/select_visionvision_valid.json", "nowalls-random",
         "vision + vision (stessa informazione)",
         "vision + vision (same information)"),
    Pair("vision-graph", "results/fusion/select_valid.json", "nowalls-random",
         "vision + graph (informazione diversa)",
         "vision + graph (different information)"),
)

LABELS = {
    "it": {
        "x": "α — peso del primo modello della coppia",
        "y": "AUC di robustezza (stanze tolte)",
        "oracle": "oracolo (vision+graph): miglior ramo per query",
        "gain": "guadagno sul miglior estremo",
    },
    "en": {
        "x": "α — weight of the first model of the pair",
        "y": "robustness AUC (rooms removed)",
        "oracle": "oracle (vision+graph): best branch per query",
        "gain": "gain over the best end",
    },
}


def load_pair(pair: Pair, with_ci: bool, boot: int) -> dict:
    """Alphas, means, peak and optional CI bands of one pair.

    Means come from the selection json; with `with_ci` they are recomputed from the per-query files and must agree.
    Returns `alphas`, `mean`, `lo`, `hi` (empty without CI), `alpha_star`, `star_auc`, `best_end`, `gain`, `oracle`, `sources`.
    """
    select = json.loads(Path(pair.select).read_text(encoding="utf-8"))
    alphas = [float(a) for a in select["alphas"]]
    means = np.asarray([float(select["auc_means"][f"{a:g}"]) for a in alphas])
    sources = [pair.select]

    lo = hi = np.zeros(0)
    if with_ci:
        lo_list, hi_list = [], []
        fusion_dir = Path(select["fusion_dir"])
        for alpha, reported in zip(alphas, means):
            prefix = fusion_dir / f"fusion_a{alpha:g}"
            data = load_auc(prefix, split=select["split"], strategy=pair.strategy)
            mean, a_lo, a_hi = mean_ci(data.auc, b=boot)
            if abs(mean - reported) > MEAN_TOLERANCE:
                raise ValueError(
                    f"{pair.key} α={alpha:g}: la media dai file ({mean:.6f}) non coincide "
                    f"con quella del json ({reported:.6f}); la figura mostrerebbe "
                    f"numeri diversi da quelli del report")
            lo_list.append(a_lo)
            hi_list.append(a_hi)
            sources += [str(fraction_path(prefix, f, select["split"], pair.strategy))
                        for f in AUC_FRACTIONS]
        lo, hi = np.asarray(lo_list), np.asarray(hi_list)

    alpha_star = float(select["alpha_star"])
    star_auc = float(select["auc_means"][f"{alpha_star:g}"])
    # reference = better end, not the average
    best_end = float(max(means[0], means[-1]))
    oracle = (select.get("oracle") or {}).get("oracle_mean")

    return {"alphas": np.asarray(alphas), "mean": means, "lo": lo, "hi": hi,
            "alpha_star": alpha_star, "star_auc": star_auc, "best_end": best_end,
            "gain": star_auc - best_end, "oracle": oracle, "sources": sources}


def draw(curves: dict, lang: str, height_in: float, show_oracle: bool):
    """Three curves with the chosen alpha and its gain over the best end."""
    text = LABELS[lang]
    fig, ax = new_figure("column", height_in=height_in)

    for pair in PAIRS:
        c = curves[pair.key]
        style = PAIR_STYLE[pair.key]
        label = pair.label_it if lang == "it" else pair.label_en
        if c["lo"].size:
            ax.fill_between(c["alphas"], c["lo"], c["hi"], color=style["color"],
                            alpha=0.16, linewidth=0)
        ax.plot(c["alphas"], c["mean"], label=label, markerfacecolor="white",
                markeredgewidth=1.0, markersize=3.2, **style)
        # chosen alpha and its height above the best end
        ax.plot([c["alpha_star"]], [c["star_auc"]], marker=style["marker"],
                color=style["color"], markersize=5.0, linestyle="none")
        ax.annotate("", xy=(c["alpha_star"], c["star_auc"]),
                    xytext=(c["alpha_star"], c["best_end"]),
                    arrowprops={"arrowstyle": "<->", "color": style["color"],
                                "linewidth": 0.7, "shrinkA": 0, "shrinkB": 0})
        ax.annotate(f"+{c['gain']:.3f}",
                    xy=(c["alpha_star"] + 0.025, (c["star_auc"] + c["best_end"]) / 2),
                    fontsize=6, color=style["color"], ha="left", va="center")

    if show_oracle:
        oracle = curves["vision-graph"]["oracle"]
        if oracle is not None:
            ax.axhline(oracle, color="0.45", linewidth=0.7, linestyle=(0, (4, 3)))
            ax.annotate(text["oracle"], xy=(0.02, oracle + 0.008), fontsize=6,
                        color="0.35", ha="left", va="bottom")

    ax.set_xlabel(text["x"])
    ax.set_ylabel(text["y"])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.325, 0.655)
    ax.legend(loc="lower center", bbox_to_anchor=(0.52, -0.01), labelspacing=0.3,
              handlelength=1.8, handletextpad=0.5)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--lang", choices=("it", "en"), default="it",
                   help="lingua delle etichette (default: it)")
    p.add_argument("--no-ci", action="store_true",
                   help="salta le bande di confidenza (non legge i file per-query)")
    p.add_argument("--boot", type=int, default=BOOTSTRAP_B,
                   help=f"ricampionamenti del bootstrap (default: {BOOTSTRAP_B})")
    p.add_argument("--no-oracle", action="store_true",
                   help="non disegnare la linea dell'oracolo")
    p.add_argument("--height", type=float, default=2.6, help="altezza in pollici")
    p.add_argument("--name", default="f_alpha_valid", help="nome base dei file in figures/")
    p.add_argument("--out-dir", default="figures", help="cartella di destinazione")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    curves, sources = {}, []
    for pair in PAIRS:
        curves[pair.key] = load_pair(pair, with_ci=not args.no_ci, boot=args.boot)
        sources += curves[pair.key]["sources"]
    fig = draw(curves, args.lang, args.height, show_oracle=not args.no_oracle)

    notes = [f"split: {SPLIT}", f"bande: {'no' if args.no_ci else f'bootstrap B={args.boot}'}"]
    for pair in PAIRS:
        c = curves[pair.key]
        notes.append(
            f"{pair.key:13s} α*={c['alpha_star']:g} AUC {c['star_auc']:.4f} | "
            f"estremi {c['mean'][0]:.4f} (α=0) / {c['mean'][-1]:.4f} (α=1) | "
            f"guadagno sul migliore +{c['gain']:.4f}"
            + (f" | oracolo {c['oracle']:.4f}" if c["oracle"] is not None else ""))
    for line in notes:
        print(line)
    save_figure(fig, args.name, sources, notes=notes, out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
