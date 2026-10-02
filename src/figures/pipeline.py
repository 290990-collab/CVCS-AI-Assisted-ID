# src/figures/pipeline.py

"""
Figure (F2): the two branches, the fusion and the shared measurement, in one
diagram.

It is a drawing, not a measurement, but the four numbers it shows (the two
embedding widths, the fused width, the size of the shared gallery and the
weight alpha) are READ from the artefacts at draw time — a diagram that says
768 while the files say something else is worse than no diagram.

What it says, left to right: the same plan enters the two branches in two
different forms — the rendered image on one side, the `.mat` record turned into
a graph of rooms on the other. The vision encoder is frozen and is followed by a
whitening whose statistics come from the train split only; the graph encoder is
the only trained part. Both produce a unit-norm vector; the late fusion is the
weighted concatenation of the two. Whatever vector one picks (vision, graph or
fused), it is searched in the SAME gallery with the SAME queries, which is what
makes the three comparable at all.

Usage (CPU, instant):

    python -m src.figures.pipeline
    python -m src.figures.pipeline --lang en
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.figures.style import OKABE_ITO, TEXT_WIDTH_IN, apply_style, save_figure

GALLERY = "results/shared_gallery.json"
SELECT = "results/fusion/select_valid.json"
VISION_QVEC = ("results/queryvec/valid/"
               "vision_pespatial_gem_whiten-train_partial-nowalls-random-f0.5_valid.npz")
GRAPH_QVEC = "results/queryvec/valid/graph_gat_asymrob_partial-random-f0.5_valid.npz"

VISION_C = OKABE_ITO["vermillion"]
GRAPH_C = OKABE_ITO["blue"]
FUSION_C = OKABE_ITO["green"]
SHARED_C = "0.35"

LABELS = {
    "it": {
        "vision_in": "pianta renderizzata\n(immagine PNG)",
        "vision_enc": "encoder congelato\n(pespatial, GeM)",
        "vision_post": "whitening\n(statistiche dal train)",
        "vision_vec": "vettore vision\n{dv}-d, norma 1",
        "graph_in": "record `.mat`\n(stanze, adiacenze)",
        "graph_enc": "GNN GAT\n(allenata, InfoNCE)",
        "graph_vec": "vettore graph\n{dg}-d, norma 1",
        "fusion": "fusione tardiva\n[√α·v ; √(1−α)·g]\nα={alpha:g} → {df}-d",
        "shared": ("Qualunque dei tre vettori si usi, la ricerca è la stessa: FAISS sulla "
                   "gallery condivisa di {n} piante,\nstesse query, stesse esclusioni. "
                   "Misura: la pianta danneggiata ritrova sé stessa (metro) · nDCG per asse "
                   "(descrittivo)."),
        "frozen": "non allenato",
        "trained": "unica parte allenata",
    },
    "en": {
        "vision_in": "rendered plan\n(PNG image)",
        "vision_enc": "frozen encoder\n(pespatial, GeM)",
        "vision_post": "whitening\n(train statistics)",
        "vision_vec": "vision vector\n{dv}-d, unit norm",
        "graph_in": "`.mat` record\n(rooms, adjacencies)",
        "graph_enc": "GAT GNN\n(trained, InfoNCE)",
        "graph_vec": "graph vector\n{dg}-d, unit norm",
        "fusion": "late fusion\n[√α·v ; √(1−α)·g]\nα={alpha:g} → {df}-d",
        "shared": ("Whichever of the three vectors is used, the search is the same: FAISS over "
                   "the shared gallery of {n} plans,\nsame queries, same exclusions. "
                   "Metric: the damaged plan finds itself (main) · per-axis nDCG "
                   "(descriptive)."),
        "frozen": "not trained",
        "trained": "the only trained part",
    },
}


def read_facts() -> dict:
    """The numbers written on the diagram, read from the artefacts themselves."""
    vision = np.load(VISION_QVEC, allow_pickle=True)
    graph = np.load(GRAPH_QVEC, allow_pickle=True)
    dv = int(vision["vectors_final"].shape[1] if "vectors_final" in vision
             else vision["vectors"].shape[1])
    dg = int(graph["vectors"].shape[1])
    gallery = json.loads(Path(GALLERY).read_text(encoding="utf-8"))
    alpha = float(json.loads(Path(SELECT).read_text(encoding="utf-8"))["alpha_star"])
    return {"dv": dv, "dg": dg, "df": dv + dg, "alpha": alpha,
            "n": f"{int(gallery['n_shared']):,}".replace(",", "."),
            "n_raw": int(gallery["n_shared"])}


def _box(ax, x, y, w, h, text, color, *, fill=0.10, fontsize=6.3, weight="normal"):
    """Rounded box with centred text; returns its (left, right, centre y)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2.0",
                                linewidth=1.0, edgecolor=color,
                                facecolor=(*plt.matplotlib.colors.to_rgb(color), fill)))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color="0.15", linespacing=1.45, fontweight=weight)
    return x, x + w, y + h / 2


def _arrow(ax, start, end, color=SHARED_C, elbow: bool = False):
    """Thin arrow between two points; `elbow` turns once instead of going diagonal."""
    style = {"arrowstyle": "-|>", "mutation_scale": 7, "linewidth": 0.9,
             "color": color, "shrinkA": 1, "shrinkB": 1}
    if elbow:
        style["connectionstyle"] = "angle,angleA=0,angleB=90,rad=4"
    ax.add_patch(FancyArrowPatch(start, end, **style))


def _badge(ax, x, y, text, color):
    """Small caption under a box (frozen / trained)."""
    ax.text(x, y, text, ha="center", va="top", fontsize=5.8, color=color, style="italic")


def draw(facts: dict, lang: str, height_in: float):
    """Build the diagram on a 0-100 canvas (no data axes, only boxes and arrows)."""
    text = LABELS[lang]
    apply_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, height_in))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    y_v, y_g, h = 74.0, 40.0, 17.0        # riga vision, riga graph, altezza dei box

    # --- vision: immagine -> encoder congelato -> whitening -> vettore ---
    v1 = _box(ax, 1, y_v, 18, h, text["vision_in"], VISION_C)
    v2 = _box(ax, 21.5, y_v, 18, h, text["vision_enc"], VISION_C)
    v3 = _box(ax, 42, y_v, 17.5, h, text["vision_post"], VISION_C)
    v4 = _box(ax, 62, y_v, 15.5, h, text["vision_vec"].format(dv=facts["dv"]),
              VISION_C, fill=0.22)
    _badge(ax, 30.5, y_v - 1.5, text["frozen"], VISION_C)
    for a, b in ((v1, v2), (v2, v3), (v3, v4)):
        _arrow(ax, (a[1], a[2]), (b[0], b[2]), VISION_C)

    # --- graph: .mat -> GNN allenata -> vettore (un passaggio in meno: niente whitening) ---
    g1 = _box(ax, 1, y_g, 18, h, text["graph_in"], GRAPH_C)
    g2 = _box(ax, 21.5, y_g, 18, h, text["graph_enc"], GRAPH_C)
    g4 = _box(ax, 42, y_g, 15.5, h, text["graph_vec"].format(dg=facts["dg"]),
              GRAPH_C, fill=0.22)
    _badge(ax, 30.5, y_g - 1.5, text["trained"], GRAPH_C)
    _arrow(ax, (g1[1], g1[2]), (g2[0], g2[2]), GRAPH_C)
    _arrow(ax, (g2[1], g2[2]), (g4[0], g4[2]), GRAPH_C)

    # --- fusione: prende i due vettori, sta in mezzo alle due righe ---
    f_box = _box(ax, 81, (y_v + y_g) / 2 - 2, 18, h + 8,
                 text["fusion"].format(alpha=facts["alpha"], df=facts["df"]),
                 FUSION_C, fill=0.16, fontsize=6.2)
    _arrow(ax, (v4[1], v4[2]), (f_box[0], f_box[2] + 5), FUSION_C, elbow=True)
    _arrow(ax, (g4[1], g4[2]), (f_box[0], f_box[2] - 5), FUSION_C, elbow=True)

    # --- cosa hanno in comune: una sola gallery, una sola misura ---
    ax.add_patch(FancyBboxPatch((1, 4), 98, 20, boxstyle="round,pad=0.6,rounding_size=2.0",
                                linewidth=0.9, edgecolor=SHARED_C, facecolor="0.96",
                                linestyle=(0, (4, 3))))
    # Thousands separator per language: 67.405 (it) vs 67,405 (en).
    n = facts["n"] if lang == "it" else f"{facts['n_raw']:,}"
    ax.text(50, 14, text["shared"].format(n=n), ha="center", va="center",
            fontsize=6.4, color="0.20", linespacing=1.5)
    # Un vettore per freccia: vision, graph, fusione. Le colonne sono sfalsate
    # apposta, cosi' ogni freccia scende libera fino alla barra.
    for x, y_from, color in ((69.8, y_v, VISION_C), (49.8, y_g, GRAPH_C),
                             (90, f_box[2] - (h + 8) / 2, FUSION_C)):
        _arrow(ax, (x, y_from), (x, 24.5), color)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--lang", choices=("it", "en"), default="it")
    p.add_argument("--height", type=float, default=2.5, help="altezza in pollici")
    p.add_argument("--name", default="f_pipeline")
    p.add_argument("--out-dir", default="figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    facts = read_facts()
    fig = draw(facts, args.lang, args.height)
    notes = [f"vision {facts['dv']}-d + graph {facts['dg']}-d → fusione {facts['df']}-d",
             f"α = {facts['alpha']:g} (da {SELECT})",
             f"gallery condivisa: {facts['n_raw']} piante"]
    for line in notes:
        print(line)
    save_figure(fig, args.name, [VISION_QVEC, GRAPH_QVEC, GALLERY, SELECT],
                notes=notes, out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
