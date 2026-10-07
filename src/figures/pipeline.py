"""
Figure: pipeline diagram of the two branches, the late fusion and the shared measurement.
A drawing, but its numbers (embedding widths, fused width, shared gallery size, alpha) are read from the artefacts at draw time.

Plan image -> frozen vision encoder + whitening (train statistics); `.mat` record -> graph -> trained graph encoder.
Both give unit-norm vectors, fused by weighted concatenation. Every vector is searched in the same gallery with the same queries.

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
# `--setup reset`: final pipeline (graph W = gat/comb, main fusion)
SETUPS = {
    "storico": (SELECT, GRAPH_QVEC),
    "reset": ("results/final_pipeline/fusion/s42/select_valid.json",
              "results/final_pipeline/queryvec/valid/s42/graph_gat_rg_comb_s42_partial-random-f0.5_valid.npz"),
}

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


def _readme_arrow(ax, start, end, color):
    """README arrow; endpoints are already moved outside the border (`_box` pads it by 0.6), no extra shrink."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>,head_length=0.55,head_width=0.28", "linewidth": 1.4,
                            "color": color, "shrinkA": 0, "shrinkB": 0, "joinstyle": "miter",
                            "capstyle": "butt"})


def _badge(ax, x, y, text, color):
    """Small caption under a box (frozen / trained)."""
    ax.text(x, y, text, ha="center", va="top", fontsize=5.8, color=color, style="italic")


def draw(facts: dict, lang: str, height_in: float, width_in: float = TEXT_WIDTH_IN):
    """Build the diagram on a 0-100 canvas (no data axes, only boxes and arrows)."""
    text = LABELS[lang]
    apply_style()
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    y_v, y_g, h = 74.0, 40.0, 17.0        # vision row, graph row, box height

    # --- vision branch ---
    v1 = _box(ax, 1, y_v, 18, h, text["vision_in"], VISION_C)
    v2 = _box(ax, 21.5, y_v, 18, h, text["vision_enc"], VISION_C)
    v3 = _box(ax, 42, y_v, 17.5, h, text["vision_post"], VISION_C)
    v4 = _box(ax, 62, y_v, 15.5, h, text["vision_vec"].format(dv=facts["dv"]),
              VISION_C, fill=0.22)
    _badge(ax, 30.5, y_v - 1.5, text["frozen"], VISION_C)
    for a, b in ((v1, v2), (v2, v3), (v3, v4)):
        _arrow(ax, (a[1], a[2]), (b[0], b[2]), VISION_C)

    # --- graph branch (no whitening) ---
    g1 = _box(ax, 1, y_g, 18, h, text["graph_in"], GRAPH_C)
    g2 = _box(ax, 21.5, y_g, 18, h, text["graph_enc"], GRAPH_C)
    g4 = _box(ax, 42, y_g, 15.5, h, text["graph_vec"].format(dg=facts["dg"]),
              GRAPH_C, fill=0.22)
    _badge(ax, 30.5, y_g - 1.5, text["trained"], GRAPH_C)
    _arrow(ax, (g1[1], g1[2]), (g2[0], g2[2]), GRAPH_C)
    _arrow(ax, (g2[1], g2[2]), (g4[0], g4[2]), GRAPH_C)

    # --- fusion, between the two rows ---
    f_box = _box(ax, 81, (y_v + y_g) / 2 - 2, 18, h + 8,
                 text["fusion"].format(alpha=facts["alpha"], df=facts["df"]),
                 FUSION_C, fill=0.16, fontsize=6.2)
    _arrow(ax, (v4[1], v4[2]), (f_box[0], f_box[2] + 5), FUSION_C, elbow=True)
    _arrow(ax, (g4[1], g4[2]), (f_box[0], f_box[2] - 5), FUSION_C, elbow=True)

    # --- shared gallery and measure ---
    ax.add_patch(FancyBboxPatch((1, 4), 98, 20, boxstyle="round,pad=0.6,rounding_size=2.0",
                                linewidth=0.9, edgecolor=SHARED_C, facecolor="0.96",
                                linestyle=(0, (4, 3))))
    # Thousands separator per language: 67.405 (it) vs 67,405 (en).
    n = facts["n"] if lang == "it" else f"{facts['n_raw']:,}"
    ax.text(50, 14, text["shared"].format(n=n), ha="center", va="center",
            fontsize=6.4, color="0.20", linespacing=1.5)
    # one arrow per vector; staggered columns keep them clear
    for x, y_from, color in ((69.8, y_v, VISION_C), (49.8, y_g, GRAPH_C),
                             (90, f_box[2] - (h + 8) / 2, FUSION_C)):
        _arrow(ax, (x, y_from), (x, 24.5), color)
    return fig


README_LABELS = {
    "vision_in": "plan image\n(rendered PNG)",
    "vision_enc": "frozen encoder\nPE-Spatial · GeM",
    "vision_post": "PCA whitening\n(train statistics)",
    "vision_vec": "vision vector\n{dv}-d, unit norm",
    "graph_in": "room graph\n(.mat record)",
    "graph_enc": "GATv2 encoder\n(InfoNCE)",
    "graph_vec": "graph vector\n{dg}-d, unit norm",
    "fusion": "late fusion\n\n[√α·v ; √(1−α)·g]\n\nα = {alpha:g}\n(chosen on valid)\n→ {df}-d",
    "shared": ("FAISS inner-product search over the shared gallery of {n} RPLAN plans → ranked list.\n"
               "Each branch alone is searched the same way: same queries, same removed rooms, same exclusions."),
    "frozen": "frozen", "trained": "trained on damaged ↔ complete plans",
}


def draw_readme(facts: dict, width_in: float = 11.0, height_in: float = 3.6):
    """Wider, larger-type variant for the repository README (same facts, cleaner arrows)."""
    t = README_LABELS
    apply_style()
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 92)
    ax.axis("off")
    fs = 10.5
    y_v, y_g, h = 66.0, 34.0, 22.0
    cols = ((1, 15), (21, 16), (42, 16), (63, 16))           # (x, width) per column
    v = [_box(ax, x, y_v, w, h, s, VISION_C, fontsize=fs, fill=0.22 if i == 3 else 0.10)
         for i, ((x, w), s) in enumerate(zip(cols, (t["vision_in"], t["vision_enc"], t["vision_post"],
                                                  t["vision_vec"].format(dv=facts["dv"]))))]
    g = [_box(ax, x, y_g, w, h, s, GRAPH_C, fontsize=fs, fill=0.22 if i == 2 else 0.10)
         for i, ((x, w), s) in enumerate(zip((cols[0], cols[1], cols[3]),
                                             (t["graph_in"], t["graph_enc"], t["graph_vec"].format(dg=facts["dg"]))))]
    gx, gy = 0.6 + 0.7, 0.6 + 1.6          # box pad + visible gap (x and y units differ in size)
    for a, b in zip(v, v[1:]):
        _readme_arrow(ax, (a[1] + gx, a[2]), (b[0] - gx, b[2]), VISION_C)
    for a, b in zip(g, g[1:]):
        _readme_arrow(ax, (a[1] + gx, a[2]), (b[0] - gx, b[2]), GRAPH_C)
    ax.text(29, y_v - 2.5, t["frozen"], ha="center", va="top", fontsize=9, color=VISION_C, style="italic")
    ax.text(29, y_g - 2.5, t["trained"], ha="center", va="top", fontsize=9, color=GRAPH_C, style="italic")
    # fusion box spans both rows
    f_x, f_w, f_y = 84, 15, y_g - 2
    f_h = y_v + h + 2 - f_y
    _box(ax, f_x, f_y, f_w, f_h, t["fusion"].format(alpha=facts["alpha"], df=facts["df"]), FUSION_C,
         fontsize=fs, fill=0.16)
    _readme_arrow(ax, (v[3][1] + gx, v[3][2]), (f_x - gx, v[3][2]), FUSION_C)
    _readme_arrow(ax, (g[2][1] + gx, g[2][2]), (f_x - gx, g[2][2]), FUSION_C)
    ax.add_patch(FancyBboxPatch((1, 3), 98, 18, boxstyle="round,pad=0.6,rounding_size=2.0", linewidth=1.0,
                                edgecolor=SHARED_C, facecolor="0.96", linestyle=(0, (4, 3))))
    ax.text(50, 12, t["shared"].format(n=f"{facts['n_raw']:,}"), ha="center", va="center", fontsize=9.5,
            color="0.20", linespacing=1.6)
    _readme_arrow(ax, (f_x + f_w / 2, f_y - gy), (f_x + f_w / 2, 21 + gy), FUSION_C)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--lang", choices=("it", "en"), default="it")
    p.add_argument("--height", type=float, default=2.5, help="altezza in pollici")
    p.add_argument("--width", type=float, default=TEXT_WIDTH_IN,
                   help="larghezza in pollici (default: larghezza del testo del paper)")
    p.add_argument("--svg", action="store_true", help="scrive anche <name>.svg (vettoriale, per il README)")
    p.add_argument("--layout", choices=("paper", "readme"), default="paper",
                   help="paper = figura del paper · readme = versione larga, caratteri grandi, solo inglese")
    p.add_argument("--name", default="f_pipeline")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--setup", choices=tuple(SETUPS), default="storico",
                   help="storico = gat/asymrob (default) · reset = W gat/comb, fusione principale")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global SELECT, GRAPH_QVEC
    SELECT, GRAPH_QVEC = SETUPS[args.setup]
    facts = read_facts()
    fig = (draw_readme(facts) if args.layout == "readme"
           else draw(facts, args.lang, args.height, args.width))
    notes = [f"vision {facts['dv']}-d + graph {facts['dg']}-d → fusione {facts['df']}-d",
             f"α = {facts['alpha']:g} (da {SELECT})",
             f"gallery condivisa: {facts['n_raw']} piante"]
    for line in notes:
        print(line)
    save_figure(fig, args.name, [VISION_QVEC, GRAPH_QVEC, GALLERY, SELECT],
                notes=notes, out_dir=Path(args.out_dir), svg=args.svg)


if __name__ == "__main__":
    main()
