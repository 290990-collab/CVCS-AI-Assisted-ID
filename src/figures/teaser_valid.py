"""
Teaser figure: one damaged query and the top results of vision, graph and late fusion (VALID).
Top strip: the plan and the same plan with half its rooms removed, walls included (`nowalls`); one row per system with the original framed where it comes back.

Nothing is recomputed. The ranking of each system is `ret_rows` of the fused per-query files (alpha=1 vision, 0 graph, 0.6 fusion);
`ret_rows` excludes the query and `self_rr` gives its position (rank = 1 / self_rr), so the full ranking is rebuilt exactly.
Removed rooms come from the `qvec/1` file and the damaged image is redrawn with `vision_damage.render_wiped_image`.

Query rule (printed in `*.sources.txt`): the first in alphabetical order among queries where the fusion ranks the original first
and neither branch does. `--query <nome>` forces another one.

Usage (CPU, seconds):

    python -m src.figures.teaser_valid
    python -m src.figures.teaser_valid --query 12345 --lang en
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from src.data.rplan_metadata import load_metadata
from src.evaluation.perquery import load_perquery
from src.evaluation.query_vectors import load_qvec
from src.evaluation.robustness_auc import fraction_path
from src.figures.style import OKABE_ITO, apply_style, TEXT_WIDTH_IN, save_figure
from src.vision.data.vision_damage import render_wiped_image

SPLIT = "valid"
FRACTION = 0.5                    # fraction of rooms removed
STRATEGY = "nowalls-random"
CONFIG = "configs/vision_retrieval.yaml"
GALLERY = "results/shared_gallery.json"
QVEC = ("results/queryvec/valid/"
        "vision_pespatial_gem_whiten-train_partial-nowalls-random-f0.5_valid.npz")

# row order: branches first, fusion last
SYSTEMS = (
    ("vision", "results/perquery/fusion_valid/fusion_a1", "vision", "vision"),
    ("graph", "results/perquery/fusion_valid/fusion_a0", "graph", "graph"),
    ("fusion", "results/perquery/fusion_valid/fusion_a0.6", "fusione", "late fusion"),
)

# `--setup reset`: fusion with the damage-trained vision (head + whitening, alpha 0.3, seed 42); same queries and removed rooms, QVEC unchanged
_RH = "results/final_pipeline/fusion_head/s42/fusion_valid/fusion_a"
SETUPS = {
    "storico": SYSTEMS,
    "reset": (("vision", _RH + "1", "vision con head", "vision with head"),
              ("graph", _RH + "0", "graph W", "graph W"),
              ("fusion", _RH + "0.3", "fusione", "late fusion")),
}

LABELS = {
    "it": {
        "query": "pianta intera",
        "damaged": "query danneggiata",
        "rank": "originale al {rank}°",
        "lost": "originale oltre il 100°",
        "original": "originale",
        "results": "metà delle stanze tolte, coi loro muri.\nOgni riga: i primi {k} risultati\nper questa stessa query.",
    },
    "en": {
        "query": "whole plan",
        "damaged": "damaged query",
        "rank": "original at rank {rank}",
        "lost": "original beyond rank 100",
        "original": "original",
        "results": "half the rooms removed, walls included.\nEach row: the first {k} results\nfor this same damaged query.",
    },
}


def _rank_of_self(self_rr: float) -> int | None:
    """Position of the query in its own ranking (None if it never came back)."""
    return int(round(1.0 / self_rr)) if self_rr > 0 else None


def load_rankings(split: str = SPLIT):
    """Three systems at f=0.5: (systems, names, sources); `systems[key]` has `rows` [Q, 100] and `rank` (None if never retrieved)."""
    systems, sources, names = {}, [], None
    for key, prefix, _, _ in SYSTEMS:
        path = fraction_path(prefix, FRACTION, split, STRATEGY)
        data = load_perquery(path)
        if data.ret_rows is None or data.self_rr is None:
            raise ValueError(f"{path}: servono `ret_rows` e `self_rr` (run partial)")
        if names is None:
            names = [str(n) for n in data.names]
        elif [str(n) for n in data.names] != names:
            raise ValueError(f"{path}: query diverse dagli altri sistemi, non è appaiato")
        systems[key] = {"rows": data.ret_rows,
                        "rank": [_rank_of_self(float(v)) for v in data.self_rr]}
        sources.append(str(path))
    return systems, names, sources


def choose_query(systems: dict, names: list[str], forced: str | None) -> tuple[int, str]:
    """Apply the query rule (or the forced name); returns (index, rule description). Raises ValueError if none applies."""
    if forced is not None:
        if forced not in names:
            raise ValueError(f"query {forced!r} non fra le {len(names)} query dello split")
        return names.index(forced), f"query imposta da riga di comando: {forced}"

    candidates = [i for i in range(len(names))
                  if systems["fusion"]["rank"][i] == 1
                  and systems["vision"]["rank"][i] != 1
                  and systems["graph"]["rank"][i] != 1]
    if not candidates:
        raise ValueError("nessuna query soddisfa la regola (fusione 1ª, nessun ramo 1º)")
    chosen = min(candidates, key=lambda i: names[i])
    return chosen, (f"prima in ordine alfabetico fra le {len(candidates)} query in cui la "
                    f"fusione mette l'originale al 1° posto e nessuno dei due rami lo fa")


def full_ranking(rows: np.ndarray, rank: int | None, self_row: int, k: int) -> list[int]:
    """First k entries of the ranking with the query reinserted at its own position (`ret_rows` omits it)."""
    ranking = [int(r) for r in rows if r >= 0]
    if rank is not None:
        ranking.insert(rank - 1, self_row)
    return ranking[:k]


def _trim_box(img: Image.Image, pad: int = 12) -> tuple[int, int, int, int]:
    """Bounding box of the drawing, so the thumbnails are plan and not margin."""
    arr = np.asarray(img.convert("RGB"))
    ink = np.any(arr < 250, axis=2)
    if not ink.any():
        return (0, 0, img.width, img.height)
    ys, xs = np.where(ink)
    return (max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0),
            min(int(xs.max()) + pad, img.width), min(int(ys.max()) + pad, img.height))


def _cell(ax, img: Image.Image, box=None, frame: str | None = None, width: float = 1.6):
    """Draw one thumbnail, optionally cropped to `box` and framed."""
    ax.imshow(img.crop(box) if box else img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(frame or "0.8")
        spine.set_linewidth(width if frame else 0.5)


def draw(images: dict, systems: dict, index: int, names: list[str],
         lang: str, top_k: int):
    """Build the teaser: query strip on top, one row of results per system."""
    text = LABELS[lang]
    apply_style()
    import matplotlib.pyplot as plt

    # portrait plans: cells taller than wide
    row_h = TEXT_WIDTH_IN / top_k * 0.95
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, row_h * 4 + 0.55))
    grid = fig.add_gridspec(4, top_k, height_ratios=[1.05] + [1.0] * 3,
                            left=0.135, right=0.998, top=0.94, bottom=0.035,
                            hspace=0.24, wspace=0.03)

    # --- strip: whole plan and damaged query ---
    box = _trim_box(images["whole"])
    for col, (key, caption) in enumerate((("whole", text["query"]),
                                          ("damaged", text["damaged"]))):
        ax = fig.add_subplot(grid[0, col])
        _cell(ax, images[key], box, frame=OKABE_ITO["black"] if key == "damaged" else None,
              width=1.2)
        ax.set_title(caption, fontsize=7, pad=3)
    note = fig.add_subplot(grid[0, 2:])
    note.axis("off")
    note.text(0.0, 0.5, text["results"].format(k=top_k), fontsize=7, color="0.35",
              ha="left", va="center", linespacing=1.5)

    # --- one row per system ---
    for row, (key, _, label_it, label_en) in enumerate(SYSTEMS, start=1):
        rank = systems[key]["rank"][index]
        label = label_it if lang == "it" else label_en
        state = text["rank"].format(rank=rank) if rank else text["lost"]
        for col in range(top_k):
            ax = fig.add_subplot(grid[row, col])
            name, img = images["results"][key][col]
            is_self = name == names[index]
            _cell(ax, img, _trim_box(img),
                  frame=OKABE_ITO["green"] if is_self else None)
            ax.set_xlabel(f"{col + 1}", fontsize=6, color="0.45", labelpad=1.5)
            if is_self:
                ax.set_xlabel(f"{col + 1} · {text['original']}", fontsize=6,
                              color=OKABE_ITO["green"], labelpad=1.5)
            if col == 0:
                ax.set_ylabel(f"{label}\n{state}", fontsize=7.5, labelpad=6,
                              rotation=0, ha="right", va="center", linespacing=1.6)
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--lang", choices=("it", "en"), default="it")
    p.add_argument("--query", default=None, help="nome della query da usare (default: la regola)")
    p.add_argument("--top-k", type=int, default=5, help="risultati per riga (default: 5)")
    p.add_argument("--split", default=SPLIT)
    p.add_argument("--name", default="f_teaser_valid", help="nome base dei file in figures/")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--setup", choices=tuple(SETUPS), default="storico",
                   help="storico = gat/asymrob (default) · reset = W + vision con head")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global SYSTEMS
    SYSTEMS = SETUPS[args.setup]
    systems, names, sources = load_rankings(args.split)
    index, rule = choose_query(systems, names, args.query)
    query_name = names[index]

    gallery = json.loads(Path(GALLERY).read_text(encoding="utf-8"))["names"]
    data_dir = Path(OmegaConf.load(CONFIG).retrieval.data_dir)
    png = lambda name: data_dir / f"{name}.png"                       # noqa: E731
    self_row = gallery.index(query_name)

    # rooms removed by the evaluation, same renderer
    qvec = load_qvec(QVEC)
    qnames = [str(n) for n in qvec.names]
    if qnames[index] != query_name:
        index_q = qnames.index(query_name)
    else:
        index_q = index
    removed = qvec.removed(index_q)
    meta = load_metadata(png(query_name))
    if meta is None:
        raise ValueError(f"{query_name}: nessun record .mat, non si può ridisegnare il danno")

    images = {"whole": Image.open(png(query_name)).convert("RGB"),
              "damaged": render_wiped_image(png(query_name), meta, removed),
              "results": {}}
    ranked_names = {}
    for key, _, _, _ in SYSTEMS:
        rows = full_ranking(systems[key]["rows"][index], systems[key]["rank"][index],
                            self_row, args.top_k)
        ranked_names[key] = [gallery[r] for r in rows]
        images["results"][key] = [(gallery[r], Image.open(png(gallery[r])).convert("RGB"))
                                  for r in rows]

    fig = draw(images, systems, index, names, args.lang, args.top_k)
    notes = [f"split: {args.split} · danno: {STRATEGY} f={FRACTION}",
             f"query: {query_name} ({len(removed)} stanze tolte: {removed})",
             f"regola di scelta: {rule}"]
    for key, _, label_it, _ in SYSTEMS:
        rank = systems[key]["rank"][index]
        notes.append(f"{label_it:8s} originale al posto {rank if rank else '>100'} | "
                     f"primi {args.top_k}: {', '.join(ranked_names[key])}")
    for line in notes:
        print(line)
    save_figure(fig, args.name, sources + [QVEC, GALLERY, CONFIG], notes=notes,
                out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
