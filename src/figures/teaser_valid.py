# src/figures/teaser_valid.py

"""
Teaser figure (F1): one damaged query and the first results of the three
systems, VALID split (18 Sep 2026).

What it shows
-------------
Top strip: a plan of the gallery and the same plan after half of its rooms have
been removed, walls included (the `nowalls` damage, the one the numbers are
measured on). Then one row per system — vision, graph, late fusion — with their
first results for that damaged query, and the original plan framed where it
comes back. The row label says at which position each system put it.

It is the abstract of the report in one picture: the same damaged query, three
rankings, and only the fused one brings the original back to the top.

Where it comes from — no job, no GPU
------------------------------------
Everything is already on disk and nothing is recomputed:

- the ranking of each system is `ret_rows` of the fused per-query files
  (alpha=1 = vision, alpha=0 = graph, alpha=0.6 = fusion). `ret_rows` stores the
  ranking WITHOUT the query itself, and `self_rr` stores where the query was
  (rank = 1 / self_rr), so the original ranking is reconstructed exactly by
  putting the plan back at its own position;
- the removed rooms are the ones the evaluation actually removed, read from the
  `qvec/1` file, and the damaged image is redrawn from them with the same
  renderer as the evaluation (`vision_damage.render_wiped_image`).

The split is the VALID on purpose: a qualitative figure does not need the test,
and the test stays what it is, a number read once.

Which query
-----------
Rule declared before looking (and printed in `*.sources.txt`): among the queries
where the FUSION puts the original first and NEITHER branch does, the first one
in alphabetical order. `--query <nome>` forces another one, and the file it
writes says which rule produced the picture.

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
FRACTION = 0.5                    # metà stanze tolte: è il livello che separa di più i sistemi
STRATEGY = "nowalls-random"
CONFIG = "configs/vision_retrieval.yaml"
GALLERY = "results/shared_gallery.json"
QVEC = ("results/queryvec/valid/"
        "vision_pespatial_gem_whiten-train_partial-nowalls-random-f0.5_valid.npz")

# Row order = story order: the two branches first, the fusion last.
SYSTEMS = (
    ("vision", "results/perquery/fusion_valid/fusion_a1", "vision", "vision"),
    ("graph", "results/perquery/fusion_valid/fusion_a0", "graph", "graph"),
    ("fusion", "results/perquery/fusion_valid/fusion_a0.6", "fusione", "late fusion"),
)

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
    """Read the three systems at f=0.5: names, ranking without self, rank of self.

    Returns:
        (systems, names, sources) where `systems[key]` has `rows` [Q, 100] and
        `rank` (list with None where the query never came back).
    """
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
    """Apply the declared rule (or the forced name) and return (index, rule used).

    Raises:
        ValueError: if the forced name is not among the queries, or if no query
            satisfies the rule.
    """
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
    """Ranking WITH the query put back at its own position, first k entries.

    `ret_rows` is saved without the query itself, so re-inserting it at `rank`
    rebuilds exactly the ranking the system produced.
    """
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

    # Le piante sono ritratte: celle piu' alte che larghe, altrimenti l'immagine
    # si adatta all'altezza e lascia meta' cella bianca.
    row_h = TEXT_WIDTH_IN / top_k * 0.95
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, row_h * 4 + 0.55))
    grid = fig.add_gridspec(4, top_k, height_ratios=[1.05] + [1.0] * 3,
                            left=0.135, right=0.998, top=0.94, bottom=0.035,
                            hspace=0.24, wspace=0.03)

    # --- strip: the plan as it is, and the query the systems actually saw ---
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

    # --- one row per system, the original framed where it comes back ---
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    systems, names, sources = load_rankings(args.split)
    index, rule = choose_query(systems, names, args.query)
    query_name = names[index]

    gallery = json.loads(Path(GALLERY).read_text(encoding="utf-8"))["names"]
    data_dir = Path(OmegaConf.load(CONFIG).retrieval.data_dir)
    png = lambda name: data_dir / f"{name}.png"                       # noqa: E731
    self_row = gallery.index(query_name)

    # The rooms the evaluation removed from THIS query, and the same rendering.
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
