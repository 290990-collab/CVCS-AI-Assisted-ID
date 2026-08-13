# src/vision/data/vision_partial_query.py

"""
Query parziali per il retrieval VISION: rimuove stanze da una pianta RPLAN
renderizzata e lascia l'incompletezza VISIBILE (bordo aperto, nessun muro nuovo).

Pipeline: scegli le stanze da togliere (3 strategie) -> mappa le bbox .mat
(griglia 256) sui pixel dello snapshot -> flood-fill della regione-stanza a
bianco -> (open_boundary) cancella i muri che la bordavano.

Solo ramo vision; il ramo graph avrà un proprio masking sui grafi.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.data.rplan_metadata import RoomMeta

STRATEGIES = ("random", "semantic", "topology")

# Soglie colore sullo snapshot RPLAN (sfondo bianco, muri grigio scuro,
# stanze a tinte pastello, porta gialla).
_BG_MIN = 248          # pixel con min-canale >= = sfondo bianco
_WALL_MAX = 120        # pixel con max-canale <= = muro
_FLOOD_THRESH = 40     # tolleranza flood-fill (assorbe l'anti-aliasing)
_WALL_ERASE = 9        # dilatazione per cancellare i muri (spessi ~6px) del bordo


# ----------------------------------------------------------------------
# Selezione delle stanze da rimuovere (indici in meta.room_types).
# ----------------------------------------------------------------------

def _room_degrees(meta: RoomMeta) -> list[int]:
    deg = [0] * meta.num_rooms
    for i, j, _ in meta.edges:
        if 0 <= i < meta.num_rooms:
            deg[i] += 1
        if 0 <= j < meta.num_rooms:
            deg[j] += 1
    return deg


def select_rooms_to_remove(
    meta: RoomMeta,
    strategy: str,
    params: dict,
    rng: random.Random,
) -> list[int]:
    """Indici delle stanze da togliere secondo la strategia.

    - random:   togli una `fraction` di stanze a caso.
    - semantic: togli le stanze il cui tipo NON è in `keep_types`.
    - topology: togli le stanze foglia (grado <= `max_degree`, default 1).
    """
    if strategy == "random":
        frac = float(params.get("fraction", 0.0))
        n_remove = int(round(frac * meta.num_rooms))
        return sorted(rng.sample(range(meta.num_rooms), n_remove)) if n_remove else []

    if strategy == "semantic":
        keep = set(params.get("keep_types", []))
        return [i for i, t in enumerate(meta.room_types) if t not in keep]

    if strategy == "topology":
        max_deg = int(params.get("max_degree", 1))
        deg = _room_degrees(meta)
        return [i for i, d in enumerate(deg) if d <= max_deg]

    raise ValueError(f"strategia sconosciuta: {strategy!r}")


# ----------------------------------------------------------------------
# Mappatura griglia-256 -> pixel dello snapshot.
# ----------------------------------------------------------------------

def _grid_extent(meta: RoomMeta) -> tuple[int, int, int, int]:
    """Estensione (x0,y0,x1,y1) dell'unione delle bbox-stanza in griglia 256.
    gtBoxNew è [x0,y0,x1,y1] = (col,row)."""
    xs0, ys0, xs1, ys1 = zip(*meta.boxes)
    return min(xs0), min(ys0), max(xs1), max(ys1)


def _nonwhite_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box (x0,y0,x1,y1) dei pixel non-sfondo dello snapshot."""
    nonwhite = np.any(arr < _BG_MIN, axis=2)
    ys, xs = np.where(nonwhite)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _affine(meta: RoomMeta, arr: np.ndarray):
    """Ritorna una funzione (x,y)_griglia -> (px,py)_snapshot.
    L'affine è fissa per immagine (estensione completa <-> non-bianco)."""
    gx0, gy0, gx1, gy1 = _grid_extent(meta)
    px0, py0, px1, py1 = _nonwhite_bbox(arr)
    sx = (px1 - px0) / max(gx1 - gx0, 1)
    sy = (py1 - py0) / max(gy1 - gy0, 1)
    return lambda x, y: (px0 + (x - gx0) * sx, py0 + (y - gy0) * sy)


# ----------------------------------------------------------------------
# Maschere colore.
# ----------------------------------------------------------------------

def _wall_mask(arr: np.ndarray) -> np.ndarray:
    """Pixel-muro: scuri su tutti i canali."""
    return np.all(arr <= _WALL_MAX, axis=2)


def _is_room(arr: np.ndarray) -> np.ndarray:
    """Pixel-stanza: né sfondo bianco né muro scuro."""
    return (arr.min(axis=2) < _BG_MIN) & (arr.max(axis=2) > _WALL_MAX)


def _room_seed(arr: np.ndarray, cx: float, cy: float, box_px) -> tuple[int, int] | None:
    """Seed per il flood-fill: il centro-box se è un pixel-stanza, altrimenti il
    pixel-stanza più vicino al centro dentro la box (robusto su stanze a L e
    sull'anti-aliasing dei bordi)."""
    h, w = arr.shape[:2]
    cx, cy = int(round(cx)), int(round(cy))
    if 0 <= cx < w and 0 <= cy < h:
        p = arr[cy, cx]
        if p.min() < _BG_MIN and p.max() > _WALL_MAX:
            return cx, cy

    x0, y0, x1, y1 = (int(round(v)) for v in box_px)
    x0, x1 = max(min(x0, x1), 0), min(max(x0, x1), w)
    y0, y1 = max(min(y0, y1), 0), min(max(y0, y1), h)
    sub = _is_room(arr[y0:y1, x0:x1])
    ys, xs = np.where(sub)
    if len(xs) == 0:
        return None
    d = (xs + x0 - cx) ** 2 + (ys + y0 - cy) ** 2
    k = int(np.argmin(d))
    return int(xs[k] + x0), int(ys[k] + y0)


# ----------------------------------------------------------------------
# Rendering della query parziale.
# ----------------------------------------------------------------------

def render_partial_image(
    png_path: str | Path,
    meta: RoomMeta,
    removed_idx: list[int],
    open_boundary: bool = True,
) -> Image.Image:
    """Snapshot con le stanze in `removed_idx` rimosse.

    Riempie ogni stanza di bianco (flood-fill); se open_boundary, cancella anche
    i muri che la bordavano così l'incompletezza resta visibile (mai muri nuovi).
    """
    img = Image.open(png_path).convert("RGB")
    if not removed_idx:
        return img

    arr = np.array(img)
    to_px = _affine(meta, arr)
    before_bg = np.all(arr >= _BG_MIN, axis=2)

    for idx in removed_idx:
        x0, y0, x1, y1 = meta.boxes[idx]
        cx, cy = to_px((x0 + x1) / 2, (y0 + y1) / 2)
        box_px = (*to_px(x0, y0), *to_px(x1, y1))
        seed = _room_seed(np.array(img), cx, cy, box_px)
        if seed is not None:
            ImageDraw.floodfill(img, seed, (255, 255, 255), thresh=_FLOOD_THRESH)

    if not open_boundary:
        return img

    # Cancella i muri adiacenti alle regioni rimosse (bordo aperto).
    arr = np.array(img)
    removed = np.all(arr >= _BG_MIN, axis=2) & ~before_bg
    if removed.any():
        from scipy.ndimage import binary_dilation
        near_removed = binary_dilation(removed, iterations=_WALL_ERASE)
        erase = near_removed & _wall_mask(arr)
        arr[erase] = (255, 255, 255)
        img = Image.fromarray(arr)
    return img


def make_partial_query(
    png_path: str | Path,
    meta: RoomMeta,
    strategy: str,
    params: dict,
    rng: random.Random,
    open_boundary: bool = True,
) -> tuple[Image.Image, list[int]]:
    """Query parziale + indici rimossi (per logging/analisi)."""
    removed = select_rooms_to_remove(meta, strategy, params, rng)
    return render_partial_image(png_path, meta, removed, open_boundary), removed
