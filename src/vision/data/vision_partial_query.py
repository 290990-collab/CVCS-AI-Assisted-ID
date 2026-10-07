"""Partial queries for vision retrieval: rooms removed from a rendered RPLAN plan, incompleteness left visible.

Pipeline: choose rooms to remove (3 strategies), map .mat bboxes (256 grid) to snapshot pixels,
flood-fill the room region to white, (open_boundary) erase the walls that bordered it.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.data.rplan_metadata import RoomMeta

STRATEGIES = ("random", "semantic", "topology")

# colour thresholds on the RPLAN snapshot (white background, dark grey walls, pastel rooms, yellow door)
_BG_MIN = 248          # min channel >= : background
_WALL_MAX = 120        # max channel <= : wall
_FLOOD_THRESH = 40     # flood-fill tolerance (absorbs anti-aliasing)
_WALL_ERASE = 9        # dilation to erase boundary walls (~6px thick)


# --- room selection (indices into meta.room_types) ---

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
    """Room indices to remove: random (`fraction` of rooms), semantic (type not in `keep_types`), topology (degree <= `max_degree`, default 1)."""
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


# --- 256 grid -> snapshot pixels ---

def _grid_extent(meta: RoomMeta) -> tuple[int, int, int, int]:
    """Extent (x0,y0,x1,y1) of the union of room bboxes on the 256 grid; gtBoxNew is [x0,y0,x1,y1] = (col,row)."""
    xs0, ys0, xs1, ys1 = zip(*meta.boxes)
    return min(xs0), min(ys0), max(xs1), max(ys1)


def _nonwhite_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box (x0,y0,x1,y1) of non-background pixels."""
    nonwhite = np.any(arr < _BG_MIN, axis=2)
    ys, xs = np.where(nonwhite)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _affine(meta: RoomMeta, arr: np.ndarray):
    """Function (x,y) grid -> (px,py) snapshot; fixed per image (full extent <-> non-white bbox)."""
    gx0, gy0, gx1, gy1 = _grid_extent(meta)
    px0, py0, px1, py1 = _nonwhite_bbox(arr)
    sx = (px1 - px0) / max(gx1 - gx0, 1)
    sy = (py1 - py0) / max(gy1 - gy0, 1)
    return lambda x, y: (px0 + (x - gx0) * sx, py0 + (y - gy0) * sy)


# --- colour masks ---

def _wall_mask(arr: np.ndarray) -> np.ndarray:
    """Wall pixels: dark on all channels."""
    return np.all(arr <= _WALL_MAX, axis=2)


def _is_room(arr: np.ndarray) -> np.ndarray:
    """Room pixels: neither white background nor dark wall."""
    return (arr.min(axis=2) < _BG_MIN) & (arr.max(axis=2) > _WALL_MAX)


def _room_seed(arr: np.ndarray, cx: float, cy: float, box_px) -> tuple[int, int] | None:
    """Flood-fill seed: box centre if it is a room pixel, else the room pixel nearest to it inside the box (robust to L-shaped rooms and edge anti-aliasing)."""
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


# --- partial query rendering ---

def render_partial_image(
    png_path: str | Path,
    meta: RoomMeta,
    removed_idx: list[int],
    open_boundary: bool = True,
) -> Image.Image:
    """Snapshot with `removed_idx` rooms flood-filled white; `open_boundary` also erases their bordering walls (no new walls)."""
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
    """Partial query and removed indices."""
    removed = select_rooms_to_remove(meta, strategy, params, rng)
    return render_partial_image(png_path, meta, removed, open_boundary), removed
