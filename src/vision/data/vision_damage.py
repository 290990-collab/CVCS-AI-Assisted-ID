# src/vision/data/vision_damage.py

"""
Damage modes for the vision partial queries, beside the historical room removal.

Three families, one entry point (`damaged_query`):

- rooms (random | semantic | topology): delegated LITERALLY to
  `vision_partial_query.make_partial_query` + the encoder transform, i.e. the
  same two calls `evaluate_partial` made before this module existed. The
  queries are bit-identical to the historical ones (golden hashes in
  `tests/test_vision_damage.py`); only the removed plan area is measured on
  top, without consuming the query rng.
- crop: one axis-aligned white rectangle on the NATIVE snapshot, sized so that
  it covers a target fraction of the plan pixels. No dilation, no wall erased:
  the plan loses a contiguous block, rooms and walls alike.
- patch: whole ViT patches (P x P cells on the RESIZED canvas, grid anchored at
  its (0,0)) set to white until a target fraction of the plan is covered. The
  damage is aligned with the encoder tokenisation, so it removes whole tokens.
- nowalls (nowalls_random | nowalls_semantic | nowalls_topology): the same room
  selection as the historical strategies and the same flood fill, but the walls
  bordering the removed rooms are erased WHATEVER their shade of grey. The
  historical mode only erases walls darker than `_WALL_MAX = 120`, which in
  RPLAN is the perimeter (79) and not the internal partitions (128): the removed
  room keeps its own walls and stays a readable white cell. Here the query loses
  the room type AND its walls, keeping only the shape of the hole - the same
  information crop and patch leave behind. Paired with the historical mode by
  construction: same `rng`, same draws, so the same query removes the same rooms.

Area convention (`area_removed`): plan pixels (any channel < `_BG_MIN`, the
same threshold as `_nonwhite_bbox`) that are background-white after the damage,
over the plan pixels before. Measured on the native snapshot for rooms/crop and
on the resized canvas for patch (`area_space` in the per-query meta): the two
are close but not the same measure.

Only the vision branch; the graph branch has its own masking on graphs.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms

from src.data.rplan_metadata import RoomMeta
from src.vision.data.preprocess import ResizeWithPad
from src.vision.data.vision_partial_query import (
    STRATEGIES as ROOM_STRATEGIES,
    _BG_MIN,
    _FLOOD_THRESH,
    _WALL_ERASE,
    _affine,
    _room_seed,
    make_partial_query,
    select_rooms_to_remove,
)

DAMAGE_STRATEGIES = ("crop", "patch")
NOWALLS_PREFIX = "nowalls_"
NOWALLS_STRATEGIES = tuple(NOWALLS_PREFIX + s for s in ROOM_STRATEGIES)

DEFAULT_TOLERANCE = 0.05
DEFAULT_MAX_TRIES = 10000


# ----------------------------------------------------------------------
# Plan mask and removed area.
# ----------------------------------------------------------------------

def plan_mask(arr: np.ndarray) -> np.ndarray:
    """Plan pixels: any channel below `_BG_MIN` (same rule as `_nonwhite_bbox`)."""
    return np.any(np.asarray(arr) < _BG_MIN, axis=2)


def plan_area_removed(before: np.ndarray, after: np.ndarray) -> float:
    """Fraction of the plan pixels of `before` that are background in `after`.

    0.0 if `before` has no plan pixel.
    """
    plan = plan_mask(before)
    n_plan = int(plan.sum())
    if n_plan == 0:
        return 0.0
    gone = plan & ~plan_mask(after)
    return float(gone.sum() / n_plan)


# ----------------------------------------------------------------------
# Nowalls: room removal that also erases the walls of the removed rooms.
# ----------------------------------------------------------------------

# A wall pixel is a NEUTRAL grey that is not background: the three channels are
# within `_WALL_SPREAD` of each other and the brightest one is at most
# `_WALL_VMAX`. On RPLAN snapshots this catches the perimeter (79), the internal
# partitions (128) and the anti-aliased greys in between, while every room
# colour stays out (the palest, (244,242,229), is above `_WALL_VMAX`).
_WALL_SPREAD = 25
_WALL_VMAX = 220


def neutral_wall_mask(arr: np.ndarray) -> np.ndarray:
    """Wall pixels of any shade: near-grey and not background."""
    a = np.asarray(arr).astype(np.int16)
    return (a.max(axis=2) - a.min(axis=2) <= _WALL_SPREAD) & (a.max(axis=2) <= _WALL_VMAX)


def render_wiped_image(png_path, meta: RoomMeta, removed_idx: list[int]) -> Image.Image:
    """Snapshot with the rooms in `removed_idx` emptied AND their walls erased.

    The flood fill is literally the historical one (same seeds, same threshold),
    so the emptied region is identical to `render_partial_image`; only the wall
    rule of the second step changes, from "dark" to "any neutral grey". No wall
    is ever drawn, so the plan can only lose information.
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

    arr = np.array(img)
    removed = np.all(arr >= _BG_MIN, axis=2) & ~before_bg
    if removed.any():
        from scipy.ndimage import binary_dilation
        near_removed = binary_dilation(removed, iterations=_WALL_ERASE)
        arr[near_removed & neutral_wall_mask(arr)] = 255
        img = Image.fromarray(arr)
    return img


def make_wiped_query(
    png_path,
    meta: RoomMeta,
    strategy: str,
    params: dict,
    rng: random.Random,
) -> tuple[Image.Image, list[int]]:
    """`make_partial_query` with the nowalls rendering: same selection, same rng.

    `strategy` is a historical room strategy (without the `nowalls_` prefix), so
    for a given (seed, query) the removed rooms are exactly the ones the paired
    historical run removes.
    """
    removed = select_rooms_to_remove(meta, strategy, params, rng)
    return render_wiped_image(png_path, meta, removed), removed


# ----------------------------------------------------------------------
# Room damage for the projection head (training pairs and epoch probe).
# ----------------------------------------------------------------------

HEAD_DAMAGES = ("random", NOWALLS_PREFIX + "random")


def check_head_damage(damage) -> str:
    """Validate `training.damage`, the render of the head's positive views and probe."""
    damage = str(damage)
    if damage not in HEAD_DAMAGES:
        raise ValueError(f"training.damage={damage!r} (expected: {' | '.join(HEAD_DAMAGES)})")
    return damage


def room_damage_image(
    png_path,
    meta: RoomMeta,
    damage: str,
    params: dict,
    rng: random.Random,
    open_boundary: bool,
) -> tuple[Image.Image, list[int]]:
    """Damaged image for the head: historical render (`random`) or `nowalls_random`.

    `random` is literally `make_partial_query`, so historical pairs and probes are
    unchanged. Both branches draw from `rng` only to select the rooms, so for the
    same rng they remove the same rooms and any later draw (flip/rot90) matches.
    """
    damage = check_head_damage(damage)
    if damage.startswith(NOWALLS_PREFIX):
        return make_wiped_query(png_path, meta, damage[len(NOWALLS_PREFIX):], params, rng)
    return make_partial_query(png_path, meta, damage, params, rng, open_boundary)


# ----------------------------------------------------------------------
# Crop: one rectangle on the native snapshot.
# ----------------------------------------------------------------------

def _integral(mask: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero first row/column: rectangle sums in O(1)."""
    ii = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    ii[1:, 1:] = mask.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    return ii


def _rect_sum(ii: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int:
    """Sum of the mask over rows [y0, y1) and columns [x0, x1)."""
    return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def make_crop_query(
    png_path: str | Path,
    fraction: float,
    rng: random.Random,
    tolerance: float = DEFAULT_TOLERANCE,
    max_tries: int = DEFAULT_MAX_TRIES,
) -> tuple[Image.Image, dict]:
    """Snapshot with one white rectangle covering ~`fraction` of the plan.

    Draws, in this order and from `rng` only: width w in U{1..W_bbox}, height
    h in U{1..H_bbox}, then the top-left corner uniformly among the positions
    that keep the rectangle inside the plan bounding box. The first rectangle
    whose plan coverage is within `tolerance` of `fraction` is accepted; after
    `max_tries` the closest one is used and `fallback=True` is reported.

    Returns:
        (image, info) with info = {rect (x0,y0,x1,y1) half-open, coverage,
        area_removed, tries, fallback}.
    """
    img = Image.open(png_path).convert("RGB")
    arr = np.array(img)
    plan = plan_mask(arr)
    n_plan = int(plan.sum())
    if n_plan == 0:
        return img, {"rect": None, "coverage": 0.0, "area_removed": 0.0,
                     "tries": 0, "fallback": abs(float(fraction)) > tolerance}

    ys, xs = np.where(plan)
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bw, bh = bx1 - bx0 + 1, by1 - by0 + 1
    ii = _integral(plan)

    best = None                      # (error, rect, coverage, tries)
    accepted = False
    for t in range(1, int(max_tries) + 1):
        w = rng.randint(1, bw)
        h = rng.randint(1, bh)
        x0 = rng.randint(bx0, bx1 - w + 1)
        y0 = rng.randint(by0, by1 - h + 1)
        rect = (x0, y0, x0 + w, y0 + h)
        coverage = _rect_sum(ii, *rect) / n_plan
        err = abs(coverage - float(fraction))
        if best is None or err < best[0]:
            best = (err, rect, coverage, t)
        if err <= tolerance:
            accepted = True
            break

    _, rect, coverage, tries = best
    x0, y0, x1, y1 = rect
    arr[y0:y1, x0:x1] = 255
    return Image.fromarray(arr), {
        "rect": rect, "coverage": float(coverage), "area_removed": float(coverage),
        "tries": tries if accepted else int(max_tries), "fallback": not accepted,
    }


# ----------------------------------------------------------------------
# Patch: whole ViT patches on the resized canvas.
# ----------------------------------------------------------------------

def split_query_transform(transform) -> tuple[ResizeWithPad, transforms.Compose]:
    """Splits the encoder transform into (resize, finish).

    `finish(resize(img))` is the same computation as `transform(img)`: the
    patch damage is applied in between, on the canvas the ViT actually sees.

    Raises:
        ValueError: if `transform` is not a Compose starting with ResizeWithPad.
    """
    steps = getattr(transform, "transforms", None)
    if not isinstance(transform, transforms.Compose) or not steps \
            or not isinstance(steps[0], ResizeWithPad):
        raise ValueError(
            "patch damage needs a transforms.Compose whose first step is "
            f"ResizeWithPad, got {transform!r}"
        )
    return steps[0], transforms.Compose(list(steps[1:]))


def select_patch_cells(
    canvas,
    patch_size: int,
    fraction: float,
    rng: random.Random,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[list[tuple[int, int]], dict]:
    """Patch cells (row, col) to blank so that ~`fraction` of the plan goes.

    Grid of G = S / P cells anchored at (0, 0) of the S x S canvas. Cells with
    at least one plan pixel are shuffled with `rng` and added until the
    cumulative plan coverage reaches `fraction`; then the closer to `fraction`
    between "with" and "without" the last cell is kept (ties keep it).
    `fallback=True` if the result is farther than `tolerance`.

    Raises:
        ValueError: non-square canvas or S not a multiple of P.
    """
    arr = np.asarray(canvas)
    s, s2 = arr.shape[:2]
    p = int(patch_size)
    if s != s2 or p <= 0 or s % p:
        raise ValueError(f"canvas {s2}x{s} is not a square multiple of patch_size={p}")
    g = s // p
    plan = plan_mask(arr)
    n_plan = int(plan.sum())
    counts = plan.reshape(g, p, g, p).sum(axis=(1, 3))
    cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(counts))]
    n_plan_cells = len(cells)
    rng.shuffle(cells)

    chosen: list[tuple[int, int]] = []
    covered = 0
    target = float(fraction)
    for cell in cells:
        if n_plan and covered / n_plan >= target:
            break
        chosen.append(cell)
        covered += int(counts[cell])
    if chosen and n_plan:
        without = covered - int(counts[chosen[-1]])
        if abs(without / n_plan - target) < abs(covered / n_plan - target):
            chosen.pop()
            covered = without

    coverage = covered / n_plan if n_plan else 0.0
    return chosen, {"coverage": float(coverage), "n_cells": len(chosen),
                    "n_plan_cells": n_plan_cells,
                    "fallback": abs(coverage - target) > tolerance}


def apply_patch_cells(canvas, cells, patch_size: int) -> Image.Image:
    """Copy of `canvas` with every chosen P x P block set to white."""
    arr = np.array(canvas)
    p = int(patch_size)
    for r, c in cells:
        arr[r * p:(r + 1) * p, c * p:(c + 1) * p] = 255
    return Image.fromarray(arr)


def _encoder_patch_size(encoder) -> int | None:
    """Best-effort patch size read from the loaded encoder, None if not found."""
    probes = (
        ("backbone", "config", "patch_size"),
        ("model", "config", "vision_config", "patch_size"),
        ("model", "config", "patch_size"),
        ("model", "patch_embed", "patch_size"),
    )
    for path in probes:
        obj = encoder
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is None:
            continue
        if isinstance(obj, (tuple, list)):
            if len(set(int(v) for v in obj)) != 1:
                raise ValueError(f"non-square patch size {tuple(obj)} at {'.'.join(path)}")
            obj = obj[0]
        try:
            return int(obj)
        except (TypeError, ValueError):
            continue
    return None


def resolve_patch_size(encoder, declared, image_size: int) -> int:
    """Patch size for the patch damage: the preset value, checked on the model.

    The value comes from the encoder preset (`model.kwargs.patch_size`); when
    the loaded encoder exposes its own patch size the two must agree. Called
    once, before the first query, so a wrong preset fails fast.

    Raises:
        ValueError: missing value, mismatch with the model, or image_size not a
            multiple of the patch size.
    """
    found = _encoder_patch_size(encoder)
    if declared is None and found is None:
        raise ValueError("patch damage: `model.kwargs.patch_size` missing and the "
                         "encoder does not expose a patch size")
    if declared is not None and found is not None and int(declared) != found:
        raise ValueError(f"patch damage: preset patch_size={declared} but the "
                         f"encoder uses {found}")
    if found is None:
        warnings.warn(f"patch damage: the encoder does not expose a patch size; "
                      f"using the preset value {declared} unchecked")
    p = int(declared if declared is not None else found)
    if int(image_size) % p:
        raise ValueError(f"patch damage: image_size={image_size} is not a multiple "
                         f"of patch_size={p}")
    return p


def make_patch_context(transform, patch_size: int) -> dict:
    """Everything the patch damage needs, built once per evaluation."""
    resize, finish = split_query_transform(transform)
    s = int(resize.target_size)
    if s % int(patch_size):
        raise ValueError(f"image_size={s} is not a multiple of patch_size={patch_size}")
    return {"resize": resize, "finish": finish,
            "patch_size": int(patch_size), "image_size": s}


# ----------------------------------------------------------------------
# Entry point.
# ----------------------------------------------------------------------

def damaged_query(
    png_path: str | Path,
    meta: RoomMeta,
    strategy: str,
    params: dict,
    rng: random.Random,
    open_boundary: bool,
    transform,
    patch_ctx: dict | None = None,
    return_image: bool = False,
):
    """Damaged query tensor + info {removed_any, area_removed, fallback}.

    Room and nowalls strategies also put the removed room indices in
    `info["removed"]`. With `return_image=True` a third element is returned: the
    damaged image right before the final transform (the native snapshot for
    rooms/nowalls/crop, the resized canvas for patch), so a visualisation shows
    exactly what was encoded. The default keeps the historical two-element return.

    Room strategies are the historical path, untouched: `make_partial_query`
    then `transform`. The area is measured afterwards on a fresh read of the
    snapshot, so `rng` sees exactly the same draws as before. The `nowalls_*`
    strategies take the same path with `make_wiped_query` instead, so they draw
    from `rng` exactly like their historical twin.
    """
    if strategy in ROOM_STRATEGIES or strategy in NOWALLS_STRATEGIES:
        if strategy in ROOM_STRATEGIES:
            img, removed = make_partial_query(png_path, meta, strategy, params, rng,
                                              open_boundary)
        else:
            # `open_boundary` does not apply: erasing the walls IS this mode.
            img, removed = make_wiped_query(png_path, meta,
                                            strategy[len(NOWALLS_PREFIX):], params, rng)
        tensor = transform(img)
        area = 0.0
        if removed:
            before = np.array(Image.open(png_path).convert("RGB"))
            area = plan_area_removed(before, np.asarray(img))
        info = {"removed_any": bool(removed), "area_removed": area,
                "fallback": False, "removed": list(removed)}
        return (tensor, info, img) if return_image else (tensor, info)

    if strategy == "crop":
        img, info = make_crop_query(
            png_path, float(params["fraction"]), rng,
            tolerance=float(params.get("tolerance", DEFAULT_TOLERANCE)),
            max_tries=int(params.get("max_tries", DEFAULT_MAX_TRIES)),
        )
        out = {"removed_any": info["area_removed"] > 0.0,
               "area_removed": info["area_removed"], "fallback": info["fallback"]}
        tensor = transform(img)
        return (tensor, out, img) if return_image else (tensor, out)

    if strategy == "patch":
        if patch_ctx is None:
            raise ValueError("patch damage needs a patch context (make_patch_context)")
        canvas = patch_ctx["resize"](Image.open(png_path).convert("RGB"))
        cells, info = select_patch_cells(
            canvas, patch_ctx["patch_size"], float(params["fraction"]), rng,
            tolerance=float(params.get("tolerance", DEFAULT_TOLERANCE)),
        )
        damaged = apply_patch_cells(canvas, cells, patch_ctx["patch_size"])
        area = plan_area_removed(np.asarray(canvas), np.asarray(damaged))
        out = {"removed_any": bool(cells), "area_removed": area,
               "fallback": info["fallback"]}
        tensor = patch_ctx["finish"](damaged)
        return (tensor, out, damaged) if return_image else (tensor, out)

    raise ValueError(f"unknown damage strategy: {strategy!r}")
