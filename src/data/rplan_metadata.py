"""Structural metadata of RPLAN floor plans (from the `.mat` files), linked to the PNGs.

- composition: `room_types`, `type_histogram`;
- topology: `edges` (i, j, relation type), `typed_adjacency(...)`;
- geometry: `boxes`, `footprint`, `entrance` and derived `footprint_area`, `footprint_aspect`, `type_area_distribution`.

Used as ground truth for ranking evaluation.

`.mat` fields (plan with R rooms):
- name     : scalar, PNG stem;
- rType    : (R,) room type 0..12;
- rEdge    : (E,3) [room_i, room_j, relation]; relation = adjacency type 0..9;
- gtBoxNew : (R,4) room bbox [xmin, ymin, xmax, ymax] on a 256 grid;
- gtBox    : (R+1,4) room bboxes + one extra row = global bbox (footprint);
- boundary : (P,4) outer contour [x, y, direction 0..3, door flag 0/1]; flag==1 rows mark the entrance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

# one aggregated .mat per split (struct array), merged into a single name -> metadata index
DEFAULT_MAT_DIR = Path(
    "/work/cvcs2026/ai_interior_design/datasets/RPLAN/Network/data"
)
_SPLITS = ("train", "valid", "test")

# grid of all RPLAN coordinates
_GRID = 256.0

NUM_ROOM_TYPES = 13

ROOM_TYPES: dict[int, str] = {
    0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom",
    4: "DiningRoom", 5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom",
    8: "GuestRoom", 9: "Balcony", 10: "Entrance", 11: "Storage", 12: "Wall-in",
}


@dataclass(frozen=True)
class RoomMeta:
    """Full structural description of a plan (frozen)."""

    name: str
    # official RPLAN split
    split: str
    # one type id per room
    room_types: tuple[int, ...]
    # (room_i, room_j, relation)
    edges: tuple[tuple[int, int, int], ...]
    # room bboxes [xmin, ymin, xmax, ymax], aligned with room_types
    boxes: tuple[tuple[int, int, int, int], ...]
    # global bbox [xmin, ymin, xmax, ymax]
    footprint: tuple[int, int, int, int]
    # entrance centre in [0,1]^2, or None
    entrance: tuple[float, float] | None

    # --- composition ---

    @property
    def num_rooms(self) -> int:
        return len(self.room_types)

    @property
    def type_histogram(self) -> tuple[int, ...]:
        """Room count per type, fixed length 13."""
        counts = Counter(self.room_types)
        return tuple(counts.get(t, 0) for t in range(NUM_ROOM_TYPES))

    # --- topology ---

    def typed_adjacency(self, include_relation: bool = False) -> Counter:
        """Counter of adjacencies as ordered (min, max) room-type pairs, permutation-invariant.

        `include_relation` adds the relation type (0..9) to the key.
        """
        out: Counter = Counter()
        for i, j, rel in self.edges:
            ti, tj = self.room_types[i], self.room_types[j]
            pair = (min(ti, tj), max(ti, tj))
            key = (*pair, rel) if include_relation else pair
            out[key] += 1
        return out

    # --- geometry ---

    @property
    def footprint_area(self) -> float:
        """Global bbox area, normalised by the 256 grid to [0,1]."""
        xmin, ymin, xmax, ymax = self.footprint
        return max(xmax - xmin, 0) * max(ymax - ymin, 0) / (_GRID * _GRID)

    @property
    def footprint_aspect(self) -> float:
        """Footprint aspect ratio (long / short side), >= 1."""
        xmin, ymin, xmax, ymax = self.footprint
        w, h = max(xmax - xmin, 1), max(ymax - ymin, 1)
        return max(w, h) / min(w, h)

    @property
    def type_area_distribution(self) -> tuple[float, ...]:
        """Fraction of area per room type, length 13."""
        areas = [0.0] * NUM_ROOM_TYPES
        for t, (xmin, ymin, xmax, ymax) in zip(self.room_types, self.boxes):
            areas[t] += max(xmax - xmin, 0) * max(ymax - ymin, 0)
        total = sum(areas)
        if total <= 0:
            return tuple(areas)
        return tuple(a / total for a in areas)


def _parse_entrance(boundary: np.ndarray) -> tuple[float, float] | None:
    """Normalised centre of the entrance segment (rows with flag==1)."""
    b = np.atleast_2d(boundary)
    if b.shape[1] < 4:
        return None
    door = b[b[:, 3] == 1]
    if door.size == 0:
        return None
    cx = float(door[:, 0].mean()) / _GRID
    cy = float(door[:, 1].mean()) / _GRID
    return (cx, cy)


@lru_cache(maxsize=1)
def _load_index(mat_dir: str) -> dict[str, RoomMeta]:
    """Load the 3 `.mat` once into {name -> RoomMeta}."""
    index: dict[str, RoomMeta] = {}
    for split in _SPLITS:
        mat = sio.loadmat(
            Path(mat_dir) / f"data_{split}.mat",
            squeeze_me=True,
            struct_as_record=False,
        )
        for rec in mat["data"]:
            room_types = tuple(int(x) for x in np.atleast_1d(rec.rType))

            edge_rows = np.atleast_2d(rec.rEdge)
            edges = (
                tuple((int(a), int(b), int(rel)) for a, b, rel in edge_rows)
                if edge_rows.size
                else ()
            )

            box_rows = np.atleast_2d(rec.gtBoxNew)
            boxes = tuple(
                (int(x0), int(y0), int(x1), int(y1)) for x0, y0, x1, y1 in box_rows
            )

            # last gtBox row = global footprint
            gtbox = np.atleast_2d(rec.gtBox)
            fp = gtbox[-1]
            footprint = (int(fp[0]), int(fp[1]), int(fp[2]), int(fp[3]))

            entrance = _parse_entrance(rec.boundary)

            index[str(rec.name)] = RoomMeta(
                name=str(rec.name),
                split=split,
                room_types=room_types,
                edges=edges,
                boxes=boxes,
                footprint=footprint,
                entrance=entrance,
            )
    return index


def load_metadata(
    png_path: str | Path,
    mat_dir: str | Path = DEFAULT_MAT_DIR,
) -> RoomMeta | None:
    """Metadata of the plan for `png_path` (path or stem); None if the PNG has no `.mat` record."""
    stem = Path(png_path).stem
    return _load_index(str(mat_dir)).get(stem)


def get_split(
    png_path: str | Path,
    mat_dir: str | Path = DEFAULT_MAT_DIR,
) -> str | None:
    """Official split ("train" | "valid" | "test") of `png_path`, or None without `.mat` record.

    `snapshot_train/` mixes all three splits: use this lookup to separate valid/test queries.
    """
    meta = load_metadata(png_path, mat_dir)
    return meta.split if meta is not None else None


def split_row_indices(
    paths: list[str | Path],
    split: str,
    mat_dir: str | Path = DEFAULT_MAT_DIR,
) -> list[int]:
    """Ascending row indices of `paths` (gallery order) belonging to `split`; plans without `.mat` are excluded."""
    if split not in _SPLITS:
        raise ValueError(f"split {split!r} non valido (usa {_SPLITS})")
    return [i for i, p in enumerate(paths) if get_split(p, mat_dir) == split]
