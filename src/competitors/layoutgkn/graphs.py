"""RoomMeta plans (`.mat`) -> graphs in the LayoutGKN format.

Built from RoomMeta (upstream's script needs the RPLAN semantic images we lack), so room removal
matches our graph branch (`select_rooms_to_remove`, `random.Random(partial_seed + qi)`) and queries are paired.

Graph fields (read by upstream `data.prep_data`):
- `category`     long [n]    rType 0..12 -> 8 classes via upstream `CAT_MAP`;
- `geometry`     float [n,6] [cx, cy, w, h, sqrt(area), perimeter/4] of the room box (`gtBoxNew`),
                 upstream vertical flip and transform (shift 128, scale 1.8/256);
- `edge_index`   long [2,E]  `rEdge` adjacency, undirected, no self-loops;
- `connectivity` long [E]    all 0 (adjacency only, no doors);
- `shp`          float [n,16] GraphHopper matrix M[v, a, b] (a = position of v on the path, b = length),
                 b < 4 flattened as upstream (`delta = 4`), over all edges, recomputed after room removal.

Deviations: no doors, rectangular rooms, paths on adjacency.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch_geometric.data import Data

from src.competitors.layoutgkn import upstream
from src.data.rplan_metadata import RoomMeta
from src.graph.graph_partial_query import filter_meta
from src.vision.data.vision_partial_query import select_rooms_to_remove

upstream()
from LayoutGKN.constants import CAT_MAP, CAT_RPLAN  # noqa: E402

GRID = 256.0
SHIFT = 128.0
SCALE = 1 / 256 * 18 / 10       # upstream `transform_polygon`
SHP_DELTA = 4                   # upstream: max path length considered
NUM_CATS = len(CAT_RPLAN)       # 8
GEOM_DIM = 6


def geometry_features(boxes) -> np.ndarray:
    """[n, 6] upstream geometry features of room boxes [x0, y0, x1, y1] (flipud, then (p - 128) * 1.8/256)."""
    out = np.zeros((len(boxes), GEOM_DIM), dtype=np.float64)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        fy0, fy1 = GRID - y1, GRID - y0
        w = max(x1 - x0, 0) * SCALE
        h = max(fy1 - fy0, 0) * SCALE
        cx = ((x0 + x1) / 2.0 - SHIFT) * SCALE
        cy = ((fy0 + fy1) / 2.0 - SHIFT) * SCALE
        out[i] = (cx, cy, w, h, np.sqrt(w * h), (w + h) / 2.0)
    return out


def undirected_pairs(meta: RoomMeta) -> list[tuple[int, int]]:
    """Sorted unique (i, j) with i < j from `meta.edges` (relation type dropped)."""
    pairs = {(min(a, b), max(a, b)) for a, b, _ in meta.edges if a != b}
    return sorted(pairs)


def shortest_path_matrix(n: int, pairs, delta: int = SHP_DELTA) -> np.ndarray:
    """GraphHopper M[v, a, b] for path lengths b < delta, shape [n, delta, delta].

    M[v, a, b] = number of shortest paths (ordered pairs s, t, s = t included) of length b with v at
    position a: sum_{s,t: d(s,t)=b, d(s,v)=a, d(v,t)=b-a} sigma(s,v) * sigma(v,t), sigma = #shortest paths.
    Same as grakel `GraphHopper.parse_input`; distances from walk counts A^k, k < delta.
    """
    A = np.zeros((n, n), dtype=np.int64)
    for i, j in pairs:
        A[i, j] = A[j, i] = 1
    D = np.full((n, n), -1, dtype=np.int64)       # -1: farther than delta-1 or unreachable
    sigma = np.zeros((n, n), dtype=np.int64)
    P = np.eye(n, dtype=np.int64)
    for k in range(delta):
        new = (D < 0) & (P > 0)
        D[new] = k
        sigma[new] = P[new]
        P = P @ A
    M = np.zeros((n, delta, delta), dtype=np.float64)
    for b in range(delta):
        mask_st = (D == b)
        for a in range(b + 1):
            S = np.where(D == a, sigma, 0)          # [s, v]
            T = np.where(D == b - a, sigma, 0)      # [v, t]
            M[:, a, b] = np.einsum("sv,vt,st->v", S, T, mask_st)
    return M


def to_lgkn_graph(meta: RoomMeta) -> Data:
    """RoomMeta -> upstream graph fields."""
    n = meta.num_rooms
    pairs = undirected_pairs(meta)
    if pairs:
        src = [i for i, j in pairs] + [j for i, j in pairs]
        dst = [j for i, j in pairs] + [i for i, j in pairs]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    shp = shortest_path_matrix(n, pairs).reshape(n, -1)
    return Data(
        edge_index=edge_index,
        geometry=torch.tensor(geometry_features(meta.boxes), dtype=torch.float),
        category=torch.tensor([CAT_MAP[int(t)] for t in meta.room_types], dtype=torch.long),
        connectivity=torch.zeros(edge_index.shape[1], dtype=torch.long),
        shp=torch.tensor(shp, dtype=torch.float),
        num_nodes=n,
    )


def damaged_lgkn_graph(meta: RoomMeta, fraction: float, rng: random.Random):
    """(graph | None, removed), same room choice as `make_partial_graph`; `rng` = `random.Random(partial_seed + qi)`.

    None when every room is removed.
    """
    removed = select_rooms_to_remove(meta, "random", {"fraction": fraction}, rng)
    reduced = filter_meta(meta, removed)
    if reduced is None:
        return None, removed
    return to_lgkn_graph(reduced), removed
