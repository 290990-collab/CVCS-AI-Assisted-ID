"""Degraded graph query: graph counterpart of the vision partial query.

Builds the graph of an incomplete floor plan from a plan and a masking strategy.
Room selection reuses `select_rooms_to_remove` from the vision branch so that both
branches remove the same rooms (paired comparison). Pairing needs the shared gallery
order: per-query seed is `seed + qi`, `qi` being the gallery row.

- The footprint stays that of the full plan (as the degraded image keeps its canvas).
- Surviving node features are unchanged (normalised on the fixed 256 grid).
- Empty graph (all rooms removed): returns None; the caller counts and skips it.
"""

from __future__ import annotations

import random

import torch
from torch_geometric.data import Data

from src.data.rplan_metadata import RoomMeta
from src.graph.graph_builder import build_graph
from src.graph.transforms import lost_neighbor_count
from src.vision.data.vision_partial_query import select_rooms_to_remove


def filter_meta(meta: RoomMeta, removed: list[int] | set[int]) -> RoomMeta | None:
    """`RoomMeta` without `removed` rooms; edges touching them are dropped, the rest reindexed. None if empty."""
    removed = set(removed)
    keep = [i for i in range(meta.num_rooms) if i not in removed]
    if not keep:
        return None

    remap = {old: new for new, old in enumerate(keep)}
    return RoomMeta(
        name=meta.name,
        split=meta.split,
        room_types=tuple(meta.room_types[i] for i in keep),
        edges=tuple(
            (remap[a], remap[b], rel)
            for a, b, rel in meta.edges
            if a in remap and b in remap
        ),
        boxes=tuple(meta.boxes[i] for i in keep),
        footprint=meta.footprint,  # unchanged
        entrance=meta.entrance,
    )


def lost_marker_for(meta: RoomMeta, removed: list[int] | set[int]) -> torch.Tensor:
    """Distinct removed neighbours per surviving room, float [n_kept].

    Counted on the full symmetrised graph (as in training), in original index order.
    """
    removed = set(removed)
    keep = torch.tensor([i not in removed for i in range(meta.num_rooms)], dtype=torch.bool)
    count = lost_neighbor_count(build_graph(meta).edge_index, keep)
    return count[keep].float()


def make_partial_graph(
    meta: RoomMeta,
    strategy: str,
    params: dict,
    rng: random.Random,
    lost_marker: bool = False,
) -> tuple[Data | None, list[int]]:
    """Degraded graph and removed room indices; same signature as the vision `make_partial_query`.

    Args:
        strategy: "random" | "semantic" | "topology".
        params: strategy parameters (`fraction`, `keep_types`, `max_degree`).
        rng: per-query seeded `random.Random` (`seed + qi`).
        lost_marker: attach `graph.lost_marker` [n_kept] (see `lost_marker_for`).

    Returns:
        (graph, removed); graph is None if the plan is emptied.
    """
    removed = select_rooms_to_remove(meta, strategy, params, rng)
    reduced = filter_meta(meta, removed)
    if reduced is None:
        return None, removed
    graph = build_graph(reduced)
    if lost_marker:
        graph.lost_marker = lost_marker_for(meta, removed)
    return graph, removed
