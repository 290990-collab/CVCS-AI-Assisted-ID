"""RoomMeta (RPLAN .mat metadata) to PyTorch Geometric graph adapter.

Does not read the .mat files: relies on `src.data.rplan_metadata.load_metadata`.

Mapping RoomMeta -> Data:
- room_types -> node features: 13-dim one-hot.
- boxes -> node features: centre, w, h, area, aspect, normalised on the 256 grid.
- edges -> edge_index + edge_attr (10-dim one-hot relation); undirected (symmetrised).
- footprint -> graph-level area/aspect.
- type_histogram -> graph-level per-type room counts (training-free baseline).

Axes: `boxes` is [x0,y0,x1,y1]; `footprint` has swapped axes [y0,x0,y1,x1]. Node geometry
uses only `boxes`; footprint contributes only area and aspect (swap-invariant).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from src.data.rplan_metadata import (
    NUM_ROOM_TYPES,
    RoomMeta,
    load_metadata,
)

# Relation types: 3rd column of rEdge (0..9).
NUM_RELATION_TYPES = 10

# RPLAN coordinate grid.
_GRID = 256.0

# [cx, cy, w, h, area, aspect]
_NODE_GEOM_DIM = 6

NODE_FEATURE_DIM = NUM_ROOM_TYPES + _NODE_GEOM_DIM


def _node_features(meta: RoomMeta) -> torch.Tensor:
    """Node features [N, NODE_FEATURE_DIM]: type one-hot + geometry normalised on the 256 grid."""
    num_nodes = meta.num_rooms
    x = torch.zeros(num_nodes, NODE_FEATURE_DIM, dtype=torch.float)

    for i, (room_type, box) in enumerate(zip(meta.room_types, meta.boxes)):
        x[i, room_type] = 1.0

        x0, y0, x1, y1 = box
        w = max(x1 - x0, 0) / _GRID
        h = max(y1 - y0, 0) / _GRID
        cx = ((x0 + x1) / 2.0) / _GRID
        cy = ((y0 + y1) / 2.0) / _GRID
        area = w * h
        # aspect >= 1, orientation-invariant; guard against zero sides
        long_side, short_side = max(w, h), max(min(w, h), 1e-6)
        aspect = long_side / short_side

        geom_start = NUM_ROOM_TYPES
        x[i, geom_start:geom_start + _NODE_GEOM_DIM] = torch.tensor(
            [cx, cy, w, h, area, aspect], dtype=torch.float
        )

    return x


def _edge_index_and_attr(
    meta: RoomMeta,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Undirected edge_index [2, E'] and one-hot edge_attr [E', NUM_RELATION_TYPES]."""
    num_nodes = meta.num_rooms

    if len(meta.edges) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, NUM_RELATION_TYPES, dtype=torch.float)
        return edge_index, edge_attr

    sources: list[int] = []
    targets: list[int] = []
    relations: list[int] = []
    for i, j, relation in meta.edges:
        sources.append(i)
        targets.append(j)
        relations.append(relation)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    # clamp guards against out-of-range relation ids
    relation_ids = torch.tensor(relations, dtype=torch.long).clamp(
        0, NUM_RELATION_TYPES - 1
    )
    edge_attr = F.one_hot(relation_ids, num_classes=NUM_RELATION_TYPES).float()

    # parallel edges are merged by mean
    edge_index, edge_attr = to_undirected(
        edge_index, edge_attr, num_nodes=num_nodes, reduce="mean"
    )
    return edge_index, edge_attr


def build_graph(meta: RoomMeta) -> Data:
    """RoomMeta to PyG `Data`; graph-level attributes (name, split, footprint, type_histogram) stay out of message passing."""
    x = _node_features(meta)
    edge_index, edge_attr = _edge_index_and_attr(meta)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    data.name = meta.name
    data.split = meta.split
    data.num_rooms = meta.num_rooms
    data.footprint_area = float(meta.footprint_area)
    data.footprint_aspect = float(meta.footprint_aspect)
    data.type_histogram = torch.tensor(
        meta.type_histogram, dtype=torch.float
    ).unsqueeze(0)

    return data


def build_graph_from_png(
    png_path: str | Path,
) -> Data | None:
    """Graph for a gallery PNG; None if it has no linkable .mat record (~0.1%)."""
    meta = load_metadata(png_path)
    if meta is None:
        return None
    return build_graph(meta)
