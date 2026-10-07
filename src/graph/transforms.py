"""PyG transforms adjusting the graph before the GNN.

Cached graphs (`RplanGraphDataset`) are raw on purpose: these corrections apply on the fly
via `transform`, so one cache serves all variants and each piece can be ablated.

- NormalizeNodeGeometry: the 6 geometric columns have incompatible scales (`aspect` reaches
  37 on degenerate rooms, `area` has mean ~0.05); z-score per column, statistics from train
  only (no leakage), with `aspect` clipped at a high percentile first.
- RemoveSpuriousSelfLoops: a few spurious (i,i) edges (~46/8k) in the .mat; GCNConv/GATConv
  add self-loops themselves.

cx/cy are not recentred: RPLAN centres every plan in the canvas (footprint centre
0.500 +/- 0.001), so coordinates carry the position inside the apartment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.utils import remove_self_loops

from src.data.rplan_metadata import NUM_ROOM_TYPES

# geometric columns [cx, cy, w, h, area, aspect] follow the type one-hot
_GEOM_START = NUM_ROOM_TYPES
_GEOM_DIM = 6
_ASPECT_COL = _GEOM_START + _GEOM_DIM - 1  # absolute column of aspect

# lost-neighbours marker: extra column after the 19 `graph_builder` features, only with `lost_marker=True`
LOST_MARKER_COL = _GEOM_START + _GEOM_DIM  # = 19 = NODE_FEATURE_DIM
LOST_MARKER_DIM = 1

_EPS = 1e-6


class RemoveSpuriousSelfLoops(BaseTransform):
    """Remove (i, i) edges (and their edge_attr rows); convolutions add self-loops themselves."""

    def forward(self, data: Data) -> Data:
        data.edge_index, data.edge_attr = remove_self_loops(
            data.edge_index, data.edge_attr
        )
        return data


class NormalizeNodeGeometry(BaseTransform):
    """Z-score the 6 geometric node columns (type one-hot untouched).

    Clips `aspect` at `aspect_clip`, then (x - mean) / std with train-only statistics
    (see `compute_geometry_stats`).

    Args:
        mean, std: per-column [6], order cx,cy,w,h,area,aspect.
        aspect_clip: cap applied to `aspect` before the z-score.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        aspect_clip: float,
    ) -> None:
        self.mean = mean.view(1, _GEOM_DIM).float()
        # guard against constant columns
        self.std = std.view(1, _GEOM_DIM).float().clamp_min(_EPS)
        self.aspect_clip = float(aspect_clip)

    def forward(self, data: Data) -> Data:
        geom = data.x[:, _GEOM_START:_GEOM_START + _GEOM_DIM].clone()

        # clip the aspect tail
        geom[:, -1] = geom[:, -1].clamp_max(self.aspect_clip)

        geom = (geom - self.mean) / self.std

        data.x = data.x.clone()
        data.x[:, _GEOM_START:_GEOM_START + _GEOM_DIM] = geom
        return data


def lost_neighbor_count(edge_index: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Distinct removed neighbours per node, long [N].

    Computed on the full graph `edge_index` [2, E] (edges to removed rooms exist only there):
    (i, j) counts +1 for i if keep[i] and not keep[j]. Pairs are deduplicated (no coalesced
    input assumed), self-loops never count, removed nodes get 0.
    """
    num_nodes = keep.numel()
    count = torch.zeros(num_nodes, dtype=torch.long, device=keep.device)
    if edge_index.numel() == 0:
        return count
    src, dst = edge_index[0], edge_index[1]
    lost = keep[src] & ~keep[dst]
    pairs = torch.unique(src[lost] * num_nodes + dst[lost])   # distinct pairs
    count.scatter_add_(0, pairs // num_nodes, torch.ones_like(pairs))
    return count


class AppendLostMarker(BaseTransform):
    """Append the "lost neighbours" marker as the last column of `x`.

    Uses and removes the graph's `lost_marker` [N] if present (partial graph, see
    `graph_partial_query.make_partial_graph`), else zeros (whole graph).

    Raises: ValueError if `x` does not have exactly LOST_MARKER_COL columns.
    """

    def forward(self, data: Data) -> Data:
        if data.x.size(1) != LOST_MARKER_COL:
            raise ValueError(
                f"AppendLostMarker: x ha {data.x.size(1)} colonne, attese "
                f"{LOST_MARKER_COL} (marcatore gia' appeso?)"
            )
        marker = getattr(data, "lost_marker", None)
        if marker is None:
            col = data.x.new_zeros(data.x.size(0), LOST_MARKER_DIM)
        else:
            col = marker.to(dtype=data.x.dtype).view(-1, LOST_MARKER_DIM)
            del data.lost_marker
        data.x = torch.cat([data.x, col], dim=1)
        return data


def compute_geometry_stats(
    train_dataset,
    aspect_percentile: float = 99.0,
) -> dict[str, np.ndarray]:
    """Mean/std of the 6 geometric columns on train only, computed after clipping `aspect` at its percentile.

    Args:
        train_dataset: RplanGraphDataset(split="train") or iterable of Data.
        aspect_percentile: clip percentile for `aspect`.
    Returns: dict with "mean" [6], "std" [6], "aspect_clip" (scalar).
    """
    rows = [
        data.x[:, _GEOM_START:_GEOM_START + _GEOM_DIM].numpy()
        for data in train_dataset
    ]
    geom = np.concatenate(rows, axis=0)  # [num_train_nodes, 6]

    # clip first, then mean/std on clipped data
    aspect_clip = float(np.percentile(geom[:, -1], aspect_percentile))
    geom[:, -1] = np.minimum(geom[:, -1], aspect_clip)

    mean = geom.mean(axis=0)
    std = geom.std(axis=0)
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "aspect_clip": np.float32(aspect_clip),
    }


def save_geometry_stats(stats: dict[str, np.ndarray], path: str | Path) -> Path:
    """Write geometry stats (.npz), creating the folder; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stats)
    return path


def load_geometry_stats(path: str | Path) -> dict[str, np.ndarray]:
    """Load stats written by `save_geometry_stats`."""
    data = np.load(path)
    return {k: data[k] for k in ("mean", "std", "aspect_clip")}


def build_node_transform(
    normalize: bool = True,
    drop_self_loops: bool = True,
    stats: dict[str, np.ndarray] | None = None,
    lost_marker: bool = False,
) -> BaseTransform | None:
    """Compose the node transform from ablation flags; None if no step is active.

    Args:
        normalize: apply NormalizeNodeGeometry (requires `stats`).
        drop_self_loops: apply RemoveSpuriousSelfLoops.
        stats: from compute/load_geometry_stats.
        lost_marker: append AppendLostMarker as last step (x: 19 -> 20 columns).
    Raises: ValueError if normalize=True without stats.
    """
    steps: list[BaseTransform] = []

    if drop_self_loops:
        steps.append(RemoveSpuriousSelfLoops())

    if normalize:
        if stats is None:
            raise ValueError("normalize=True richiede `stats` (mean/std/aspect_clip)")
        steps.append(
            NormalizeNodeGeometry(
                mean=torch.as_tensor(stats["mean"]),
                std=torch.as_tensor(stats["std"]),
                aspect_clip=float(stats["aspect_clip"]),
            )
        )

    if lost_marker:
        steps.append(AppendLostMarker())   # last: must skip the z-score

    if not steps:
        return None
    if len(steps) == 1:
        return steps[0]
    return Compose(steps)
