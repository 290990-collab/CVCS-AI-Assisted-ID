"""Graph augmentation for contrastive (InfoNCE) training.

Two views of the same graph are positives against the other plans in the batch. Each
augmentation teaches an invariance, so it should match the evaluation ground truth:
- rotation / reflection: exactly invariant on all three relevance axes (composition and
  topology ignore coordinates; geometry components are symmetric).
- geometric jitter: nearly invariant.
- edge drop: partially invariant (removes the topology axis); keep low.
- node drop: not invariant for full retrieval (changes composition) but makes the model
  robust to incomplete queries; counterpart of the vision masking.
- feature mask: harmless on geometry columns; on the type one-hot it teaches to ignore
  room type (composition axis).

On 4-8 node graphs probabilities need care: with 5 nodes and `node_drop=0.1` no node is
dropped with probability 0.9^5 ~ 59%, so views are often identical and the task near trivial.

Views keep the node count and `batch` vector (dropped nodes are zeroed and isolated, not
removed), so both views yield the same B graphs in the same order and InfoNCE rows stay
aligned. With `add` pooling a zeroed isolated node contributes 0.

Asymmetric pairs (`pair_mode="asym_partial"`): view A = whole graph (rotations/reflections
only), view B = the same graph with a fraction f ~ U[`partial_frac_min`, `partial_frac_max`]
of rooms truly removed (induced subgraph, edges renumbered), as the evaluation partial
(`graph_partial_query.make_partial_graph`: same `round(f*n)`, unchanged survivor features).
At least one room always survives. node_drop/edge_drop/feat_mask/geom_jitter do not apply in
this mode. Default `symmetric`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from src.data.rplan_metadata import NUM_ROOM_TYPES
from src.graph.transforms import LOST_MARKER_COL, lost_neighbor_count

# geometric columns [cx, cy, w, h, area, aspect] follow the type one-hot
# (as graph_builder._node_features, transforms._GEOM_START)
_GEOM_START = NUM_ROOM_TYPES
_GEOM_DIM = 6
_CX, _CY, _W, _H = 0, 1, 2, 3   # indices within the geometric block


@dataclass(frozen=True)
class AugmentParams:
    """Augmentation strengths for one view.

    Args:
        node_drop: expected fraction of zeroed/isolated nodes.
        edge_drop: expected fraction of surviving edges removed.
        feat_mask: probability of zeroing a single (node, column) cell.
        geom_jitter: std of Gaussian noise on the 6 geometric columns, in input units
            (z-scores when `normalize=True`).
        flip_prob: probability of reflecting the plan (vertical axis).
        rot_prob: probability of rotating by 90/180/270 degrees (k uniform in 1..3).
    """

    node_drop: float = 0.1
    edge_drop: float = 0.1
    feat_mask: float = 0.1
    geom_jitter: float = 0.0
    flip_prob: float = 0.0
    rot_prob: float = 0.0
    pair_mode: str = "symmetric"        # symmetric | asym_partial
    partial_frac_min: float = 0.25
    partial_frac_max: float = 0.75
    lost_marker: bool = False           # "lost neighbours" column in view B

    @classmethod
    def from_cfg(cls, cfg) -> "AugmentParams":
        """Build from the training config (Namespace/object)."""
        return cls(
            node_drop=cfg.node_drop,
            edge_drop=cfg.edge_drop,
            feat_mask=cfg.feat_mask,
            geom_jitter=getattr(cfg, "geom_jitter", 0.0),
            flip_prob=getattr(cfg, "flip_prob", 0.0),
            rot_prob=getattr(cfg, "rot_prob", 0.0),
            pair_mode=getattr(cfg, "pair_mode", "symmetric"),
            partial_frac_min=getattr(cfg, "partial_frac_min", 0.25),
            partial_frac_max=getattr(cfg, "partial_frac_max", 0.75),
            lost_marker=getattr(cfg, "lost_marker", False),
        )

    @property
    def uses_geometry(self) -> bool:
        """True if any augmentation touches the geometric block."""
        return self.geom_jitter > 0.0 or self.flip_prob > 0.0 or self.rot_prob > 0.0


def _rand(shape, generator, device) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device=device)


def _apply_geometric_symmetry(
    geom: torch.Tensor,
    node_graph: torch.Tensor,
    num_graphs: int,
    params: AugmentParams,
    generator: torch.Generator,
) -> torch.Tensor:
    """Reflection/rotation of raw geom [N, 6] (grid coords in [0,1]), drawn per graph via `node_graph` [N].

    Reflection: cx -> 1 - cx. Rotation by 90 degrees: (cx, cy) -> (1 - cy, cx), (w, h) -> (h, w);
    area and aspect are unchanged.
    """
    device = geom.device

    if params.flip_prob > 0.0:
        flip = (_rand(num_graphs, generator, device) < params.flip_prob)[node_graph]
        geom[flip, _CX] = 1.0 - geom[flip, _CX]

    if params.rot_prob > 0.0:
        do_rot = _rand(num_graphs, generator, device) < params.rot_prob
        # k uniform in {1,2,3} on selected graphs, 0 elsewhere
        k_graph = torch.randint(1, 4, (num_graphs,), generator=generator, device=device)
        k_graph = torch.where(do_rot, k_graph, torch.zeros_like(k_graph))
        k_node = k_graph[node_graph]

        # apply the 90-degree rotation up to 3 times, only to nodes with turns left
        for _ in range(3):
            todo = k_node > 0
            if not bool(todo.any()):
                break
            cx = geom[todo, _CX].clone()
            cy = geom[todo, _CY].clone()
            geom[todo, _CX] = 1.0 - cy
            geom[todo, _CY] = cx
            w = geom[todo, _W].clone()
            h = geom[todo, _H].clone()
            geom[todo, _W] = h
            geom[todo, _H] = w
            k_node = k_node - todo.long()

    return geom


def augment_view(
    batch,
    params: AugmentParams,
    generator: torch.Generator,
    geom_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> Data:
    """One augmented view of a graph batch (same `batch` vector: same graphs, same order).

    Args:
        batch: PyG `Batch`.
        generator: torch.Generator for reproducibility.
        geom_stats: (mean, std) of the 6 geometric columns, each [1,6], or None for raw data
            (`--no-normalize`). Used only by reflection/rotation, defined on raw [0,1]
            coordinates: z-scored features are de-normalised, transformed, re-normalised
            (std(cy)/std(cx) = 1.24, std(h)/std(w) = 1.26 on RPLAN, so swapping z-scored
            columns would distort the scale by ~25%).
    """
    device = batch.x.device
    x = batch.x.clone()
    edge_index = batch.edge_index
    edge_attr = getattr(batch, "edge_attr", None)

    num_nodes = x.size(0)
    num_edges = edge_index.size(1)

    # geometric symmetries + jitter, before node drop
    if params.uses_geometry:
        # symmetries modify in place: work on a copy
        geom = x[:, _GEOM_START:_GEOM_START + _GEOM_DIM].clone()

        node_graph = getattr(batch, "batch", None)
        if node_graph is None:
            node_graph = x.new_zeros(num_nodes, dtype=torch.long)

        if params.flip_prob > 0.0 or params.rot_prob > 0.0:
            # symmetries act on raw [0,1] coordinates: de-normalise, transform, re-normalise
            if geom_stats is not None:
                mean, std = geom_stats
                geom = geom * std + mean
            geom = _apply_geometric_symmetry(
                geom, node_graph, int(node_graph.max()) + 1, params, generator
            )
            if geom_stats is not None:
                geom = (geom - mean) / std

        if params.geom_jitter > 0.0:
            noise = torch.randn(
                geom.shape, generator=generator, device=device
            ) * params.geom_jitter
            geom = geom + noise

        x[:, _GEOM_START:_GEOM_START + _GEOM_DIM] = geom

    keep_node = _rand(num_nodes, generator, device) >= params.node_drop
    x[~keep_node] = 0.0

    # keep edges with both endpoints alive
    edge_keep = keep_node[edge_index[0]] & keep_node[edge_index[1]]

    if params.edge_drop > 0.0:
        edge_rand = _rand(num_edges, generator, device) >= params.edge_drop
        edge_keep = edge_keep & edge_rand

    # same mask on edge_index and edge_attr keeps them aligned
    new_edge_index = edge_index[:, edge_keep]
    new_edge_attr = edge_attr[edge_keep] if edge_attr is not None else None

    # per-cell mask (a per-column batch-wide mask would shift the space uniformly)
    if params.feat_mask > 0.0:
        cell_keep = _rand(x.shape, generator, device) >= params.feat_mask
        x = x * cell_keep

    view = Data(x=x, edge_index=new_edge_index)
    if new_edge_attr is not None:
        view.edge_attr = new_edge_attr
    view.batch = batch.batch
    return view


def keep_subgraph(batch, keep: torch.Tensor, lost_marker: bool = False) -> Data:
    """Induced subgraph on `keep` [N] bool: nodes truly removed, edges renumbered, `batch` filtered.

    With `lost_marker=True` LOST_MARKER_COL receives the distinct removed-neighbour count,
    computed on the full graph (as `graph_partial_query.lost_marker_for`).
    """
    if lost_marker:
        if batch.x.size(1) != LOST_MARKER_COL + 1:
            raise ValueError(
                f"keep_subgraph(lost_marker=True): x ha {batch.x.size(1)} colonne, "
                f"attese {LOST_MARKER_COL + 1} (manca AppendLostMarker nella transform?)"
            )
        count = lost_neighbor_count(batch.edge_index, keep)
    edge_attr = getattr(batch, "edge_attr", None)
    edge_index, edge_attr = subgraph(keep, batch.edge_index, edge_attr,
                                     relabel_nodes=True, num_nodes=batch.x.size(0))
    if lost_marker:
        x = batch.x[keep].clone()
        x[:, LOST_MARKER_COL] = count[keep].to(x.dtype)
        view = Data(x=x, edge_index=edge_index)
    else:
        view = Data(x=batch.x[keep], edge_index=edge_index)
    if edge_attr is not None:
        view.edge_attr = edge_attr
    node_graph = getattr(batch, "batch", None)
    if node_graph is None:
        node_graph = batch.x.new_zeros(batch.x.size(0), dtype=torch.long)
    view.batch = node_graph[keep]
    return view


def remove_rooms_view(batch, params: AugmentParams, generator: torch.Generator) -> Data:
    """Partial view: removes round(f*n) random rooms per graph, f ~ U[partial_frac_min, partial_frac_max].

    Rounding as `select_rooms_to_remove` (`random`): half-to-even. At least one room is
    kept, since an empty graph would vanish from pooling and misalign InfoNCE rows.
    """
    device = batch.x.device
    node_graph = getattr(batch, "batch", None)
    if node_graph is None:
        node_graph = batch.x.new_zeros(batch.x.size(0), dtype=torch.long)
    num_graphs = int(node_graph.max()) + 1
    counts = torch.bincount(node_graph, minlength=num_graphs)

    frac = params.partial_frac_min + (params.partial_frac_max - params.partial_frac_min) \
        * _rand(num_graphs, generator, device)
    n_remove = torch.round(frac * counts.float()).long()
    n_remove = torch.minimum(n_remove, counts - 1).clamp_min(0)

    # random rank of each node within its graph: sort by (graph, u)
    u = _rand(node_graph.numel(), generator, device)
    order = torch.argsort(node_graph.double() * 2.0 + u.double())
    ptr = torch.cumsum(counts, 0) - counts                       # first node of each graph
    rank = torch.empty_like(node_graph)
    rank[order] = torch.arange(node_graph.numel(), device=device) - ptr[node_graph[order]]
    keep = rank >= n_remove[node_graph]
    # the marker draws no random numbers: same seed removes the same rooms as `asym`
    return keep_subgraph(batch, keep, lost_marker=params.lost_marker)


def two_views(
    batch,
    params: AugmentParams,
    generator: torch.Generator,
    geom_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[Data, Data]:
    """Positive pair (view_a, view_b); see `augment_view`.

    `symmetric`: two independent views. `asym_partial`: A = whole graph (flip/rot only),
    B = graph with a fraction of rooms removed.
    """
    if params.lost_marker and params.pair_mode != "asym_partial":
        raise ValueError(
            f"lost_marker=True richiede pair_mode='asym_partial' (dato: '{params.pair_mode}')"
        )
    if params.pair_mode == "asym_partial":
        whole = AugmentParams(node_drop=0.0, edge_drop=0.0, feat_mask=0.0, geom_jitter=0.0,
                              flip_prob=params.flip_prob, rot_prob=params.rot_prob)
        return (augment_view(batch, whole, generator, geom_stats),
                remove_rooms_view(batch, params, generator))
    if params.pair_mode != "symmetric":
        raise ValueError(f"pair_mode='{params.pair_mode}' (attesi: symmetric | asym_partial)")
    return (
        augment_view(batch, params, generator, geom_stats),
        augment_view(batch, params, generator, geom_stats),
    )
