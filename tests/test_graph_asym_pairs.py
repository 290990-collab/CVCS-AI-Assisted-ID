# tests/test_graph_asym_pairs.py

"""
Smoke test CPU dell'opzione D (status.md § 30): coppie asimmetriche nel training
del graph (`pair_mode="asym_partial"` in `src/graph/training/augment.py`).

Invarianti:
1. il sottografo del training coincide con il grafo parziale della valutazione
   (`graph_partial_query.filter_meta` + `build_graph`) per lo stesso insieme di
   stanze tolte: stesse feature, stessi archi;
2. `round(f*n)` stanze tolte con f in [0.25, 0.75], mai tutte: ogni grafo del
   batch sopravvive e le righe di InfoNCE restano allineate;
3. il default `symmetric` e' bit-identico al comportamento storico;
4. la vista A e' intera (stessi nodi e archi), e il GCN produce B embedding.

Esecuzione: python -m pytest tests/test_graph_asym_pairs.py -v
"""

from __future__ import annotations

import random

import pytest
import torch
from torch_geometric.data import Batch

from src.data.rplan_metadata import RoomMeta
from src.graph.graph_builder import NODE_FEATURE_DIM, build_graph
from src.graph.graph_partial_query import filter_meta
from src.graph.models import build_graph_encoder
from src.graph.training.augment import (
    AugmentParams,
    augment_view,
    keep_subgraph,
    remove_rooms_view,
    two_views,
)


def _meta(name, n, rng):
    types = [rng.randrange(0, 13) for _ in range(n)]
    edges = [(i, i + 1, rng.randrange(0, 9)) for i in range(n - 1)]
    edges += [(0, n - 1, 3)] if n > 3 else []
    boxes = [(10 * i, 5, 10 * i + 9, 30 + i) for i in range(n)]
    return RoomMeta(name=name, split="train", room_types=tuple(types), edges=tuple(edges),
                    boxes=tuple(boxes), footprint=(0, 0, 10 * n, 40), entrance=None)


def _batch(sizes=(4, 5, 6, 7, 8, 4), seed=0):
    rng = random.Random(seed)
    metas = [_meta(f"p{i}", n, rng) for i, n in enumerate(sizes)]
    graphs = [build_graph(m) for m in metas]
    return metas, Batch.from_data_list([Data_(g) for g in graphs])


def Data_(g):
    # solo i campi del message passing (i metadati stringa non servono qui)
    from torch_geometric.data import Data
    return Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr)


def _edge_set(data):
    return {(int(a), int(b), tuple(e.tolist()))
            for (a, b), e in zip(data.edge_index.t(), data.edge_attr)}


def test_subgraph_matches_partial_graph_of_evaluation():
    metas, batch = _batch()
    rng = random.Random(1)
    keep = torch.ones(batch.x.size(0), dtype=torch.bool)
    removed_by_graph = []
    for gi, m in enumerate(metas):
        removed = sorted(rng.sample(range(m.num_rooms), m.num_rooms // 2))
        removed_by_graph.append(removed)
        start = int(batch.ptr[gi])
        for r in removed:
            keep[start + r] = False
    view = keep_subgraph(batch, keep)
    for gi, (m, removed) in enumerate(zip(metas, removed_by_graph)):
        ref = build_graph(filter_meta(m, removed))
        mask = view.batch == gi
        idx = torch.nonzero(mask).flatten()
        assert torch.equal(view.x[idx], ref.x)                       # feature bit-identiche
        emask = mask[view.edge_index[0]]
        sub_ei = view.edge_index[:, emask] - int(idx.min())
        sub = type(ref)(edge_index=sub_ei, edge_attr=view.edge_attr[emask])
        assert _edge_set(sub) == _edge_set(ref)                      # stessi archi


def test_remove_rooms_count_range_and_survival():
    _, batch = _batch(sizes=(2, 3, 4, 5, 6, 7, 8) * 20, seed=3)
    params = AugmentParams(pair_mode="asym_partial")
    gen = torch.Generator().manual_seed(0)
    counts = torch.bincount(batch.batch)
    view = remove_rooms_view(batch, params, gen)
    kept = torch.bincount(view.batch, minlength=counts.numel())
    removed = counts - kept
    assert bool((kept >= 1).all())                                   # nessun grafo sparisce
    lo = torch.round(0.25 * counts.float()).long()
    hi = torch.minimum(torch.round(0.75 * counts.float()).long(), counts - 1)
    assert bool((removed >= torch.minimum(lo, counts - 1)).all()) and bool((removed <= hi).all())
    assert int(view.batch.max()) + 1 == counts.numel()


def test_symmetric_default_is_bitwise_unchanged():
    _, batch = _batch()
    params = AugmentParams(node_drop=0.2, edge_drop=0.1, feat_mask=0.1)
    assert params.pair_mode == "symmetric"
    a1, b1 = two_views(batch, params, torch.Generator().manual_seed(7))
    g = torch.Generator().manual_seed(7)
    a2, b2 = augment_view(batch, params, g), augment_view(batch, params, g)
    for u, v in ((a1, a2), (b1, b2)):
        assert torch.equal(u.x, v.x) and torch.equal(u.edge_index, v.edge_index)


def test_asym_views_whole_and_partial_and_gcn_forward():
    _, batch = _batch()
    params = AugmentParams(node_drop=0.2, edge_drop=0.1, feat_mask=0.1, geom_jitter=0.1,
                           pair_mode="asym_partial")
    a, b = two_views(batch, params, torch.Generator().manual_seed(0))
    # vista A intera: stessi nodi, stessi archi, one-hot del tipo intatto (niente feat_mask)
    assert a.x.size(0) == batch.x.size(0) and torch.equal(a.edge_index, batch.edge_index)
    assert torch.equal(a.x[:, :13], batch.x[:, :13])
    assert b.x.size(0) < batch.x.size(0)
    enc = build_graph_encoder("gcn", in_dim=NODE_FEATURE_DIM, hidden_dim=16, out_dim=8,
                              num_layers=2, pooling="add", dropout=0.0, raw_skip=True)
    za, zb = enc(a), enc(b)
    assert za.shape == zb.shape == (batch.num_graphs, za.shape[1])


def test_unknown_pair_mode_is_rejected():
    _, batch = _batch()
    with pytest.raises(ValueError, match="pair_mode"):
        two_views(batch, AugmentParams(pair_mode="asym"), torch.Generator().manual_seed(0))
