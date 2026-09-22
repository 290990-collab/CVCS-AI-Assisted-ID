# tests/test_graph_lost_marker.py

"""
Smoke test CPU della variante `asymlost` (14 set 2026): il marcatore "vicini
persi" appeso come colonna 19 di `x` quando una stanza vicina viene tolta dalla
query parziale, e il suo contratto training <-> valutazione.

Invarianti coperte (vedi anche `src/graph/transforms.py:50-166`,
`src/graph/training/augment.py:108-373`, `src/graph/graph_partial_query.py:89-136`):

1. `lost_neighbor_count` conta coppie (i, j) DISTINTE con keep[i] e non keep[j],
   dedup su archi paralleli, self-loop esclusi, nodi tolti a 0 (T1).
2. Il conteggio del training (`keep_subgraph(lost_marker=True)` sul grafo intero
   trasformato) coincide bit per bit con quello della valutazione
   (`graph_partial_query.make_partial_graph` + transform), qualunque sia
   `drop_self_loops` (T2).
3. Sulla pianta intera (nessuna stanza tolta, f=0.0) il marcatore e' zero (T3).
4. Nelle coppie asimmetriche la vista A (grafo intero) ha SEMPRE marcatore zero,
   e accendere/spegnere il flag a parita' di seed cambia SOLO la colonna 19
   della vista B (T4).
5. Il default (flag spento) resta bit-identico al comportamento storico (T5).
6. Le guardie (ValueError) sui contratti rotti (T6).
7. Un checkpoint a in_dim 19 non si ricarica su un encoder a in_dim 20 e
   viceversa; il forward con in_dim 20 produce la shape attesa su entrambe le
   viste (T7).

Esecuzione: python -m pytest tests/test_graph_lost_marker.py -v
"""

from __future__ import annotations

import random

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph

from src.data.rplan_metadata import RoomMeta
from src.graph.graph_builder import NODE_FEATURE_DIM, NUM_RELATION_TYPES, build_graph
from src.graph.graph_partial_query import make_partial_graph
from src.graph.models import build_graph_encoder
from src.graph.training.augment import (
    AugmentParams,
    keep_subgraph,
    two_views,
)
from src.graph.transforms import (
    LOST_MARKER_COL,
    LOST_MARKER_DIM,
    AppendLostMarker,
    RemoveSpuriousSelfLoops,
    build_node_transform,
    lost_neighbor_count,
)

# ---------------------------------------------------------------------------
# Fixture (stile tests/test_graph_asym_pairs.py:40-64)
# ---------------------------------------------------------------------------


def _meta(name, n, rng):
    types = [rng.randrange(0, 13) for _ in range(n)]
    edges = [(i, i + 1, rng.randrange(0, 9)) for i in range(n - 1)]
    edges += [(0, n - 1, 3)] if n > 3 else []
    boxes = [(10 * i, 5, 10 * i + 9, 30 + i) for i in range(n)]
    return RoomMeta(name=name, split="train", room_types=tuple(types), edges=tuple(edges),
                    boxes=tuple(boxes), footprint=(0, 0, 10 * n, 40), entrance=None)


def _meta_with_self_loop_and_reverse(seed: int) -> RoomMeta:
    """5 stanze; un self-loop (2,2) e un arco duplicato in senso inverso (0,1)/(1,0)."""
    rng = random.Random(seed)
    n = 5
    types = [rng.randrange(0, 13) for _ in range(n)]
    edges = ((0, 1, 2), (1, 0, 2), (1, 2, 3), (2, 2, 0), (0, 3, 5), (3, 4, 1))
    boxes = tuple((10 * i, 5, 10 * i + 9, 30 + i) for i in range(n))
    return RoomMeta(name=f"self{seed}", split="train", room_types=tuple(types), edges=edges,
                    boxes=boxes, footprint=(0, 0, 10 * n, 40), entrance=None)


def _plain_data(g) -> Data:
    """Solo i campi del message passing (come Data_ in test_graph_asym_pairs.py)."""
    return Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr)


def _batch20(sizes=(4, 5, 6, 7, 8, 4), seed=0, drop_self_loops=True):
    """Batch con x gia' a 20 colonne (dataset transform con lost_marker=True su
    grafi INTERI: ultima colonna a zero, come farebbe la gallery/il training)."""
    rng = random.Random(seed)
    metas = [_meta(f"p{i}", n, rng) for i, n in enumerate(sizes)]
    transform = build_node_transform(normalize=False, drop_self_loops=drop_self_loops,
                                      lost_marker=True)
    graphs = [transform(build_graph(m)) for m in metas]
    return metas, Batch.from_data_list([_plain_data(g) for g in graphs])


def _edge_set(data):
    return {(int(a), int(b), tuple(e.tolist()))
            for (a, b), e in zip(data.edge_index.t(), data.edge_attr)}


# ---------------------------------------------------------------------------
# T1 -- conteggio a mano
# ---------------------------------------------------------------------------


def test_lost_neighbor_count_hand_computed():
    """5 nodi, edge_index non deduplicato, tolti {1, 3}.

    Archi (src, dst) dati: (0,1) (1,0) (0,1) (0,3) (1,2) (2,2) (3,3) (2,4) (4,3).
    Coppie (src,dst) con keep[src] e non keep[dst], deduplicate: (0,1) (0,3) (4,3).
    -> count[0] = 2 (da (0,1) e (0,3)), count[4] = 1 (da (4,3)), il resto 0
    (nodo 2 e' sorgente solo verso vicini TENUTI: (2,2) self-loop, (2,4) dst vivo).
    Verificato a mano prima di fissarlo (il valore [2,0,1,0,1] proposto nel task
    non torna: il nodo 2 non perde nessun vicino).
    """
    edge_index = torch.tensor(
        [[0, 1, 0, 0, 1, 2, 3, 2, 4],
         [1, 0, 1, 3, 2, 2, 3, 4, 3]],
        dtype=torch.long,
    )
    keep = torch.tensor([True, False, True, False, True])
    count = lost_neighbor_count(edge_index, keep)
    assert count.tolist() == [2, 0, 0, 0, 1]


# ---------------------------------------------------------------------------
# T2 -- fedelta' training <-> valutazione
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("drop_self_loops", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lost_marker_matches_between_training_and_evaluation(drop_self_loops, seed):
    meta = _meta_with_self_loop_and_reverse(seed)
    transform = build_node_transform(normalize=False, drop_self_loops=drop_self_loops,
                                      lost_marker=True)

    # --- valutazione: make_partial_graph -> transform ---
    rng_eval = random.Random(seed)
    partial, removed = make_partial_graph(meta, "random", {"fraction": 0.5}, rng_eval,
                                           lost_marker=True)
    assert partial is not None and len(removed) > 0
    eval_view = transform(partial)

    # --- training: build_graph(meta) -> stessa transform -> Batch -> keep_subgraph ---
    full_transformed = transform(build_graph(meta))
    batch = Batch.from_data_list([_plain_data(full_transformed)])
    keep = torch.ones(meta.num_rooms, dtype=torch.bool)
    keep[torch.tensor(removed, dtype=torch.long)] = False
    train_view = keep_subgraph(batch, keep, lost_marker=True)

    assert torch.equal(eval_view.x, train_view.x)                    # colonna 19 inclusa
    assert _edge_set(eval_view) == _edge_set(train_view)
    assert getattr(eval_view, "lost_marker", None) is None            # attributo rimosso
    assert getattr(train_view, "lost_marker", None) is None


# ---------------------------------------------------------------------------
# T3 -- zero sulla pianta intera
# ---------------------------------------------------------------------------


def test_lost_marker_zero_on_whole_plan_and_f0_query():
    meta = _meta_with_self_loop_and_reverse(seed=7)
    transform = build_node_transform(normalize=False, drop_self_loops=True, lost_marker=True)

    whole = transform(build_graph(meta))
    assert torch.equal(whole.x[:, LOST_MARKER_COL], torch.zeros(meta.num_rooms))

    partial, removed = make_partial_graph(meta, "random", {"fraction": 0.0}, random.Random(0),
                                           lost_marker=True)
    assert removed == []
    f0_view = transform(partial)
    assert torch.equal(f0_view.x, whole.x)                            # bit-identica


# ---------------------------------------------------------------------------
# T4 -- viste asimmetriche
# ---------------------------------------------------------------------------


def test_asym_view_a_marker_always_zero():
    _, batch = _batch20(seed=1)
    params = AugmentParams(feat_mask=0.3, geom_jitter=0.2, flip_prob=0.7, rot_prob=0.7,
                            pair_mode="asym_partial", lost_marker=True)
    a, _ = two_views(batch, params, torch.Generator().manual_seed(0))
    assert a.x.size(0) == batch.x.size(0)                             # vista A intera
    assert torch.equal(a.x[:, LOST_MARKER_COL], torch.zeros(a.x.size(0)))


def test_asym_view_b_flag_changes_only_marker_column():
    _, batch = _batch20(seed=2)
    params_on = AugmentParams(pair_mode="asym_partial", lost_marker=True)
    params_off = AugmentParams(pair_mode="asym_partial", lost_marker=False)
    _, b_on = two_views(batch, params_on, torch.Generator().manual_seed(3))
    _, b_off = two_views(batch, params_off, torch.Generator().manual_seed(3))

    assert torch.equal(b_on.edge_index, b_off.edge_index)
    assert torch.equal(b_on.batch, b_off.batch)
    assert torch.equal(b_on.x[:, :LOST_MARKER_COL], b_off.x[:, :LOST_MARKER_COL])
    assert torch.equal(b_off.x[:, LOST_MARKER_COL], torch.zeros(b_off.x.size(0)))
    assert bool((b_on.x[:, LOST_MARKER_COL] > 0).any())               # almeno un nodo di bordo


# ---------------------------------------------------------------------------
# T5 -- default bit-identico
# ---------------------------------------------------------------------------


def test_build_node_transform_default_still_19_columns_and_unchanged():
    meta = _meta("p", 6, random.Random(4))
    g = build_graph(meta)
    transform = build_node_transform(normalize=False, drop_self_loops=True)
    out = transform(_plain_data(g))
    assert out.x.size(1) == NODE_FEATURE_DIM                          # 19, niente marcatore

    manual = RemoveSpuriousSelfLoops()(_plain_data(g))
    assert torch.equal(out.x, manual.x)
    assert torch.equal(out.edge_index, manual.edge_index)


def test_keep_subgraph_default_matches_manual_subgraph():
    _, batch = _batch20(seed=5)                    # 20 colonne, ma flag spento sotto
    keep = torch.ones(batch.x.size(0), dtype=torch.bool)
    keep[0] = False
    view = keep_subgraph(batch, keep)               # lost_marker default False
    ei, ea = subgraph(keep, batch.edge_index, batch.edge_attr,
                       relabel_nodes=True, num_nodes=batch.x.size(0))
    assert torch.equal(view.x, batch.x[keep])
    assert torch.equal(view.edge_index, ei)
    assert torch.equal(view.edge_attr, ea)


def test_encoder_kwargs_in_dim_default_19_and_20_with_flag():
    import types

    from src.graph.training.train_gnn import _encoder_kwargs

    cfg = types.SimpleNamespace(hidden_dim=16, out_dim=8, num_layers=2, pooling="add",
                                 dropout=0.0, encoder="gcn")
    assert _encoder_kwargs(cfg)["in_dim"] == NODE_FEATURE_DIM
    cfg.lost_marker = True
    assert _encoder_kwargs(cfg)["in_dim"] == NODE_FEATURE_DIM + LOST_MARKER_DIM


# ---------------------------------------------------------------------------
# T6 -- guardie
# ---------------------------------------------------------------------------


def test_two_views_lost_marker_requires_asym_partial():
    _, batch = _batch20(seed=6)
    with pytest.raises(ValueError, match="lost_marker"):
        two_views(batch, AugmentParams(pair_mode="symmetric", lost_marker=True),
                  torch.Generator().manual_seed(0))


def test_append_lost_marker_rejects_already_20_columns():
    data = Data(x=torch.zeros(3, LOST_MARKER_COL + 1),
                edge_index=torch.zeros(2, 0, dtype=torch.long))
    with pytest.raises(ValueError):
        AppendLostMarker()(data)


def test_keep_subgraph_lost_marker_rejects_19_columns():
    batch = Batch.from_data_list([
        Data(x=torch.zeros(3, NODE_FEATURE_DIM), edge_index=torch.zeros(2, 0, dtype=torch.long))
    ])
    with pytest.raises(ValueError):
        keep_subgraph(batch, torch.ones(3, dtype=torch.bool), lost_marker=True)


# ---------------------------------------------------------------------------
# T7 -- contratto checkpoint
# ---------------------------------------------------------------------------


def _gat(in_dim):
    return build_graph_encoder("gat", in_dim=in_dim, hidden_dim=16, out_dim=8, num_layers=2,
                                pooling="add", dropout=0.0, heads=2, edge_dim=NUM_RELATION_TYPES,
                                attn_dropout=0.0, raw_skip=True)


def test_checkpoint_shape_mismatch_rejected_both_directions():
    enc19, enc20 = _gat(NODE_FEATURE_DIM), _gat(NODE_FEATURE_DIM + LOST_MARKER_DIM)
    with pytest.raises(RuntimeError):
        enc20.load_state_dict(enc19.state_dict())
    with pytest.raises(RuntimeError):
        enc19.load_state_dict(enc20.state_dict())


def test_gat_forward_in_dim_20_on_both_views():
    _, batch = _batch20(sizes=(4, 5, 6, 7, 8, 4), seed=8)
    params = AugmentParams(pair_mode="asym_partial", lost_marker=True)
    a, b = two_views(batch, params, torch.Generator().manual_seed(2))
    enc = _gat(NODE_FEATURE_DIM + LOST_MARKER_DIM)
    za, zb = enc(a), enc(b)
    assert za.shape == (batch.num_graphs, 8)
    assert zb.shape == (batch.num_graphs, 8)
