
"""
CPU tests of the LayoutGKN adaptation, the correctness criterion of the pilot.

(a) graph fields: shapes, category map, geometry of a known box;
(b) removed rooms = those of our graph branch (`make_partial_graph`) and those recorded on disk by the
    graph-branch evaluation (gat/comb seed 42);
(c) vectorised gallery score = upstream `ghopper_sim`, within 1e-5;
(d) at f=0 a query finds itself with similarity 1 (+-1e-5);
plus the shortest-path matrix against grakel (upstream tool) on random graphs.

Reads the RPLAN `.mat` and one query-vector file; no GPU.
Run: python -m pytest tests/test_layoutgkn.py -v
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from src.competitors.layoutgkn import UPSTREAM_CONF, upstream
from src.competitors.layoutgkn import graphs as lg
from src.competitors.layoutgkn import score as ls
from src.data.rplan_metadata import load_metadata
from src.graph.graph_partial_query import make_partial_graph

upstream()
from LayoutGKN.loss import ghopper_sim  # noqa: E402
from LayoutGKN.model import GraphSiameseNetwork  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
W_QVEC = ROOT_DIR / "results/final_pipeline/queryvec/valid/s42/graph_gat_rg_comb_s42_partial-random-f{f}_valid.npz"
PARTIAL_SEED = 42


def _w_queries(f: str):
    z = np.load(str(W_QVEC).format(f=f), allow_pickle=True)
    return z["names"], z["qi"], z["removed_ptr"], z["removed_idx"]


@pytest.fixture(scope="module")
def metas():
    names, _, _, _ = _w_queries("0.5")
    return [load_metadata(str(n)) for n in names[:40]]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = OmegaConf.load(UPSTREAM_CONF)
    m = GraphSiameseNetwork(cfg)
    m.eval()
    return m, ls.kernel_mu(cfg.hid_dim)


# --- (a) graph fields ---

def test_a_fields_and_category_map(metas):
    for meta in metas:
        g = lg.to_lgkn_graph(meta)
        n = meta.num_rooms
        assert g.num_nodes == n
        assert g.geometry.shape == (n, 6) and g.shp.shape == (n, 16)
        assert g.category.tolist() == [lg.CAT_MAP[t] for t in meta.room_types]
        assert int(g.category.max()) < lg.NUM_CATS
        assert g.edge_index.shape[1] == 2 * len(lg.undirected_pairs(meta))
        assert g.connectivity.shape == (g.edge_index.shape[1],) and int(g.connectivity.sum()) == 0
        assert torch.all(g.shp.view(n, 4, 4)[:, 0, 0] == 1)    # every room: path of length 0


def test_a_geometry_of_known_box():
    # box x 64..192, y 0..128 on the 256 grid -> after flip y 128..256
    cx, cy, w, h, a, p = lg.geometry_features([(64, 0, 192, 128)])[0]
    s = 1.8 / 256
    assert np.allclose([cx, cy, w, h], [0.0, 64 * s, 128 * s, 128 * s])
    assert np.isclose(a, 128 * s) and np.isclose(p, 128 * s)


def test_a_category_map_is_upstream():
    # rType 0..12 (RPLAN original) -> 8 classes: bedrooms merged, storage + walk-in merged
    assert [lg.CAT_MAP[t] for t in range(13)] == [0, 1, 2, 3, 4, 1, 1, 1, 1, 6, 7, 5, 5]


# --- (b) removed rooms ---

@pytest.mark.parametrize("f", ["0.25", "0.5", "0.75"])
def test_b_removed_rooms_match_our_branch_and_disk(f):
    names, qi, ptr, idx = _w_queries(f)
    for k in range(60):
        meta = load_metadata(str(names[k]))
        q = int(qi[k])
        g, removed = lg.damaged_lgkn_graph(meta, float(f), random.Random(PARTIAL_SEED + q))
        _, ours = make_partial_graph(meta, "random", {"fraction": float(f)}, random.Random(PARTIAL_SEED + q))
        assert removed == ours
        assert removed == idx[ptr[k]:ptr[k + 1]].tolist()        # rooms removed by the graph-branch evaluation
        assert g is not None and g.num_nodes == meta.num_rooms - len(removed)


# --- shortest paths vs grakel ---

def _grakel_shp(n, pairs):
    from grakel import Graph
    from grakel.kernels import GraphHopper

    def gr(n, pairs):
        edges = {i: [] for i in range(n)}
        for i, j in pairs:
            edges[i].append(j)
            edges[j].append(i)
        return Graph(edges, node_labels={i: [0.0] for i in range(n)})

    # upstream parses all graphs together (max diameter over the set); a long path in the
    # same call reproduces that and avoids grakel's squeeze bug on tiny graphs
    gh = GraphHopper(kernel_type=("gaussian", 1.0))
    gh._method_calling, gh._max_diam, gh.calculate_norm_ = 1, 5, False
    M, _ = gh.parse_input([gr(n, pairs), gr(7, [(i, i + 1) for i in range(6)])])[0]
    return M[:, :4, :4]


def test_shortest_paths_equal_grakel():
    rng = random.Random(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(150):
            n = rng.randint(2, 12)
            p = rng.choice([0.0, 0.15, 0.3, 0.6])
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
            assert np.array_equal(_grakel_shp(n, pairs), lg.shortest_path_matrix(n, pairs))


# --- (c) gallery score, (d) self-similarity ---

def _rooms(model, graphs):
    E, M, own = ls.embed_rooms(model, graphs, "cpu", batch_size=16)
    return E.double(), M.double(), own


def test_c_vectorised_score_equals_upstream(metas, model):
    net, mu = model
    gal = [lg.to_lgkn_graph(m) for m in metas[:30]]
    qry = [lg.damaged_lgkn_graph(m, 0.5, random.Random(7 + i))[0] for i, m in enumerate(metas[30:36])]
    gE, gM, gown = _rooms(net, gal)
    qE, qM, qown = _rooms(net, qry)
    sim = ls.GalleryScorer(gE, gM, gown, len(gal), mu).similarity(qE, qM, qown, len(qry), room_block=50)
    for a in range(len(qry)):
        for b in range(len(gal)):
            ref = ghopper_sim(qE[qown == a], gE[gown == b], qM[qown == a], gM[gown == b], mu=mu)
            assert abs(float(ref) - float(sim[a, b])) < 1e-5


def test_d_self_similarity_is_one(metas, model):
    net, mu = model
    gal = [lg.to_lgkn_graph(m) for m in metas]
    qry = [lg.damaged_lgkn_graph(m, 0.0, random.Random(1))[0] for m in metas[:10]]
    gE, gM, gown = _rooms(net, gal)
    qE, qM, qown = _rooms(net, qry)
    sim = ls.GalleryScorer(gE, gM, gown, len(gal), mu).similarity(qE, qM, qown, len(qry))
    for k in range(len(qry)):
        assert abs(float(sim[k, k]) - 1.0) < 1e-5


def test_expected_rr_ties():
    s = np.array([0.5, 0.9, 0.9, 0.9, 0.1])
    assert ls.expected_rr(s, 1) == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
    assert ls.expected_rr(s, 0) == pytest.approx(1 / 4)     # three plans above
    assert ls.expected_rr(np.array([0.2, 0.9]), 0, max_k=1) == 0.0
