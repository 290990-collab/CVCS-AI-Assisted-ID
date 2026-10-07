
"""
CPU smoke tests of the damaged-query vectors (`qvec/1`, late fusion).

- T1: file format roundtrip (dtypes, shapes, empty CSR, long names refused)
- T8: defaults bit-identical: vision `query()` without the flag, the two `evaluate_partial` without the
  context; with the flag same results and vectors from the same forward
- T9: qvec without per-query refused (vision config, graph CLI); the vision job dotlist expands into the
  4 `nowalls-random` runs
- T10 (real .mat, ~30 s): vision `nowalls_random` and graph `random` remove the same rooms with `Random(42 + qi)`

Run: python -m pytest tests/test_query_vectors.py -v
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.data.rplan_metadata import RoomMeta
from src.evaluation.perquery import gallery_sha1, load_perquery
from src.evaluation.query_vectors import (
    SCHEMA_VERSION,
    QueryVectorRecorder,
    array_sha1,
    load_qvec,
    write_qvec,
)
from src.evaluation.relevance import GalleryAxes
from src.vision.models.retrieval_model import VisionRetrievalPipeline

ROOT = Path(__file__).resolve().parents[1]


# --- T1: file format ---

def _meta():
    return {"branch": "graph", "run_tag": "t", "mode": "partial", "split": "valid"}


def test_qvec_roundtrip_dtypes_shapes_and_csr(tmp_path):
    path = tmp_path / "q.npz"
    removed = [[3, 1], [], [0]]
    vec = np.arange(12, dtype=np.float64).reshape(3, 4)
    write_qvec(path, meta=_meta(), names=["a", "bb", "123456"], qi=[5, 0, 2],
               vectors=vec, removed=removed,
               vectors_final=np.ones((3, 2)), whiten_mean=np.zeros((1, 4)),
               whiten_matrix=np.eye(4)[:, :2])
    d = load_qvec(path)
    assert d.meta["schema_version"] == SCHEMA_VERSION
    assert d.meta["num_queries"] == 3
    assert d.names.dtype == np.dtype("<U16")
    assert d.qi.dtype == np.int32 and d.qi.tolist() == [5, 0, 2]
    assert d.vectors.dtype == np.float32 and d.vectors.shape == (3, 4)
    np.testing.assert_array_equal(d.vectors, vec.astype(np.float32))
    assert d.removed_ptr.dtype == np.int32 and d.removed_ptr.tolist() == [0, 2, 2, 3]
    assert d.removed_idx.dtype == np.int32
    assert [d.removed(i) for i in range(3)] == removed
    assert d.vectors_final.shape == (3, 2) and d.whiten_matrix.shape == (4, 2)


def test_qvec_all_empty_removed_and_no_optional_arrays(tmp_path):
    path = tmp_path / "q.npz"
    write_qvec(path, meta=_meta(), names=["a", "b"], qi=[0, 1],
               vectors=np.zeros((2, 3)), removed=[[], []])
    d = load_qvec(path)
    assert d.removed_ptr.tolist() == [0, 0, 0]
    assert d.removed_idx.shape == (0,)
    assert d.vectors_final is None and d.whiten_mean is None and d.whiten_matrix is None


def test_qvec_refuses_long_names_and_bad_shapes(tmp_path):
    with pytest.raises(ValueError, match="too long"):
        write_qvec(tmp_path / "q.npz", meta=_meta(), names=["x" * 17], qi=[0],
                   vectors=np.zeros((1, 2)), removed=[[]])
    with pytest.raises(ValueError, match="vectors"):
        write_qvec(tmp_path / "q.npz", meta=_meta(), names=["a"], qi=[0],
                   vectors=np.zeros((2, 2)), removed=[[]])
    with pytest.raises(ValueError, match="together"):
        write_qvec(tmp_path / "q.npz", meta=_meta(), names=["a"], qi=[0],
                   vectors=np.zeros((1, 2)), removed=[[]], whiten_mean=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="removed"):
        write_qvec(tmp_path / "q.npz", meta=_meta(), names=["a"], qi=[0],
                   vectors=np.zeros((1, 2)), removed=[[], []])


def test_array_sha1_depends_on_bytes_shape_and_dtype():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert array_sha1(a) == array_sha1(a.copy())
    assert array_sha1(a) != array_sha1(a.reshape(3, 2))
    assert array_sha1(a) != array_sha1(a.astype(np.float64))
    b = a.copy()
    b[1, 2] += 1e-6
    assert array_sha1(a) != array_sha1(b)


def test_recorder_refuses_mixed_final_vectors():
    rec = QueryVectorRecorder()
    rec.add(name="a", qi=0, vector=np.zeros(2), removed=[], vector_final=np.zeros(2))
    with pytest.raises(ValueError, match="vector_final"):
        rec.add(name="b", qi=1, vector=np.zeros(2), removed=[])


# --- T8: vision query(): default unchanged, flag = same forward ---

class _FlatEncoder(torch.nn.Module):
    """[B, 3, 4, 4] -> [B, 8], deterministic."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(48, 8, bias=False)

    def forward(self, x):
        return self.lin(x.flatten(1))


class _EncoderStub(VisionRetrievalPipeline):
    """Pipeline with a tiny encoder and no transform/weights to download."""

    def __init__(self, raw_embeddings, image_paths):
        self.encoder = _FlatEncoder().eval()
        self.device = "cpu"
        self.transform = None
        self.raw_embeddings = raw_embeddings
        self.image_paths = image_paths
        self.head = None
        self.embeddings = None
        self.index = None
        self.whiten_mean = None
        self.whiten_matrix = None


def _vision_stub(n=40):
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(n, 8)).astype(np.float32)
    paths = [f"/fake/p{i}.png" for i in range(n)]
    pipe = _EncoderStub(raw, paths)
    pipe.prepare_index(whiten=True, fit_rows=list(range(30)))
    return pipe


def _historical_query(pipe, image, top_k):
    """Body of `query()` before the damaged-query flag, same logic."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    with torch.no_grad():
        q = pipe.encoder(image.to(pipe.device)).cpu().numpy().astype("float32")
    q = pipe._apply_head(q)
    q = pipe._apply_whitening(q)
    scores, indices = pipe.index.search(q, top_k)
    return [{"path": pipe.image_paths[i], "score": float(s)} for s, i in zip(scores[0], indices[0])]


def test_vision_query_default_is_unchanged_and_flag_returns_the_same_forward():
    pipe = _vision_stub()
    torch.manual_seed(1)
    image = torch.randn(3, 4, 4)

    default = pipe.query(image, top_k=7)
    assert isinstance(default, list)
    assert default == _historical_query(pipe, image, 7)

    results, raw, final = pipe.query(image, top_k=7, return_embeddings=True)
    assert results == default
    with torch.no_grad():
        expected_raw = pipe.encoder(image.unsqueeze(0)).numpy().astype("float32")
    np.testing.assert_array_equal(raw, expected_raw)
    np.testing.assert_array_equal(final, pipe._apply_whitening(pipe._apply_head(raw)))


# --- T8: vision evaluate_partial: context off = identical files, on = vectors ---

def _plan_metas(n):
    return [RoomMeta(name=f"p{i}", split="valid", room_types=(0, 1, 2, 3)[: 3 + i % 2],
                     edges=((0, 1, 1), (1, 2, 1)),
                     boxes=tuple((10 * j, 10 * i, 10 * j + 8, 10 * i + 8) for j in range(4))[: 3 + i % 2],
                     footprint=(0, 0, 40 + i, 40), entrance=None)
            for i in range(n)]


def _fake_damaged_query(png_path, meta, strategy, params, rng, open_boundary, transform,
                        patch_ctx=None):
    """Deterministic stand-in: the tensor and the removed rooms come from `rng`."""
    tensor = torch.tensor([rng.random() for _ in range(48)], dtype=torch.float32).view(3, 4, 4)
    info = {"removed_any": True, "area_removed": 0.1, "fallback": False}
    if strategy != "crop":
        info["removed"] = sorted(rng.sample(range(len(meta.room_types)), 1))
    return tensor, info


def _run_vision_partial(tmp_path, monkeypatch, sub, query_vectors):
    from src.vision.evaluation import evaluate as ev
    n = 40
    metas = _plan_metas(n)
    monkeypatch.setattr(ev, "load_metadata", lambda p, *a, **k: metas[int(Path(p).stem[1:])])
    monkeypatch.setattr(ev, "damaged_query", _fake_damaged_query)
    pipe = _vision_stub(n)
    names = [Path(p).stem for p in pipe.image_paths]
    stem2row = {s: i for i, s in enumerate(names)}
    pcfg = OmegaConf.create({"seed": 42, "open_boundary": True, "strategies": {
        "nowalls_random": {"enabled": True, "fractions": [0.0, 0.5]},
        "crop": {"enabled": True, "fractions": [0.25]},
    }})
    out = tmp_path / sub
    perquery = {"dir": out / "pq", "tag": "stub_gem_whiten-train", "split": "valid", "seed": 42,
                "gallery": {"n": n, "sha1": gallery_sha1(names), "source": "synthetic"}}
    qctx = None
    if query_vectors:
        qctx = {"dir": out / "qv", "head": None, "gallery_vectors": {"path": "x", "sha1": "y",
                "shape": [n, 8]}, "whitening": {"enabled": True, "fit_split": "train",
                                                 "eps": 1e-6, "dim": None}}
    ev.evaluate_partial(pipe, GalleryAxes(metas), stem2row, [3, 11, 25, 7], pipe.image_paths,
                        (1, 5), pcfg, perquery=perquery, query_vectors=qctx)
    return pipe, out


def test_vision_evaluate_partial_is_unchanged_and_writes_query_vectors(tmp_path, monkeypatch):
    _, off = _run_vision_partial(tmp_path, monkeypatch, "off", query_vectors=False)
    pipe, on = _run_vision_partial(tmp_path, monkeypatch, "on", query_vectors=True)

    files = sorted(p.name for p in (off / "pq").glob("*.npz"))
    assert files == sorted(p.name for p in (on / "pq").glob("*.npz"))
    assert len(files) == 3
    for name in files:
        a, b = load_perquery(off / "pq" / name), load_perquery(on / "pq" / name)
        for key in ("names", "qi", "ndcg", "recall", "map", "num_relevant", "ret_rows",
                    "n_ret", "self_rr", "area_removed"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key), err_msg=key)
    assert not (off / "qv").exists()

    # crop has no `removed`: skipped; the two nowalls runs are written.
    written = sorted(p.name for p in (on / "qv").glob("*.npz"))
    assert written == ["vision_stub_gem_whiten-train_partial-nowalls-random-f0.0_valid.npz",
                       "vision_stub_gem_whiten-train_partial-nowalls-random-f0.5_valid.npz"]
    d = load_qvec(on / "qv" / written[1])
    pq = load_perquery(on / "pq" / written[1])
    np.testing.assert_array_equal(d.names, pq.names)
    np.testing.assert_array_equal(d.qi, pq.qi)
    assert d.meta["partial_label"] == "nowalls-random f=0.5"
    assert d.meta["partial_seed"] == 42 and d.meta["k_values"] == [1, 5]
    assert d.meta["head"] is None and d.meta["perquery_file"].endswith(written[1])
    np.testing.assert_array_equal(d.whiten_mean, pipe.whiten_mean)
    np.testing.assert_array_equal(d.whiten_matrix, pipe.whiten_matrix)
    for i, qi in enumerate(d.qi):
        rng = random.Random(42 + int(qi))
        tensor, info = _fake_damaged_query(None, _plan_metas(40)[int(qi)], "nowalls_random",
                                           {}, rng, True, None)
        with torch.no_grad():
            raw = pipe.encoder(tensor.unsqueeze(0)).numpy()
        np.testing.assert_array_equal(d.vectors[i], raw[0])
        np.testing.assert_array_equal(d.vectors_final[i], pipe._apply_whitening(raw)[0])
        assert d.removed(i) == info["removed"]


# --- T8: graph evaluate_partial ---

def _graph_setup():
    faiss = pytest.importorskip("faiss")
    from src.graph.graph_builder import build_graph
    from src.graph.models import build_graph_encoder
    metas = _plan_metas(6)
    torch.manual_seed(0)
    encoder = build_graph_encoder("gcn", pooling="add", raw_skip=False,
                                  in_dim=19, hidden_dim=8, out_dim=4, num_layers=2).eval()
    with torch.no_grad():
        gallery = np.ascontiguousarray(
            np.vstack([encoder(build_graph(m)).numpy() for m in metas]), dtype=np.float32)
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery)
    return metas, encoder, gallery, index


def _graph_args(**extra):
    return SimpleNamespace(partial_seed=42, partial_strategies=["random"],
                           partial_fractions=[0.0, 0.5], partial_keep_types=[0, 2, 3],
                           partial_max_degree=1, batch_size=4, baseline_hist=False, **extra)


def test_graph_evaluate_partial_is_unchanged_and_writes_query_vectors(tmp_path, monkeypatch):
    from src.graph.evaluation import graph_evaluate as ge
    from src.graph.graph_partial_query import make_partial_graph
    metas, encoder, gallery, index = _graph_setup()
    by_name = {m.name: m for m in metas}
    monkeypatch.setattr(ge, "load_metadata", lambda name, *a, **k: by_name[Path(str(name)).stem])
    names = [m.name for m in metas]
    rows = list(range(len(metas)))

    def ctx(d):
        return {"dir": d, "tag": "gcn_test", "split": "valid", "seed": 42,
                "gallery": {"n": len(names), "sha1": gallery_sha1(names), "source": "synthetic"}}

    # namespace without the new field: no context, nothing new written
    args = _graph_args()
    assert ge.query_vectors_context(args, gallery) is None
    ge.evaluate_partial(index, GalleryAxes(metas), rows, (1, 3), names, encoder, None, args,
                        "cpu", perquery=ctx(tmp_path / "off"))

    args_on = _graph_args(query_vectors_out=str(tmp_path / "qv"), save_dir=str(tmp_path / "ckpt"),
                          encoder="gcn", variant="test", raw_skip=False, lost_marker=False,
                          normalize=False, drop_self_loops=True)
    qctx = ge.query_vectors_context(args_on, gallery)
    assert qctx["gallery_vectors"]["sha1"] == array_sha1(gallery)
    ge.evaluate_partial(index, GalleryAxes(metas), rows, (1, 3), names, encoder, None, args_on,
                        "cpu", perquery=ctx(tmp_path / "on"), query_vectors=qctx)

    off = sorted(p.name for p in (tmp_path / "off").glob("*.npz"))
    assert off == sorted(p.name for p in (tmp_path / "on").glob("*.npz")) and len(off) == 2
    for name in off:
        a, b = load_perquery(tmp_path / "off" / name), load_perquery(tmp_path / "on" / name)
        for key in ("names", "qi", "ndcg", "recall", "map", "num_relevant", "ret_rows",
                    "n_ret", "self_rr"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key), err_msg=key)

    d = load_qvec(tmp_path / "qv" / off[1])       # f=0.5
    assert d.meta["branch"] == "graph" and d.meta["partial_label"] == "random f=0.5"
    assert d.meta["model"]["encoder"] == "gcn" and "n_degenerate" in d.meta
    from src.graph.graph_builder import build_graph
    for i, qi in enumerate(d.qi):
        graph, removed = make_partial_graph(metas[int(qi)], "random", {"fraction": 0.5},
                                            random.Random(42 + int(qi)))
        assert d.removed(i) == list(removed)
        with torch.no_grad():
            np.testing.assert_allclose(d.vectors[i], encoder(graph).numpy()[0], atol=1e-6)


# --- T9: qvec requires per-query; vision job dotlist ---

def test_vision_query_vectors_context_requires_perquery():
    from src.vision.evaluation.evaluate import query_vectors_context
    cfg = OmegaConf.create({"eval": {"query_vectors_dir": "x", "perquery_dir": None},
                            "whitening": {"enabled": True, "eps": 1e-6},
                            "retrieval": {"save_dir": "s"}})
    with pytest.raises(ValueError, match="perquery_dir"):
        query_vectors_context(cfg, None)
    cfg.eval.query_vectors_dir = None
    assert query_vectors_context(cfg, None) is None


def test_graph_cli_refuses_query_vectors_without_perquery(monkeypatch):
    from src.graph.evaluation import graph_evaluate as ge
    monkeypatch.setattr(sys, "argv", ["graph_evaluate", "--query-vectors-out", "x", "--partial"])
    with pytest.raises(SystemExit):
        ge.parse_args()


def test_vision_job_dotlist_expands_into_four_nowalls_random_runs():
    """The overrides of scripts/evaluation/08_queryvec_vision.sh (kept in sync by hand)."""
    from src.vision.evaluation.evaluate import partial_runs
    from src.vision.utils.config import load_vision_config
    overrides = [
        "model.name=pespatial", "model.variant=gem", "model.kwargs.pooling=gem",
        "head.enabled=false", "whitening.enabled=true", "partial.enabled=true",
        "partial.strategies.random.enabled=false", "partial.strategies.semantic.enabled=false",
        "partial.strategies.topology.enabled=false", "partial.strategies.crop.enabled=false",
        "partial.strategies.patch.enabled=false", "partial.strategies.nowalls_random.enabled=true",
        "partial.strategies.nowalls_random.fractions=[0.0,0.25,0.5,0.75]",
        "partial.strategies.nowalls_semantic.enabled=false",
        "partial.strategies.nowalls_topology.enabled=false",
        "eval.split=valid", "eval.perquery_dir=results/perquery/fusion_branches_valid",
        "eval.query_vectors_dir=results/queryvec/valid",
    ]
    cfg = load_vision_config(str(ROOT / "configs/vision_retrieval.yaml"), overrides)
    runs = partial_runs(cfg.partial)
    assert [r[0] for r in runs] == ["nowalls-random f=0.0", "nowalls-random f=0.25",
                                    "nowalls-random f=0.5", "nowalls-random f=0.75"]
    assert all(r[1] == "nowalls_random" for r in runs)
    assert cfg.eval.query_vectors_dir == "results/queryvec/valid"


# --- T10: real .mat, the two branches remove the same rooms ---

def test_real_plans_nowalls_random_and_graph_random_remove_the_same_rooms():
    from src.data.rplan_metadata import DEFAULT_MAT_DIR, load_metadata
    from src.graph.graph_partial_query import make_partial_graph
    from src.vision.data.vision_damage import damaged_query
    snapshot = Path(OmegaConf.load(ROOT / "configs/vision_retrieval.yaml").retrieval.data_dir)
    if not (DEFAULT_MAT_DIR / "data_valid.mat").exists() or not snapshot.exists():
        pytest.skip("RPLAN .mat / snapshot not reachable")
    shared = __import__("json").loads((ROOT / "results/shared_gallery.json").read_text())["names"]
    picked = []
    for qi, name in enumerate(shared):
        meta = load_metadata(name)
        png = snapshot / f"{name}.png"
        if meta is not None and meta.split == "valid" and png.exists():
            picked.append((qi, name, meta, png))
        if len(picked) == 20:
            break
    assert len(picked) == 20
    for f in (0.25, 0.5, 0.75):
        for qi, name, meta, png in picked:
            _, info = damaged_query(png, meta, "nowalls_random", {"fraction": f},
                                    random.Random(42 + qi), True, lambda img: img)
            _, removed = make_partial_graph(meta, "random", {"fraction": f}, random.Random(42 + qi))
            assert info["removed"] == list(removed), (name, f)
