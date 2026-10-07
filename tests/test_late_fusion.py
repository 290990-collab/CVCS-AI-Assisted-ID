
"""
CPU smoke tests of the late fusion (`src/evaluation/late_fusion.py`) and of the alpha choice
(`src/evaluation/fusion_select.py`) on a synthetic world (no dataset, no weights).

- T2: alpha=1 == vision only, alpha=0 == graph only; fused vectors have unit norm
- T3: join by name (shuffle-invariant, missing names excluded and counted, inconsistent qi -> ValueError)
- T4: refusals (gallery sha1, split/seed/partial_seed, vector sha1, graph names, vision head, graph norm,
  whitening across fractions)
- T5: different removed rooms -> ValueError with name and fraction
- T6: fused files readable by `load_perquery`, `load_auc(nowalls-random)`, `check_compatible`
- T7: pure rules of `select` and threshold of C3

Run: python -m pytest tests/test_late_fusion.py -v
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from src.data.rplan_metadata import RoomMeta
from src.evaluation import late_fusion as lf
from src.evaluation import fusion_select as fs
from src.evaluation.gallery_join import write_shared_gallery
from src.evaluation.perquery import gallery_sha1, load_perquery
from src.evaluation.query_vectors import array_sha1, load_qvec, write_qvec
from src.evaluation.robustness_auc import fraction_path, load_auc
from src.evaluation.significance import check_compatible
from src.vision.evaluation.evaluate import partial_rows
from src.vision.models.retrieval_model import VisionRetrievalPipeline

N, DV, DG, Q = 500, 16, 8, 40
K_VALUES = [1, 5, 10]
FRACTIONS = [0.0, 0.25, 0.5, 0.75]
ALPHAS = [0.0, 0.5, 1.0]
SPLIT = "valid"


class _Stub(VisionRetrievalPipeline):
    def __init__(self):
        self.device = "cpu"
        self.head = None
        self.whiten_mean = None
        self.whiten_matrix = None


def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _metas(names, rng):
    out = []
    for i, name in enumerate(names):
        k = 3 + int(rng.integers(0, 3))
        types = tuple(int(t) for t in rng.integers(0, 6, size=k))
        edges = tuple((j, j + 1, int(rng.integers(0, 4))) for j in range(k - 1))
        boxes = tuple((10 * j, 0, 10 * j + 8 + int(rng.integers(0, 5)), 20) for j in range(k))
        out.append(RoomMeta(name=name, split="valid", room_types=types, edges=edges, boxes=boxes,
                            footprint=(0, 0, 60 + int(rng.integers(0, 100)), 80), entrance=None))
    return out


class World:
    """All files of a fusion run, written in `root`, with the ground truth in memory."""

    def __init__(self, root: Path, monkeypatch):
        rng = np.random.default_rng(7)
        self.root = root
        self.shared = [f"{i:05d}" for i in range(N)]
        self.sha1 = gallery_sha1(self.shared)
        write_shared_gallery(root / "shared.json", self.shared, sources={})
        metas = _metas(self.shared, rng)
        by_name = {m.name: m for m in metas}
        monkeypatch.setattr(lf, "load_metadata", lambda n, *a, **k: by_name[str(n)])

        # vision gallery: raw on disk, 3 extra rows, shuffled order
        self.raw = rng.normal(size=(N, DV)).astype(np.float32)
        stub = _Stub()
        stub._fit_whitening(self.raw[:300])
        self.wmean, self.wmatrix = stub.whiten_mean, stub.whiten_matrix
        order = rng.permutation(N + 3)
        extra = rng.normal(size=(3, DV)).astype(np.float32)
        disk = np.vstack([self.raw, extra])[order]
        paths = [f"/snap/{n}.png" for n in self.shared + ["x1", "x2", "x3"]]
        vdir = root / "emb_vision"
        vdir.mkdir()
        np.save(vdir / "embeddings.npy", disk)
        (vdir / "image_paths.json").write_text(json.dumps([paths[i] for i in order]))
        self.vision_emb = vdir / "embeddings.npy"

        # graph gallery: unit rows, shared order
        self.gg = _unit(rng.normal(size=(N, DG)))
        gdir = root / "emb_graph"
        gdir.mkdir()
        np.save(gdir / "embeddings.npy", self.gg)
        (gdir / "names.json").write_text(json.dumps(self.shared))
        self.graph_emb = gdir / "embeddings.npy"

        # damaged queries, one qvec per branch and fraction
        self.qi = rng.choice(N, size=Q, replace=False)
        self.vprefix = root / "qv" / "vision_stub_gem_whiten-train"
        self.gprefix = root / "qv" / "graph_gat_asymrob"
        self.vq, self.gq, self.removed = {}, {}, {}
        for f in FRACTIONS:
            self.vq[f] = self.raw[self.qi] + (0.2 + f) * rng.normal(size=(Q, DV)).astype(np.float32)
            self.gq[f] = _unit(self.gg[self.qi] + (0.2 + f) * rng.normal(size=(Q, DG)))
            self.removed[f] = [sorted(random.Random(42 + int(q)).sample(range(3), int(3 * f)))
                               for q in self.qi]
            self.write_vision(f)
            self.write_graph(f)

    def base_meta(self, branch, f, label):
        return {"branch": branch, "run_tag": "tag", "mode": "partial", "partial_label": label,
                "damage": {"strategy": "x", "params": {"fraction": f}}, "split": SPLIT,
                "query_seed": 42, "partial_seed": 42, "k_values": K_VALUES,
                "gallery": {"n": N, "sha1": self.sha1, "source": "synthetic"},
                # per-query of the "same job" (see test_select_and_check_run_end_to_end)
                "perquery_file": str(fraction_path(
                    self.root / "pq" / f"{branch}_x", f, SPLIT,
                    "nowalls-random" if branch == "vision" else "random"))}

    def write_vision(self, f, meta_update=None, rows=None, **arrays):
        rows = np.arange(Q) if rows is None else rows
        meta = self.base_meta("vision", f, f"nowalls-random f={f}")
        meta.update({"gallery_vectors": {"path": str(self.vision_emb), "sha1": array_sha1(self.raw),
                                         "shape": [N, DV]},
                     "whitening": {"enabled": True, "fit_split": "train", "eps": 1e-6, "dim": None},
                     "head": None})
        meta.update(meta_update or {})
        kw = dict(names=[self.shared[q] for q in self.qi[rows]], qi=self.qi[rows],
                  vectors=self.vq[f][rows], removed=[self.removed[f][i] for i in rows],
                  whiten_mean=self.wmean, whiten_matrix=self.wmatrix)
        kw.update(arrays)
        write_qvec(fraction_path(self.vprefix, f, SPLIT, "nowalls-random"), meta=meta, **kw)

    def write_graph(self, f, meta_update=None, rows=None, **arrays):
        rows = np.arange(Q) if rows is None else rows
        meta = self.base_meta("graph", f, f"random f={f}")
        meta.update({"gallery_vectors": {"path": str(self.graph_emb), "sha1": array_sha1(self.gg),
                                         "shape": [N, DG]},
                     "model": {"encoder": "gat"}, "n_degenerate": 0})
        meta.update(meta_update or {})
        kw = dict(names=[self.shared[q] for q in self.qi[rows]], qi=self.qi[rows],
                  vectors=self.gq[f][rows], removed=[self.removed[f][i] for i in rows])
        kw.update(arrays)
        write_qvec(fraction_path(self.gprefix, f, SPLIT, "random"), meta=meta, **kw)

    def argv(self, out="fused", alphas=ALPHAS, modes=("partial", "full")):
        return ["run", "--split", SPLIT, "--gallery-names", str(self.root / "shared.json"),
                "--vision-qvec", str(self.vprefix), "--graph-qvec", str(self.gprefix),
                "--vision-embeddings", str(self.vision_emb), "--graph-embeddings", str(self.graph_emb),
                "--fractions", *[str(f) for f in FRACTIONS], "--alphas", *[str(a) for a in alphas],
                "--modes", *modes, "--out", str(self.root / out)]


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


# --- T2: alpha=1 / alpha=0 reproduce one branch ---

def _vision_reference(w, f):
    """Vision job loop on the whitened gallery: FAISS + `evaluate.partial_rows`."""
    stub = _Stub()
    stub.whiten_mean, stub.whiten_matrix = w.wmean, w.wmatrix
    vg = stub._apply_whitening(w.raw)
    index = faiss.IndexFlatIP(DV)
    index.add(vg)
    stem2row = {n: i for i, n in enumerate(w.shared)}
    max_k = max(K_VALUES)
    out = []
    for j, qi in enumerate(w.qi):
        scores, idx = index.search(stub._apply_whitening(w.vq[f][j:j + 1]), max_k + 1)
        results = [{"path": f"/snap/{w.shared[i]}.png", "score": float(s)}
                   for s, i in zip(scores[0], idx[0])]
        self_rows, axis_rows = partial_rows(results, int(qi), stem2row, max_k)
        hit = np.where(np.asarray(self_rows) == qi)[0]
        out.append((axis_rows, 1.0 / (int(hit[0]) + 1) if len(hit) else 0.0))
    return out


def _graph_reference(w, f):
    index = faiss.IndexFlatIP(DG)
    index.add(w.gg)
    max_k = max(K_VALUES)
    _, ranked = index.search(w.gq[f], max_k + 1)
    out = []
    for j, qi in enumerate(w.qi):
        row_list = [int(r) for r in ranked[j] if r != -1]
        axis_rows = [r for r in row_list if r != qi][:max_k]
        hit = [i for i, r in enumerate(row_list[:max_k], start=1) if r == qi]
        out.append((axis_rows, 1.0 / hit[0] if hit else 0.0))
    return out


def _assert_matches(data, reference):
    for j, (axis_rows, rr) in enumerate(reference):
        n = int(data.n_ret[j])
        assert data.ret_rows[j, :n].tolist() == axis_rows
        assert data.self_rr[j] == np.float32(rr)       # stored as float32


def test_alpha_one_is_vision_and_alpha_zero_is_graph(world):
    lf.main(world.argv())
    for f in FRACTIONS:
        _assert_matches(load_perquery(lf.fused_partial_path(world.root / "fused", 1.0, f, SPLIT)),
                        _vision_reference(world, f))
        _assert_matches(load_perquery(lf.fused_partial_path(world.root / "fused", 0.0, f, SPLIT)),
                        _graph_reference(world, f))
    # full plan, alpha=0: query = gallery row, self dropped
    full = load_perquery(lf.fused_full_path(world.root / "fused", 0.0, SPLIT))
    index = faiss.IndexFlatIP(DG)
    index.add(world.gg)
    _, ranked = index.search(world.gg[world.qi], max(K_VALUES) + 1)
    for j, qi in enumerate(world.qi):
        expected = [int(r) for r in ranked[j] if r != qi and r != -1][:max(K_VALUES)]
        assert full.ret_rows[j, :full.n_ret[j]].tolist() == expected
    assert full.self_rr is None and full.meta["mode"] == "full"


def test_fused_vectors_have_unit_norm_and_exact_endpoints():
    rng = np.random.default_rng(0)
    v, g = _unit(rng.normal(size=(20, DV))), _unit(rng.normal(size=(20, DG)))
    for a in (0.0, 0.1, 0.35, 0.5, 0.9, 1.0):
        np.testing.assert_allclose(np.linalg.norm(lf.fuse(v, g, a), axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(lf.fuse(v, g, 1.0)[:, :DV], v)
    assert not lf.fuse(v, g, 1.0)[:, DV:].any()
    np.testing.assert_array_equal(lf.fuse(v, g, 0.0)[:, DV:], g)
    with pytest.raises(ValueError):
        lf.fuse(v, g, 1.2)


# --- T3: join by name ---

_ARRAYS = ("names", "qi", "ndcg", "recall", "map", "num_relevant", "ret_rows", "n_ret", "self_rr")


def test_shuffled_graph_rows_give_identical_output(world):
    lf.main(world.argv(out="a", alphas=[0.5], modes=("partial",)))
    perm = np.random.default_rng(3).permutation(Q)
    for f in FRACTIONS:
        world.write_graph(f, rows=perm)
    lf.main(world.argv(out="b", alphas=[0.5], modes=("partial",)))
    for f in FRACTIONS:
        a = load_perquery(lf.fused_partial_path(world.root / "a", 0.5, f, SPLIT))
        b = load_perquery(lf.fused_partial_path(world.root / "b", 0.5, f, SPLIT))
        for key in _ARRAYS:
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key), err_msg=key)


def test_missing_graph_query_is_excluded_and_counted(world):
    world.write_graph(0.5, rows=np.arange(1, Q))
    lf.main(world.argv(alphas=[0.5], modes=("partial",)))
    d = load_perquery(lf.fused_partial_path(world.root / "fused", 0.5, 0.5, SPLIT))
    assert len(d.names) == Q - 1 and world.shared[world.qi[0]] not in set(d.names)
    assert d.meta["join"] == {"n_only_vision": 1, "n_only_graph": 0}
    assert len(load_perquery(lf.fused_partial_path(world.root / "fused", 0.5, 0.25, SPLIT)).names) == Q


def test_inconsistent_qi_is_refused(world):
    bad = world.qi.copy()
    bad[2] = (bad[2] + 1) % N
    world.write_graph(0.25, qi=bad)
    with pytest.raises(ValueError, match="shared\\[qi\\]"):
        lf.main(world.argv())
    d_v = load_qvec(fraction_path(world.vprefix, 0.25, SPLIT, "nowalls-random"))
    d_g = load_qvec(fraction_path(world.gprefix, 0.25, SPLIT, "random"))
    with pytest.raises(ValueError, match="qi"):
        lf.join_queries(d_v, d_g, 0.25)


# --- T4: refusals ---

def test_refuses_gallery_sha1_of_another_gallery(world):
    world.write_vision(0.5, meta_update={"gallery": {"n": N, "sha1": "0" * 40, "source": "x"}})
    with pytest.raises(ValueError, match="shared gallery"):
        lf.main(world.argv())


@pytest.mark.parametrize("key,value", [("split", "test"), ("query_seed", 7), ("partial_seed", 7)])
def test_refuses_protocol_mismatch_between_branches(world, key, value):
    for f in FRACTIONS:
        world.write_graph(f, meta_update={key: value})
    with pytest.raises(ValueError, match=key):
        lf.main(world.argv())


def test_refuses_rewritten_gallery_vectors(world):
    emb = np.load(world.graph_emb)
    emb[0] = _unit(emb[:1] + 0.1)[0]
    np.save(world.graph_emb, emb)
    with pytest.raises(ValueError, match="sha1"):
        lf.main(world.argv())


def test_refuses_vision_gallery_rewritten(world):
    emb = np.load(world.vision_emb)
    emb[5] += 1.0
    np.save(world.vision_emb, emb)
    with pytest.raises(ValueError, match="vision gallery vectors"):
        lf.main(world.argv())


def test_refuses_graph_names_not_equal_to_shared(world):
    names = json.loads((world.graph_emb.parent / "names.json").read_text())
    names[0], names[1] = names[1], names[0]
    (world.graph_emb.parent / "names.json").write_text(json.dumps(names))
    with pytest.raises(ValueError, match="names differ"):
        lf.main(world.argv())


def test_refuses_vision_head(world):
    for f in FRACTIONS:
        world.write_vision(f, meta_update={"head": {"enabled": True, "file": "head.pt"}})
    with pytest.raises(ValueError, match="head"):
        lf.main(world.argv())


def test_refuses_graph_vectors_without_unit_norm(world):
    scaled = (world.gg * 1.01).astype(np.float32)
    np.save(world.graph_emb, scaled)
    for f in FRACTIONS:
        world.write_graph(f, meta_update={"gallery_vectors": {
            "path": str(world.graph_emb), "sha1": array_sha1(scaled), "shape": [N, DG]}})
    with pytest.raises(ValueError, match="norm"):
        lf.main(world.argv())


def test_refuses_graph_queries_without_unit_norm(world):
    world.write_graph(0.75, vectors=world.gq[0.75] * 2.0)
    with pytest.raises(ValueError, match="graph queries f=0.75"):
        lf.main(world.argv())


def test_refuses_whitening_that_differs_across_fractions(world):
    world.write_vision(0.5, whiten_mean=world.wmean + 1e-3)
    with pytest.raises(ValueError, match="whitening"):
        lf.main(world.argv())


# --- T5: different removed rooms ---

def test_different_removed_rooms_name_the_query_and_the_fraction(world):
    removed = [list(r) for r in world.removed[0.5]]
    removed[4] = [2] if removed[4] != [2] else [0]
    world.write_graph(0.5, removed=removed)
    with pytest.raises(ValueError) as exc:
        lf.main(world.argv())
    assert world.shared[world.qi[4]] in str(exc.value) and "f=0.5" in str(exc.value)


# --- T6: fused files follow the per-query contract ---

def test_fused_files_are_readable_by_the_existing_tools(world):
    lf.main(world.argv())
    out = world.root / "fused"
    a0 = load_perquery(lf.fused_partial_path(out, 0.0, 0.25, SPLIT))
    a1 = load_perquery(lf.fused_partial_path(out, 1.0, 0.25, SPLIT))
    assert a0.meta["branch"] == "fusion" and a0.meta["exclude_self"] is True
    assert a0.meta["damage"]["graph_strategy"] == "random"
    assert a0.meta["fusion"]["method"] == "weighted_concat_sqrt"
    assert a0.meta["gallery"]["sha1"] == world.sha1
    check_compatible(a0.meta, a1.meta, k=10, allow_gallery_mismatch=False)
    check_compatible(load_perquery(lf.fused_full_path(out, 0.0, SPLIT)).meta,
                     load_perquery(lf.fused_full_path(out, 0.5, SPLIT)).meta,
                     k=10, allow_gallery_mismatch=False)
    auc = load_auc(lf.fusion_prefix(out, 0.5), SPLIT, strategy="nowalls-random")
    assert len(auc.names) == Q and np.isfinite(auc.mean)
    assert lf.fused_partial_path(out, 0.5, 0.25, SPLIT).name == \
        "fusion_a0.5_partial-nowalls-random-f0.25_valid.npz"
    assert lf.fused_full_path(out, 1.0, SPLIT).name == "fusion_a1_full_valid.npz"


def test_run_twice_on_the_same_inputs_is_bit_identical(world):
    """Same inputs give byte-identical fused files (meta compared without `timestamp`/`argv`)."""
    lf.main(world.argv(out="rep1"))
    lf.main(world.argv(out="rep2"))
    paths = (
        [lf.fused_partial_path(world.root / "rep1", a, f, SPLIT) for a in ALPHAS for f in FRACTIONS]
        + [lf.fused_full_path(world.root / "rep1", a, SPLIT) for a in ALPHAS]
    )
    for p1 in paths:
        p2 = Path(str(p1).replace("rep1", "rep2"))
        d1, d2 = load_perquery(p1), load_perquery(p2)
        for key in _ARRAYS:
            v1, v2 = getattr(d1, key), getattr(d2, key)
            if v1 is None or v2 is None:
                assert v1 is None and v2 is None, key
            else:
                np.testing.assert_array_equal(v1, v2, err_msg=f"{p1.name}:{key}")
        m1 = {k: v for k, v in d1.meta.items() if k not in ("timestamp", "argv")}
        m2 = {k: v for k, v in d2.meta.items() if k not in ("timestamp", "argv")}
        assert m1 == m2, p1.name


def test_select_and_check_run_end_to_end(world, tmp_path):
    """The CLI of fusion_select on the synthetic world: runs and writes its json."""
    from src.evaluation.perquery import write_npz
    lf.main(world.argv(alphas=[i / 10 for i in range(11)]))
    out = world.root / "fused"
    fs.main(["select", "--split", SPLIT, "--fusion-dir", str(out),
                   "--out-json", str(tmp_path / "sel.json")])
    sel = json.loads((tmp_path / "sel.json").read_text())
    assert sel["alpha_star"] in [i / 10 for i in range(11)]
    assert sel["reference_alpha"] in (0.0, 1.0) and "verdict" in sel and "full" in sel
    fs.main(["select", "--split", SPLIT, "--fusion-dir", str(out),
             "--fixed-alpha-from", str(tmp_path / "sel.json"), "--out-json", str(tmp_path / "t.json")])
    assert json.loads((tmp_path / "t.json").read_text())["alpha_star_rule"] == "fixed"
    args = lf.parse_args(["run", "--split", SPLIT, "--vision-qvec", "v", "--graph-qvec", "g",
                          "--out", "o"])
    assert lf._alphas(args) == [i / 10 for i in range(11)]
    args = lf.parse_args(["run", "--split", SPLIT, "--vision-qvec", "v", "--graph-qvec", "g",
                          "--alphas-from", str(tmp_path / "sel.json"), "--out", "o"])
    assert lf._alphas(args) == sorted({0.0, sel["alpha_star"], 1.0})

    # branch per-query files of the "same job" = copies of alpha=1 / alpha=0 relabelled
    for f in FRACTIONS:
        for alpha, prefix, strategy, label in ((1.0, world.root / "pq" / "vision_x", "nowalls-random",
                                                f"nowalls-random f={f}"),
                                               (0.0, world.root / "pq" / "graph_x", "random",
                                                f"random f={f}")):
            d = load_perquery(lf.fused_partial_path(out, alpha, f, SPLIT))
            meta = {**d.meta, "branch": "x", "partial_label": label}
            write_npz(fraction_path(prefix, f, SPLIT, strategy), meta=meta, names=d.names, qi=d.qi,
                      ndcg=d.ndcg, recall=d.recall, map_=d.map, num_relevant=d.num_relevant,
                      ret_rows=d.ret_rows, n_ret=d.n_ret, self_rr=d.self_rr)
    report = fs.cmd_check(fs.parse_args([
        "check", "--split", SPLIT, "--fusion-dir", str(out),
        "--gallery-names", str(world.root / "shared.json"),
        "--vision-qvec", str(world.vprefix), "--graph-qvec", str(world.gprefix),
        "--vision-perquery", str(world.root / "pq" / "vision_x"),
        "--graph-perquery", str(world.root / "pq" / "graph_x")]))
    for c in ("C1", "C2", "C3"):
        assert report[c]["status"] == "PASS", c
    assert report["C5"]["status"] in ("PASS", "FAIL")     # synthetic MRR: range not meaningful

    # same file names in another folder (e.g. a historical run) -> C3 must fail
    other = world.root / "hist"
    other.mkdir()
    for f in FRACTIONS:
        src = fraction_path(world.root / "pq" / "vision_x", f, SPLIT, "nowalls-random")
        (other / src.name).write_bytes(src.read_bytes())
    report = fs.cmd_check(fs.parse_args([
        "check", "--split", SPLIT, "--fusion-dir", str(out),
        "--gallery-names", str(world.root / "shared.json"),
        "--vision-qvec", str(world.vprefix), "--graph-qvec", str(world.gprefix),
        "--vision-perquery", str(other / "vision_x"),
        "--graph-perquery", str(world.root / "pq" / "graph_x")]))
    assert report["C3"]["status"] == "FAIL" and report["C3"]["same_job"] is False


# --- T7: pure rules of select and C3 ---

def _cmp(lo, hi, delta=0.0):
    return {"ci_lo": lo, "ci_hi": hi, "delta": delta}


def test_ties_are_the_alphas_whose_ci_contains_zero():
    comps = {0.0: _cmp(0.01, 0.05), 0.4: _cmp(-0.01, 0.02), 0.5: _cmp(0.0, 0.0),
             0.6: _cmp(-0.03, 0.0), 1.0: _cmp(0.001, 0.1)}
    assert fs.ties_from_comparisons(comps) == [0.4, 0.5, 0.6]


def test_central_alpha_odd_even_and_equidistant():
    means = {a: 0.5 for a in (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8)}
    assert fs.central_alpha([0.2, 0.8, 0.3], means) == 0.3                 # odd: middle
    assert fs.central_alpha([0.1, 0.2, 0.4, 0.8], means) == 0.4           # even: closer to 0.5
    means_eq = {0.3: 0.40, 0.7: 0.41}
    assert fs.central_alpha([0.3, 0.7], means_eq) == 0.7                  # equidistant -> higher AUC
    means_eq = {0.3: 0.42, 0.7: 0.41}
    assert fs.central_alpha([0.3, 0.7], means_eq) == 0.3
    assert fs.central_alpha([0.3, 0.7], {0.3: 0.4, 0.7: 0.4}) == 0.3      # full tie: smaller alpha
    assert fs.central_alpha([1.0], {1.0: 0.3}) == 1.0
    with pytest.raises(ValueError):
        fs.central_alpha([], {})


def test_central_alpha_even_pair_closer_to_half():
    means = {a: 0.5 for a in (0.1, 0.6, 0.7, 0.8)}
    # sorted ties 0.1, 0.6, 0.7, 0.8: central pair (0.6, 0.7), 0.6 is closer to 0.5
    assert fs.central_alpha([0.1, 0.6, 0.7, 0.8], means) == 0.6


def test_reference_is_the_branch_with_higher_mean_auc():
    assert fs.reference_alpha({0.0: 0.45, 0.5: 0.60, 1.0: 0.39}) == 0.0
    assert fs.reference_alpha({0.0: 0.30, 1.0: 0.39}) == 1.0


def test_verdict_rule_and_noise_threshold():
    assert fs.verdict(0.5, _cmp(0.01, 0.09, 0.05)) == {
        "helps": True, "below_noise": False, "text": "la fusione aiuta"}
    v = fs.verdict(0.5, _cmp(0.001, 0.05, 0.03))
    assert v["helps"] is True and v["below_noise"] is True
    assert fs.verdict(0.5, _cmp(-0.001, 0.05, 0.05))["helps"] is False     # CI touches 0
    assert fs.verdict(1.0, _cmp(0.01, 0.09, 0.05))["helps"] is False       # a single branch
    assert fs.verdict(0.0, _cmp(0.01, 0.09, 0.05))["helps"] is False
    assert fs.verdict(0.5, _cmp(0.04, 0.05, 0.04))["below_noise"] is False  # 0.04 is not < 0.04


def test_oracle_is_descriptive_and_can_be_exceeded():
    pf0 = {0.25: np.array([1.0, 0.5, 0.0]), 0.5: np.array([1.0, 0.5, 0.0]), 0.75: np.array([0.5, 0.5, 0.0])}
    pf1 = {0.25: np.array([0.5, 1.0, 0.0]), 0.5: np.array([0.5, 0.2, 0.0]), 0.75: np.array([1.0, 0.1, 0.0])}
    orc = fs.oracle_auc(pf0, pf1)
    np.testing.assert_allclose(orc, [1.0, (1.0 + 0.5 + 0.5) / 3, 0.0])
    star = np.array([0.9, 0.9, 0.5])       # beats the oracle on 2 queries
    s = fs.oracle_summary(star, orc)
    assert s["n_star_exceeds_oracle"] == 2 and s["n"] == 3
    assert s["delta_star_minus_oracle"] == pytest.approx(float(np.mean(star - orc)))


def test_c3_threshold():
    max_k = 100
    a = np.full(2000, 1.0, dtype=np.float32)
    b = a.copy()
    b[:6] = 0.5                      # 6 queries moved from rank 1 to rank 2
    s = fs.c3_stats(a, b, max_k)
    assert s["n_diff"] == 6 and s["max_jump"] == 1 and s["jumps"] == {"1": 6}
    assert fs.c3_pass([s, s])
    b[6] = 0.5
    assert not fs.c3_pass([fs.c3_stats(a, b, max_k)])          # 7 differ
    c = a.copy()
    c[0] = 1.0 / 4                   # jump of 3
    assert not fs.c3_pass([fs.c3_stats(a, c, max_k)])
    d = a.copy()
    d[0] = 0.0                       # self lost -> rank max_k + 1
    assert fs.c3_stats(a, d, max_k)["max_jump"] == max_k
    assert fs.c5_pass(0.97) and not fs.c5_pass(0.96)


# --- fit_split, spread, ties, notes, self at the top-k boundary ---

def test_refuses_vision_whitening_not_fit_on_train(world):
    for f in FRACTIONS:
        world.write_vision(f, meta_update={"whitening": {"enabled": True, "fit_split": "all",
                                                          "eps": 1e-6, "dim": None}})
    with pytest.raises(ValueError, match="fit_split"):
        lf.main(world.argv())


def test_similarity_spread_by_hand():
    scores = np.array([[0.9, 0.8, 0.5, 0.1], [1.0, 0.7, 0.6, 0.2]], dtype=np.float32)
    ranked = np.array([[3, 1, 2, 0], [1, 2, -1, 0]])
    part = lf.similarity_spread(scores, ranked, qi=[3, 1], partial=True, k=3)
    # query 0: [0.9, 0.8, 0.5] (self kept) · query 1: only 2 valid in top-3 -> [1.0, 0.7, 0.2]
    exp0, exp1 = np.array([0.9, 0.8, 0.5]), np.array([1.0, 0.7, 0.2])
    assert part["n"] == 2
    assert part["mean_std_topk"] == pytest.approx((np.std(exp0) + np.std(exp1)) / 2, abs=1e-6)
    assert part["mean_gap_rank1_rankk"] == pytest.approx((0.4 + 0.8) / 2, abs=1e-6)
    full = lf.similarity_spread(scores, ranked, qi=[3, 1], partial=False, k=3)
    # self removed: query 0 -> [0.8, 0.5, 0.1]; query 1 -> [0.7, 0.2] too short, skipped
    assert full["n"] == 1
    assert full["mean_gap_rank1_rankk"] == pytest.approx(0.7, abs=1e-6)


def test_spread_sidecar_matches_the_single_branches(world):
    lf.main(world.argv())
    sp = json.loads(lf.spread_path(world.root / "fused", SPLIT).read_text())
    assert sp["note"] == lf.ALPHA_NOMINAL_NOTE and sp["k"] == lf.SPREAD_K
    assert set(sp["vision"]) == set(sp["graph"]) == {f"f{f}" for f in FRACTIONS} | {"full"}
    # graph at f=0.5: FAISS scores of the graph gallery alone are sim_G
    index = faiss.IndexFlatIP(DG)
    index.add(world.gg)
    sc, rk = index.search(world.gq[0.5], max(K_VALUES) + 1)
    top = sc[:, :lf.SPREAD_K].astype(np.float64)
    assert sp["graph"]["f0.5"]["mean_std_topk"] == pytest.approx(float(np.mean(np.std(top, axis=1))), abs=1e-5)
    assert sp["graph"]["f0.5"]["mean_gap_rank1_rankk"] == pytest.approx(
        float(np.mean(top[:, 0] - top[:, -1])), abs=1e-5)


def test_no_sidecar_without_the_two_endpoints(world):
    lf.main(world.argv(alphas=[0.5]))
    assert not lf.spread_path(world.root / "fused", SPLIT).exists()


def test_tie_set_shape():
    grid = [i / 10 for i in range(11)]
    assert fs.tie_set_shape(grid, [0.3], 0.3) == {"only_best": True, "contiguous": True}
    assert fs.tie_set_shape(grid, [0.3, 0.4, 0.5], 0.4) == {"only_best": False, "contiguous": True}
    assert fs.tie_set_shape(grid, [0.3, 0.5], 0.3) == {"only_best": False, "contiguous": False}
    assert fs.split_notes("valid") == [fs.VALID_NOTE] and fs.split_notes("test") == [fs.TEST_NOTE]


def test_select_output_carries_ties_table_spread_and_declarations(world, tmp_path):
    grid = [i / 10 for i in range(11)]
    lf.main(world.argv(alphas=grid))
    out = world.root / "fused"
    fs.main(["select", "--split", SPLIT, "--fusion-dir", str(out), "--out-json", str(tmp_path / "s.json")])
    sel = json.loads((tmp_path / "s.json").read_text())
    assert [row["alpha"] for row in sel["ties_table"]] == grid
    assert [row["alpha"] for row in sel["ties_table"] if row["tie"]] == sel["ties"]
    for row in sel["ties_table"]:
        c = sel["best_vs_alpha"][f"{row['alpha']:g}"]
        assert (row["ci_lo"], row["ci_hi"], row["delta_best_minus_alpha"]) == (c["ci_lo"], c["ci_hi"], c["delta"])
    assert sel["ties_shape"] == fs.tie_set_shape(grid, sel["ties"], sel["best"])
    assert fs.VALID_NOTE in sel["notes"]
    assert sel["verdict"]["notes"] == list(fs.VERDICT_NOTES)
    assert sel["full"]["note"] == fs.FULL_NOTE
    assert sel["similarity_spread"]["note"] == lf.ALPHA_NOMINAL_NOTE
    assert "full" in sel["similarity_spread"]["vision"]
    # test split, fixed alpha: conservative-reference declaration (valid files renamed to *_test;
    # `select` reads files, not the meta split)
    import shutil
    (world.root / "fused_test").mkdir()
    for path in out.iterdir():
        shutil.copy(path, world.root / "fused_test" / path.name.replace("_valid.", "_test."))
    fs.main(["select", "--split", "test", "--fusion-dir", str(world.root / "fused_test"),
             "--fixed-alpha-from", str(tmp_path / "s.json"), "--out-json", str(tmp_path / "t.json")])
    t = json.loads((tmp_path / "t.json").read_text())
    assert fs.TEST_NOTE in t["notes"] and fs.VALID_NOTE not in t["notes"]
    assert "ties_table" not in t


def _vision_rank(w, vg_index, raw_query, qi):
    stub = _Stub()
    stub.whiten_mean, stub.whiten_matrix = w.wmean, w.wmatrix
    _, idx = vg_index.search(stub._apply_whitening(raw_query[None, :]), N)
    return int(np.where(idx[0] == qi)[0][0]) + 1


def test_self_exactly_at_the_top_k_boundary(world):
    """Self at rank max_k (rr = 1/max_k) and max_k + 1 (rr = 0); catches an off-by-one on the top-k slice."""
    max_k = max(K_VALUES)
    stub = _Stub()
    stub.whiten_mean, stub.whiten_matrix = world.wmean, world.wmatrix
    index = faiss.IndexFlatIP(DV)
    index.add(stub._apply_whitening(world.raw))
    f = 0.25
    targets = {0: max_k, 1: max_k + 1}
    for j, rank in targets.items():
        qi = int(world.qi[j])
        found = None
        for other in range(N):
            if other == qi:
                continue
            for t in np.linspace(0.0, 1.0, 801):
                q = ((1 - t) * world.raw[qi] + t * world.raw[other]).astype(np.float32)
                r = _vision_rank(world, index, q, qi)
                if r == rank:
                    found = q
                    break
                if r > rank:
                    break
            if found is not None:
                break
        assert found is not None, f"no synthetic query with self at rank {rank}"
        world.vq[f][j] = found
    world.write_vision(f)
    lf.main(world.argv(alphas=[1.0], modes=("partial",)))
    d = load_perquery(lf.fused_partial_path(world.root / "fused", 1.0, f, SPLIT))
    assert d.self_rr[0] == np.float32(1.0 / max_k)
    assert d.self_rr[1] == np.float32(0.0)
    reference = _vision_reference(world, f)
    assert reference[0][1] == pytest.approx(1.0 / max_k) and reference[1][1] == 0.0
    _assert_matches(d, reference)
