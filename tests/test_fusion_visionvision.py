# tests/test_fusion_visionvision.py

"""
CPU smoke tests of the control «different models, same information» (status.md
§51): vision + vision fusion (`late_fusion --pair vision-vision`), its
`check`/`select`, and `complementarity --control-pair vision-vision` (four-way
rule, descriptives). Synthetic world of tests/test_late_fusion.py plus a second
vision run with its OWN whitening, and the graph replica of
tests/test_fusion_graphgraph.py for the gain scale and the Spearman.

Run: python -m pytest tests/test_fusion_visionvision.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from src.evaluation import fusion_select as fs
from src.evaluation import late_fusion as lf
from src.evaluation.perquery import load_perquery, write_npz
from src.evaluation.query_vectors import array_sha1, write_qvec
from src.evaluation.robustness_auc import fraction_path, load_auc
from src.vision.evaluation.evaluate import partial_rows
from tests.test_fusion_graphgraph import add_replica, gg_argv
from tests.test_late_fusion import (  # noqa: F401  (fixture re-exported)
    DV, FRACTIONS, K_VALUES, N, Q, SPLIT, World, _Stub, _assert_matches, _vision_reference, world,
)

GAMMAS = [0.0, 0.5, 1.0]


def add_vision2(w: World, seed: int = 23) -> World:
    """Second vision run («radio»): own RAW gallery, own whitening fit on its train rows."""
    rng = np.random.default_rng(seed)
    w.raw2 = (0.7 * w.raw + rng.normal(size=(N, DV))).astype(np.float32)
    stub = _Stub()
    stub._fit_whitening(w.raw2[:300])
    w.wmean2, w.wmatrix2 = stub.whiten_mean, stub.whiten_matrix
    vdir = w.root / "emb_vision2"
    vdir.mkdir()
    np.save(vdir / "embeddings.npy", w.raw2)
    (vdir / "image_paths.json").write_text(json.dumps([f"/snap/{n}.png" for n in w.shared]))
    w.vision2_emb = vdir / "embeddings.npy"
    w.v2prefix = w.root / "qv" / "vision_radio_natural_whiten-train"
    w.vq2 = {}
    for f in FRACTIONS:
        w.vq2[f] = w.raw2[w.qi] + (0.2 + f) * rng.normal(size=(Q, DV)).astype(np.float32)
        write_vision2(w, f)
    return w


def write_vision2(w, f, meta_update=None, rows=None, **arrays):
    rows = np.arange(Q) if rows is None else rows
    meta = w.base_meta("vision", f, f"nowalls-random f={f}")
    meta.update({"run_tag": "radio_natural_whiten-train",
                 "perquery_file": str(fraction_path(w.root / "pq" / "vision2_x", f, SPLIT,
                                                    "nowalls-random")),
                 "gallery_vectors": {"path": str(w.vision2_emb), "sha1": array_sha1(w.raw2),
                                     "shape": [N, DV]},
                 "whitening": {"enabled": True, "fit_split": "train", "eps": 1e-6, "dim": None},
                 "head": None})
    meta.update(meta_update or {})
    kw = dict(names=[w.shared[q] for q in w.qi[rows]], qi=w.qi[rows], vectors=w.vq2[f][rows],
              removed=[w.removed[f][i] for i in rows], whiten_mean=w.wmean2,
              whiten_matrix=w.wmatrix2)
    kw.update(arrays)
    write_qvec(fraction_path(w.v2prefix, f, SPLIT, "nowalls-random"), meta=meta, **kw)


def vv_argv(w, out="vv", gammas=GAMMAS, modes=("partial", "full")):
    return ["run", "--pair", "vision-vision", "--split", SPLIT,
            "--gallery-names", str(w.root / "shared.json"),
            "--vision-qvec", str(w.vprefix), "--vision2-qvec", str(w.v2prefix),
            "--vision-embeddings", str(w.vision_emb), "--vision2-embeddings", str(w.vision2_emb),
            "--fractions", *[str(f) for f in FRACTIONS], "--alphas", *[str(g) for g in gammas],
            "--modes", *modes, "--out", str(w.root / out)]


@pytest.fixture
def vworld(world):
    return add_vision2(world)


def _vv_path(w, gamma, f, out="vv"):
    return lf.fused_partial_path(w.root / out, gamma, f, SPLIT, "nowalls-random")


def _reference(w, raw_gallery, raw_queries, wmean, wmatrix):
    """Vision job loop (FAISS + `evaluate.partial_rows`) with the given whitening."""
    stub = _Stub()
    stub.whiten_mean, stub.whiten_matrix = wmean, wmatrix
    index = faiss.IndexFlatIP(DV)
    index.add(stub._apply_whitening(raw_gallery))
    stem2row = {n: i for i, n in enumerate(w.shared)}
    max_k = max(K_VALUES)
    out = []
    for j, qi in enumerate(w.qi):
        scores, idx = index.search(stub._apply_whitening(raw_queries[j:j + 1]), max_k + 1)
        results = [{"path": f"/snap/{w.shared[i]}.png", "score": float(s)}
                   for s, i in zip(scores[0], idx[0])]
        self_rows, axis_rows = partial_rows(results, int(qi), stem2row, max_k)
        hit = np.where(np.asarray(self_rows) == qi)[0]
        out.append((axis_rows, 1.0 / (int(hit[0]) + 1) if len(hit) else 0.0))
    return out


# ======================================================================
# late_fusion --pair vision-vision
# ======================================================================

def test_gamma_one_is_vision1_and_gamma_zero_is_vision2_each_with_its_own_whitening(vworld):
    w = vworld
    lf.main(vv_argv(w))
    for f in FRACTIONS:
        _assert_matches(load_perquery(_vv_path(w, 1.0, f)), _vision_reference(w, f))
        _assert_matches(load_perquery(_vv_path(w, 0.0, f)),
                        _reference(w, w.raw2, w.vq2[f], w.wmean2, w.wmatrix2))
    # the whitening of the OTHER job would give a different ranking somewhere
    cross = [_reference(w, w.raw2, w.vq2[f], w.wmean, w.wmatrix) for f in FRACTIONS]
    own = [_reference(w, w.raw2, w.vq2[f], w.wmean2, w.wmatrix2) for f in FRACTIONS]
    assert cross != own
    d = load_perquery(_vv_path(w, 0.5, 0.25))
    assert _vv_path(w, 0.5, 0.25).name == "fusion_a0.5_partial-nowalls-random-f0.25_valid.npz"
    assert d.meta["fusion"]["pair"] == "vision-vision"
    assert d.meta["fusion"]["vision1"]["run_tag"] == "tag"
    assert d.meta["fusion"]["vision2"]["run_tag"] == "radio_natural_whiten-train"
    assert d.meta["fusion"]["vision2"]["whitening"]["fit_split"] == "train"
    assert d.meta["damage"]["strategy"] == "nowalls_random"
    assert len(load_auc(lf.fusion_prefix(w.root / "vv", 0.5), SPLIT, strategy="nowalls-random").names) == Q


def test_refuses_different_removed_rooms(vworld):
    removed = [list(r) for r in vworld.removed[0.5]]
    removed[7] = [2] if removed[7] != [2] else [0]
    write_vision2(vworld, 0.5, removed=removed)
    with pytest.raises(ValueError) as exc:
        lf.main(vv_argv(vworld))
    msg = str(exc.value)
    assert vworld.shared[vworld.qi[7]] in msg and "f=0.5" in msg and "vision2" in msg


def test_refuses_rewritten_galleries(vworld):
    emb = np.load(vworld.vision2_emb)
    emb[3] += 1.0
    np.save(vworld.vision2_emb, emb)
    with pytest.raises(ValueError, match="vision2 gallery vectors"):
        lf.main(vv_argv(vworld))


@pytest.mark.parametrize("update,match", [
    ({"whitening": {"enabled": True, "fit_split": "all", "eps": 1e-6, "dim": None}}, "fit_split"),
    ({"head": {"enabled": True, "file": "head.pt"}}, "head"),
])
def test_refuses_whitening_not_on_train_or_head(vworld, update, match):
    for f in FRACTIONS:
        write_vision2(vworld, f, meta_update=update)
    with pytest.raises(ValueError, match=match):
        lf.main(vv_argv(vworld))


def test_refuses_same_run_test_split_and_mixed_folders(vworld):
    w = vworld
    for f in FRACTIONS:
        write_vision2(w, f, meta_update={"run_tag": "tag"})
    with pytest.raises(ValueError, match="same run"):
        lf.main(vv_argv(w))
    for f in FRACTIONS:
        write_vision2(w, f)
    argv = vv_argv(w)
    argv[argv.index("--split") + 1] = "test"
    with pytest.raises(ValueError, match="valid only"):
        lf.main(argv)
    lf.main(w.argv(out="mixed", alphas=[0.5], modes=("partial",)))
    with pytest.raises(ValueError, match="vision-graph fused files"):
        lf.main(vv_argv(w, out="mixed", gammas=[0.5], modes=("partial",)))
    lf.main(vv_argv(w, out="mixed2", gammas=[0.5], modes=("partial",)))
    with pytest.raises(ValueError, match="vision-vision fused files"):
        lf.main(w.argv(out="mixed2", alphas=[0.5], modes=("partial",)))


# ======================================================================
# check / select
# ======================================================================

def _same_job_perquery(w, beta, prefix, out="vv"):
    for f in FRACTIONS:
        d = load_perquery(_vv_path(w, beta, f, out))
        meta = {**d.meta, "branch": "vision", "partial_label": f"nowalls-random f={f}"}
        meta.pop("fusion")
        write_npz(fraction_path(prefix, f, SPLIT, "nowalls-random"), meta=meta, names=d.names,
                  qi=d.qi, ndcg=d.ndcg, recall=d.recall, map_=d.map, num_relevant=d.num_relevant,
                  ret_rows=d.ret_rows, n_ret=d.n_ret, self_rr=d.self_rr)


def test_check_and_select_visionvision(vworld, tmp_path):
    w = vworld
    lf.main(vv_argv(w))
    _same_job_perquery(w, 1.0, w.root / "pq" / "vision_x")
    _same_job_perquery(w, 0.0, w.root / "pq" / "vision2_x")
    argv = ["check", "--pair", "vision-vision", "--split", SPLIT, "--fusion-dir", str(w.root / "vv"),
            "--gallery-names", str(w.root / "shared.json"),
            "--vision-qvec", str(w.vprefix), "--vision2-qvec", str(w.v2prefix),
            "--vision-perquery", str(w.root / "pq" / "vision_x"),
            "--vision2-perquery", str(w.root / "pq" / "vision2_x"),
            "--vision2-hist", str(w.root / "does_not_exist" / "radio"),        # C4 not blocking
            "--vision2-full-hist", str(w.root / "does_not_exist.npz"),          # C3-full absent
            "--alphas", *[str(g) for g in GAMMAS]]
    report = fs.cmd_check(fs.parse_args(argv))
    for c in ("C1", "C2", "C3"):
        assert report[c]["status"] == "PASS", c
    assert report["C5"]["range"] == [fs.C5_LOW, None]
    for v in report["C5"]["per_alpha"].values():
        assert v["ok"] == (v["mrr"] >= fs.C5_LOW)
    assert "error" in report["C4_descriptive"]["vision2"]
    assert "error" in report["C3_full_descriptive"]["vision2"]
    # C3 fails when the gamma=0 per-query comes from the other job
    bad = list(argv)
    bad[bad.index("--vision2-perquery") + 1] = str(w.root / "pq" / "vision_x")
    assert fs.cmd_check(fs.parse_args(bad))["C3"]["status"] == "FAIL"

    fs.main(["select", "--pair", "vision-vision", "--split", SPLIT, "--fusion-dir", str(w.root / "vv"),
             "--alphas", *[str(g) for g in GAMMAS], "--out-json", str(tmp_path / "vv.json")])
    sel = json.loads((tmp_path / "vv.json").read_text())
    assert sel["pair"] == "vision-vision" and sel["strategy"] == "nowalls-random"
    assert sel["verdict"]["notes"] == list(fs.VISIONVISION_NOTES)
    assert "star_minus_vision1" in sel["full"]
    assert fs.pair_spec("vision-vision")["select_json"].format(split="valid") == \
        "select_visionvision_valid.json"
    assert fs.pair_spec("vision-vision")["c5_high"] is None
    with pytest.raises(ValueError, match="valid only"):
        fs.main(["select", "--pair", "vision-vision", "--split", "test", "--fusion-dir",
                 str(w.root / "vv"), "--out-json", str(tmp_path / "y.json")])


# ======================================================================
# §51 decision: four-way rule, Spearman, end to end
# ======================================================================

@pytest.mark.parametrize("lo,hi,expected", [
    (0.001, 0.2, "complementarity"),
    (-0.2, -0.001, "refuted"),
    (-0.04, 0.04, "equivalent"),          # bounds included
    (-0.01, 0.03, "equivalent"),
    (0.0, 0.04, "equivalent"),            # CI touching 0 still contains it
    (-0.0401, 0.01, "inconclusive"),
    (-0.01, 0.0401, "inconclusive"),
    (0.0, 0.05, "inconclusive"),
])
def test_four_way_verdict(lo, hi, expected):
    assert fs.four_way_verdict(lo, hi) == expected
    assert fs.NEGLIGIBLE_DELTA == 0.04


def test_three_way_rule_of_section_50_unchanged():
    assert fs.complementarity_verdict(-0.01, 0.02) == "indistinguishable"
    assert fs.complementarity_verdict(-0.3, 0.3) == "indistinguishable"


def test_four_way_stats_on_constructed_cases_and_determinism():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.2, 0.6, size=2000)
    g = rng.normal(scale=0.01, size=2000)
    g -= g.mean()
    eq = fs.complementarity_stats(base + 0.1 + g, base, base + 0.1 - g, base, rule="four_way")
    assert eq["verdict"] == "equivalent"
    big = rng.normal(scale=2.0, size=2000)
    big -= big.mean()
    inc = fs.complementarity_stats(base + 0.1 + big, base, base + 0.1, base, rule="four_way")
    assert inc["verdict"] == "inconclusive"
    comp = fs.complementarity_stats(base + 0.2 + g, base, base + 0.05, base, rule="four_way")
    assert comp["verdict"] == "complementarity"
    ref = fs.complementarity_stats(base + 0.05 + g, base, base + 0.2, base, rule="four_way")
    assert ref["verdict"] == "refuted"
    assert fs.complementarity_stats(base + 0.1 + big, base, base, base, rule="four_way") == \
        fs.complementarity_stats(base + 0.1 + big, base, base, base, rule="four_way")


def test_spearman_known_cases():
    assert fs.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert fs.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    np.testing.assert_array_equal(fs.average_ranks([3.0, 1.0, 3.0, 2.0]), [3.5, 1.0, 3.5, 2.0])
    # ties: ranks a = [1, 2.5, 2.5, 4], b = [1, 3, 2, 4]
    expected = np.corrcoef([1, 2.5, 2.5, 4], [1, 3, 2, 4])[0, 1]
    assert fs.spearman([1, 2, 2, 3], [1, 3, 2, 4]) == pytest.approx(expected)
    assert np.isnan(fs.spearman([1, 1, 1], [1, 2, 3]))
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 5, 50), rng.normal(size=50)
    assert fs.spearman(a, b) == pytest.approx(scipy_stats.spearmanr(a, b).correlation)


def _full_world(world, tmp_path):
    """True fusion (vg), graph+graph control with its complementarity json, vision+vision."""
    w = world
    for f in FRACTIONS:
        w.write_graph(f, meta_update={"run_tag": "gat_asymrob"})
    add_replica(w)
    add_vision2(w)
    lf.main(w.argv(out="fused", alphas=GAMMAS, modes=("partial",)))
    lf.main(gg_argv(w, out="gg", betas=GAMMAS, modes=("partial",)))
    lf.main(vv_argv(w, out="vv", gammas=GAMMAS, modes=("partial",)))
    sel = {"t": ("fused", 0.5), "c": ("gg", 0.5), "v": ("vv", 0.5)}
    for key, (d, star) in sel.items():
        (tmp_path / f"{key}.json").write_text(json.dumps({
            "split": SPLIT, "alpha_star": star, "fusion_dir": str(w.root / d),
            "full": {"means": {"1": {"composition": 0.5, "topology": 0.4, "geometry": 0.9},
                               "0.5": {"composition": 0.52, "topology": 0.41, "geometry": 0.9}}}}))
    fs.main(["complementarity", "--split", SPLIT,
             "--true-select", str(tmp_path / "t.json"), "--true-dir", str(w.root / "fused"),
             "--control-select", str(tmp_path / "c.json"), "--control-dir", str(w.root / "gg"),
             "--out-json", str(tmp_path / "gg_comp.json")])
    return ["complementarity", "--control-pair", "vision-vision", "--split", SPLIT,
            "--true-select", str(tmp_path / "t.json"), "--true-dir", str(w.root / "fused"),
            "--control-select", str(tmp_path / "v.json"), "--control-dir", str(w.root / "vv"),
            "--graphgraph-json", str(tmp_path / "gg_comp.json"), "--graphgraph-dir", str(w.root / "gg"),
            "--out-json", str(tmp_path / "vv_comp.json")]


def test_complementarity_visionvision_end_to_end(world, tmp_path):
    w = world
    argv = _full_world(w, tmp_path)
    fs.main(argv)
    r = json.loads((tmp_path / "vv_comp.json").read_text())
    gg = json.loads((tmp_path / "gg_comp.json").read_text())
    assert r["rule"] == "status.md §51" and r["margin"] == 0.04
    assert r["verdict"] == fs.four_way_verdict(r["D2"]["ci_lo"], r["D2"]["ci_hi"])
    assert r["verdict_text"] == fs.VISIONVISION_VERDICTS[r["verdict"]]
    assert r["notes"][:len(fs.VISIONVISION_COMPLEMENTARITY_NOTES)] == list(fs.VISIONVISION_COMPLEMENTARITY_NOTES)
    # gain scale: G_C read (not recomputed) from the §50 json
    assert {k: r["gain_scale"]["G_C"][k] for k in ("mean", "ci_lo", "ci_hi")} == \
        {k: gg["G_C"][k] for k in ("mean", "ci_lo", "ci_hi")}
    assert "letto da" in r["gain_scale"]["G_C"]["source"]
    assert r["gain_scale"]["G_F"] == gg["G_F"]

    F = {g: load_auc(lf.fusion_prefix(w.root / "fused", g), SPLIT, strategy="nowalls-random") for g in GAMMAS}
    V = {g: load_auc(lf.fusion_prefix(w.root / "vv", g), SPLIT, strategy="nowalls-random") for g in GAMMAS}
    C = {g: load_auc(lf.fusion_prefix(w.root / "gg", g), SPLIT, strategy="random") for g in GAMMAS}

    def by_name(d):
        return dict(zip(map(str, d.names), d.auc))
    names = list(map(str, F[0.5].names))
    comp_f = 1.0 if F[1.0].mean > F[0.0].mean else 0.0
    comp_v = 1.0 if V[1.0].mean > V[0.0].mean else 0.0
    assert r["control"]["best_component"] == comp_v
    f5, fc, v5, vc = by_name(F[0.5]), by_name(F[comp_f]), by_name(V[0.5]), by_name(V[comp_v])
    d2 = [(f5[n] - fc[n]) - (v5[n] - vc[n]) for n in names]
    assert r["D2"]["mean"] == pytest.approx(float(np.mean(d2)), abs=1e-12)

    def rho(d1, d0):
        a, b = by_name(d1), by_name(d0)
        return fs.spearman([a[n] for n in names], [b[n] for n in names])
    sp = r["spearman_components"]
    assert sp["vision/graph"] == pytest.approx(rho(F[1.0], F[0.0]))
    assert sp["pespatial/radio"] == pytest.approx(rho(V[1.0], V[0.0]))
    assert sp["graph/replica"] == pytest.approx(rho(C[1.0], C[0.0]))
    assert r["full_descriptive"]["V_star_minus_pespatial"] == pytest.approx(
        {"composition": 0.02, "topology": 0.01, "geometry": 0.0})
    # the §50 output is untouched and the default file names differ
    assert json.loads((tmp_path / "gg_comp.json").read_text()) == gg
    a = fs.parse_args(["complementarity", "--control-pair", "vision-vision"])
    assert a.control_select == "results/fusion/select_visionvision_valid.json"
    assert a.control_dir == "results/perquery/fusion_visionvision_valid"
    b = fs.parse_args(["complementarity"])
    assert b.control_select == "results/fusion/select_graphgraph_valid.json"


def test_complementarity_visionvision_refuses_gg_json_of_another_true_fusion(world, tmp_path):
    argv = _full_world(world, tmp_path)
    gg = json.loads((tmp_path / "gg_comp.json").read_text())
    gg["true"]["alpha_star"] = 0.7
    (tmp_path / "gg_comp.json").write_text(json.dumps(gg))
    with pytest.raises(ValueError, match="another true fusion"):
        fs.main(argv)
