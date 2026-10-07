
"""
CPU smoke tests of the ensemble-effect control: graph + graph fusion (`late_fusion --pair graph-graph`),
its `check`/`select`, and the `complementarity` decision. Synthetic world of tests/test_late_fusion.py plus a second graph run ("replica").

Usage: python -m pytest tests/test_fusion_graphgraph.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from src.evaluation import fusion_select as fs
from src.evaluation import late_fusion as lf
from src.evaluation.perquery import load_perquery, write_npz
from src.evaluation.query_vectors import array_sha1, write_qvec
from src.evaluation.robustness_auc import fraction_path, load_auc
from tests.test_late_fusion import (  # noqa: F401  (fixture re-exported)
    DG, FRACTIONS, K_VALUES, N, Q, SPLIT, World, _assert_matches, _unit, world,
)

BETAS = [0.0, 0.5, 1.0]


def add_replica(w: World, seed: int = 11) -> World:
    """Second graph run: own gallery (unit rows), own damaged queries, same rooms."""
    rng = np.random.default_rng(seed)
    w.gg2 = _unit(w.gg + 0.6 * rng.normal(size=(N, DG)))
    gdir = w.root / "emb_graph2"
    gdir.mkdir()
    np.save(gdir / "embeddings.npy", w.gg2)
    (gdir / "names.json").write_text(json.dumps(w.shared))
    w.graph2_emb = gdir / "embeddings.npy"
    w.g2prefix = w.root / "qv" / "graph_gat_asymrobrep"
    w.gq2 = {}
    for f in FRACTIONS:
        w.gq2[f] = _unit(w.gg2[w.qi] + (0.2 + f) * rng.normal(size=(Q, DG)))
        write_replica(w, f)
    return w


def write_replica(w, f, meta_update=None, rows=None, **arrays):
    rows = np.arange(Q) if rows is None else rows
    meta = w.base_meta("graph", f, f"random f={f}")
    meta.update({"run_tag": "gat_asymrobrep",
                 "perquery_file": str(fraction_path(w.root / "pq" / "graph2_x", f, SPLIT, "random")),
                 "gallery_vectors": {"path": str(w.graph2_emb), "sha1": array_sha1(w.gg2),
                                     "shape": [N, DG]},
                 "model": {"encoder": "gat"}, "n_degenerate": 0})
    meta.update(meta_update or {})
    kw = dict(names=[w.shared[q] for q in w.qi[rows]], qi=w.qi[rows], vectors=w.gq2[f][rows],
              removed=[w.removed[f][i] for i in rows])
    kw.update(arrays)
    write_qvec(fraction_path(w.g2prefix, f, SPLIT, "random"), meta=meta, **kw)


def gg_argv(w, out="gg", betas=BETAS, modes=("partial", "full")):
    return ["run", "--pair", "graph-graph", "--split", SPLIT,
            "--gallery-names", str(w.root / "shared.json"),
            "--graph-qvec", str(w.gprefix), "--graph2-qvec", str(w.g2prefix),
            "--graph-embeddings", str(w.graph_emb), "--graph2-embeddings", str(w.graph2_emb),
            "--fractions", *[str(f) for f in FRACTIONS], "--alphas", *[str(b) for b in betas],
            "--modes", *modes, "--out", str(w.root / out)]


@pytest.fixture
def gworld(world):
    # graph1 needs a distinct run_tag from the replica
    for f in FRACTIONS:
        world.write_graph(f, meta_update={"run_tag": "gat_asymrob"})
    return add_replica(world)


def _graph_reference_on(gallery, queries, qi):
    index = faiss.IndexFlatIP(gallery.shape[1])
    index.add(gallery)
    max_k = max(K_VALUES)
    _, ranked = index.search(queries, max_k + 1)
    out = []
    for j, q in enumerate(qi):
        row_list = [int(r) for r in ranked[j] if r != -1]
        axis_rows = [r for r in row_list if r != q][:max_k]
        hit = [i for i, r in enumerate(row_list[:max_k], start=1) if r == q]
        out.append((axis_rows, 1.0 / hit[0] if hit else 0.0))
    return out


def _gg_path(w, beta, f, out="gg"):
    return lf.fused_partial_path(w.root / out, beta, f, SPLIT, "random")


# --- late_fusion --pair graph-graph ---

def test_beta_one_is_graph1_and_beta_zero_is_graph2(gworld):
    w = gworld
    lf.main(gg_argv(w))
    for f in FRACTIONS:
        _assert_matches(load_perquery(_gg_path(w, 1.0, f)), _graph_reference_on(w.gg, w.gq[f], w.qi))
        _assert_matches(load_perquery(_gg_path(w, 0.0, f)), _graph_reference_on(w.gg2, w.gq2[f], w.qi))
    d = load_perquery(_gg_path(w, 0.5, 0.25))
    assert _gg_path(w, 0.5, 0.25).name == "fusion_a0.5_partial-random-f0.25_valid.npz"
    assert d.meta["fusion"]["pair"] == "graph-graph"
    assert d.meta["fusion"]["method"] == "weighted_concat_sqrt"
    assert d.meta["fusion"]["graph1"]["run_tag"] == "gat_asymrob"
    assert d.meta["fusion"]["graph2"]["run_tag"] == "gat_asymrobrep"
    assert d.meta["partial_label"] == "random f=0.25"
    auc = load_auc(lf.fusion_prefix(w.root / "gg", 0.5), SPLIT, strategy="random")
    assert len(auc.names) == Q
    full = load_perquery(lf.fused_full_path(w.root / "gg", 0.0, SPLIT))
    ref = _graph_reference_on(w.gg2, w.gg2[w.qi], w.qi)
    for j, qi in enumerate(w.qi):
        n = int(full.n_ret[j])
        # full: self removed, max_k rows from max_k + 1 answers
        assert full.ret_rows[j, :n].tolist() == [r for r in ref[j][0] if r != qi][:max(K_VALUES)]


def test_vision_graph_files_carry_no_pair_key(world):
    lf.main(world.argv(alphas=[0.5], modes=("partial",)))
    d = load_perquery(lf.fused_partial_path(world.root / "fused", 0.5, 0.25, SPLIT))
    assert "pair" not in d.meta["fusion"]


def test_graphgraph_refuses_different_removed_rooms(gworld):
    removed = [list(r) for r in gworld.removed[0.75]]
    removed[3] = [0] if removed[3] != [0] else [1]
    write_replica(gworld, 0.75, removed=removed)
    with pytest.raises(ValueError) as exc:
        lf.main(gg_argv(gworld))
    msg = str(exc.value)
    assert gworld.shared[gworld.qi[3]] in msg and "f=0.75" in msg and "graph2" in msg


def test_graphgraph_refuses_rewritten_replica_gallery(gworld):
    emb = np.load(gworld.graph2_emb)
    emb[1] = _unit(emb[1:2] + 0.3)[0]
    np.save(gworld.graph2_emb, emb)
    with pytest.raises(ValueError, match="graph2 gallery vectors"):
        lf.main(gg_argv(gworld))


def test_graphgraph_refuses_first_gallery_rewritten(gworld):
    emb = np.load(gworld.graph_emb)
    emb[1] = _unit(emb[1:2] + 0.3)[0]
    np.save(gworld.graph_emb, emb)
    with pytest.raises(ValueError, match="graph1 gallery vectors"):
        lf.main(gg_argv(gworld))


def test_graphgraph_refuses_same_run_and_non_unit_queries_and_test_split(gworld):
    for f in FRACTIONS:
        write_replica(gworld, f, meta_update={"run_tag": "gat_asymrob"})
    with pytest.raises(ValueError, match="same run"):
        lf.main(gg_argv(gworld))
    for f in FRACTIONS:
        write_replica(gworld, f)
    write_replica(gworld, 0.5, vectors=gworld.gq2[0.5] * 1.5)
    with pytest.raises(ValueError, match="graph2 queries f=0.5"):
        lf.main(gg_argv(gworld))
    argv = gg_argv(gworld)
    argv[argv.index("--split") + 1] = "test"
    with pytest.raises(ValueError, match="valid only"):
        lf.main(argv)


def test_a_folder_never_mixes_the_two_pairs(gworld):
    w = gworld
    lf.main(w.argv(out="mixed", alphas=[0.5], modes=("partial",)))
    with pytest.raises(ValueError, match="vision-graph fused files"):
        lf.main(gg_argv(w, out="mixed", betas=[0.5], modes=("partial",)))
    lf.main(gg_argv(w, out="mixed2", betas=[0.5], modes=("partial",)))
    with pytest.raises(ValueError, match="graph-graph fused files"):
        lf.main(w.argv(out="mixed2", alphas=[0.5], modes=("partial",)))


# --- check / select on graph-graph ---

def test_c5_upper_bound_only_for_vision_graph():
    assert fs.c5_pass(0.97) and not fs.c5_pass(0.99) and not fs.c5_pass(0.96)
    assert fs.c5_pass(0.99, None) and fs.c5_pass(1.0, None) and not fs.c5_pass(0.96, None)
    assert fs.pair_spec("vision-graph")["c5_high"] == fs.C5_HIGH
    # on the test the vision-graph pair keeps only the lower bound
    assert fs.pair_spec("vision-graph", "test")["c5_high"] is None
    assert fs.pair_spec("vision-graph", "valid")["c5_high"] == fs.C5_HIGH
    assert fs.pair_spec("graph-graph")["c5_high"] is None


def _same_job_perquery(w, out, beta, prefix):
    for f in FRACTIONS:
        d = load_perquery(_gg_path(w, beta, f, out))
        meta = {**d.meta, "branch": "graph", "partial_label": f"random f={f}"}
        meta.pop("fusion")
        write_npz(fraction_path(prefix, f, SPLIT, "random"), meta=meta, names=d.names, qi=d.qi,
                  ndcg=d.ndcg, recall=d.recall, map_=d.map, num_relevant=d.num_relevant,
                  ret_rows=d.ret_rows, n_ret=d.n_ret, self_rr=d.self_rr)


def _gg_check_argv(w, out="gg", betas=BETAS):
    return ["check", "--pair", "graph-graph", "--split", SPLIT, "--fusion-dir", str(w.root / out),
            "--gallery-names", str(w.root / "shared.json"),
            "--graph-qvec", str(w.gprefix), "--graph2-qvec", str(w.g2prefix),
            "--graph-perquery", str(w.root / "pq" / "graph_x"),
            "--graph2-perquery", str(w.root / "pq" / "graph2_x"),
            "--alphas", *[str(b) for b in betas]]


def test_check_and_select_graphgraph(gworld, tmp_path):
    w = gworld
    lf.main(gg_argv(w))
    _same_job_perquery(w, "gg", 1.0, w.root / "pq" / "graph_x")
    _same_job_perquery(w, "gg", 0.0, w.root / "pq" / "graph2_x")
    report = fs.cmd_check(fs.parse_args(_gg_check_argv(w)))
    for c in ("C1", "C2", "C3"):
        assert report[c]["status"] == "PASS", c
    assert report["C5"]["range"] == [fs.C5_LOW, None]
    assert set(report["C3"]) >= {"graph1", "graph2"}
    # C5 computed with the lower bound only
    for v in report["C5"]["per_alpha"].values():
        assert v["ok"] == (v["mrr"] >= fs.C5_LOW)

    fs.main(["select", "--pair", "graph-graph", "--split", SPLIT, "--fusion-dir", str(w.root / "gg"),
             "--alphas", *[str(b) for b in BETAS], "--out-json", str(tmp_path / "gg.json")])
    sel = json.loads((tmp_path / "gg.json").read_text())
    assert sel["pair"] == "graph-graph" and sel["strategy"] == "random"
    assert sel["verdict"]["notes"] == list(fs.GRAPHGRAPH_NOTES)
    assert "star_minus_graph1" in sel["full"] and "star_minus_graph2" in sel["full"]
    assert fs.pair_spec("graph-graph")["select_json"].format(split="valid") == "select_graphgraph_valid.json"
    # a vision-graph select on a graph-graph folder finds no nowalls-random files
    with pytest.raises((FileNotFoundError, ValueError)):
        fs.main(["select", "--split", SPLIT, "--fusion-dir", str(w.root / "gg"),
                 "--alphas", *[str(b) for b in BETAS], "--out-json", str(tmp_path / "x.json")])
    with pytest.raises(ValueError, match="valid only"):
        fs.main(["select", "--pair", "graph-graph", "--split", "test", "--fusion-dir",
                 str(w.root / "gg"), "--out-json", str(tmp_path / "y.json")])


def test_check_graphgraph_fails_c3_on_another_job(gworld):
    w = gworld
    lf.main(gg_argv(w))
    _same_job_perquery(w, "gg", 1.0, w.root / "pq" / "graph_x")
    _same_job_perquery(w, "gg", 0.0, w.root / "pq" / "graph2_x")
    # replica per-query taken from the graph1 job: same_job fails, self_rr differ
    argv = _gg_check_argv(w)
    argv[argv.index("--graph2-perquery") + 1] = str(w.root / "pq" / "graph_x")
    report = fs.cmd_check(fs.parse_args(argv))
    assert report["C3"]["status"] == "FAIL"


# --- complementarity ---

def test_complementarity_verdict_three_ways():
    assert fs.complementarity_verdict(0.001, 0.02) == "complementarity"
    assert fs.complementarity_verdict(-0.01, 0.02) == "indistinguishable"
    assert fs.complementarity_verdict(0.0, 0.02) == "indistinguishable"
    assert fs.complementarity_verdict(-0.02, 0.0) == "indistinguishable"
    assert fs.complementarity_verdict(-0.02, -0.001) == "ensemble"


def test_complementarity_stats_on_constructed_cases_and_determinism():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 0.6, size=2000)
    noise = lambda s: rng.normal(scale=s, size=2000)  # noqa: E731
    # F gains 0.2, C gains 0.02 -> complementarity
    s = fs.complementarity_stats(base + 0.2 + noise(0.05), base, base + 0.02 + noise(0.05), base)
    assert s["verdict"] == "complementarity" and s["D"]["ci_lo"] > 0
    assert s["share_G_C_over_G_F"] == pytest.approx(s["G_C"]["mean"] / s["G_F"]["mean"])
    # same gains in expectation, zero-mean paired noise -> indistinguishable
    g = rng.normal(scale=0.05, size=2000)
    s = fs.complementarity_stats(base + 0.1 + g, base, base + 0.1 - g, base)
    assert s["D"]["mean"] == pytest.approx(float(np.mean(2 * g)))
    e = fs.complementarity_stats(base + 0.1 + g - np.mean(g), base, base + 0.1 - g + np.mean(g), base)
    assert e["verdict"] == "indistinguishable"
    # C gains more -> ensemble
    s = fs.complementarity_stats(base + 0.02 + noise(0.05), base, base + 0.2 + noise(0.05), base)
    assert s["verdict"] == "ensemble" and s["D"]["ci_hi"] < 0
    # bootstrap determinism (seed 0)
    a = fs.complementarity_stats(base + 0.1 + g, base, base, base)
    b = fs.complementarity_stats(base + 0.1 + g, base, base, base)
    assert a == b


def test_align_by_name_reorders_and_refuses_different_sets():
    names_ref = ["a", "b", "c"]
    np.testing.assert_array_equal(fs.align_by_name(names_ref, ["c", "a", "b"], [3.0, 1.0, 2.0]),
                                  [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="query sets differ"):
        fs.align_by_name(names_ref, ["a", "b"], [1.0, 2.0])
    with pytest.raises(ValueError, match="query sets differ"):
        fs.align_by_name(names_ref, ["a", "b", "c", "d"], [1.0, 2.0, 3.0, 4.0])


def _run_both(w, tmp_path, a_star=0.5, b_star=0.5):
    lf.main(w.argv(out="fused", alphas=BETAS, modes=("partial",)))
    lf.main(gg_argv(w, out="gg", betas=BETAS, modes=("partial",)))
    (tmp_path / "t.json").write_text(json.dumps({"split": SPLIT, "alpha_star": a_star,
                                                 "fusion_dir": str(w.root / "fused")}))
    (tmp_path / "c.json").write_text(json.dumps({"split": SPLIT, "alpha_star": b_star, "full": None,
                                                 "fusion_dir": str(w.root / "gg")}))
    return ["complementarity", "--split", SPLIT,
            "--true-select", str(tmp_path / "t.json"), "--true-dir", str(w.root / "fused"),
            "--control-select", str(tmp_path / "c.json"), "--control-dir", str(w.root / "gg"),
            "--out-json", str(tmp_path / "comp.json")]


def test_complementarity_end_to_end_joins_by_name(gworld, tmp_path):
    w = gworld
    # graph-graph queries written in a different order: the fused C files follow it
    perm = np.random.default_rng(5).permutation(Q)
    for f in FRACTIONS:
        w.write_graph(f, rows=perm, meta_update={"run_tag": "gat_asymrob"})
        write_replica(w, f, rows=perm)
    fs.main(_run_both(w, tmp_path))
    r = json.loads((tmp_path / "comp.json").read_text())

    F = {b: load_auc(lf.fusion_prefix(w.root / "fused", b), SPLIT, strategy="nowalls-random") for b in BETAS}
    C = {b: load_auc(lf.fusion_prefix(w.root / "gg", b), SPLIT, strategy="random") for b in BETAS}
    assert list(C[0.5].names) != list(F[0.5].names)          # orders really differ
    comp_f = 1.0 if F[1.0].mean > F[0.0].mean else 0.0
    comp_c = 1.0 if C[1.0].mean > C[0.0].mean else 0.0
    assert r["true"]["best_component"] == comp_f and r["control"]["best_component"] == comp_c

    def by_name(d):
        return dict(zip(map(str, d.names), d.auc))
    f5, fc, c5, cc = by_name(F[0.5]), by_name(F[comp_f]), by_name(C[0.5]), by_name(C[comp_c])
    d = [(f5[n] - fc[n]) - (c5[n] - cc[n]) for n in map(str, F[0.5].names)]
    assert r["n"] == Q and r["D"]["mean"] == pytest.approx(float(np.mean(d)), abs=1e-12)
    assert r["verdict"] == fs.complementarity_verdict(r["D"]["ci_lo"], r["D"]["ci_hi"])
    assert r["notes"] == list(fs.COMPLEMENTARITY_NOTES)


def test_complementarity_refuses_different_query_sets(gworld, tmp_path):
    w = gworld
    for f in FRACTIONS:
        write_replica(w, f, rows=np.arange(1, Q))
    with pytest.raises(ValueError, match="query sets differ"):
        fs.main(_run_both(w, tmp_path))


def test_complementarity_refuses_select_json_of_another_folder(gworld, tmp_path):
    argv = _run_both(gworld, tmp_path)
    sel = json.loads((tmp_path / "c.json").read_text())
    sel["fusion_dir"] = str(gworld.root / "fused")
    (tmp_path / "c.json").write_text(json.dumps(sel))
    with pytest.raises(ValueError, match="fusion_dir"):
        fs.main(argv)


def test_complementarity_refuses_swapped_roles(gworld, tmp_path):
    argv = _run_both(gworld, tmp_path)
    argv[argv.index("--true-dir") + 1], argv[argv.index("--control-dir") + 1] = \
        argv[argv.index("--control-dir") + 1], argv[argv.index("--true-dir") + 1]
    with pytest.raises((ValueError, FileNotFoundError)):
        fs.main(argv)
