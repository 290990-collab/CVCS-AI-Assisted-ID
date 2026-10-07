
"""
Smoke test CPU of the second multi-seed round (`src/evaluation/multiseed_runs.py`, curve with `--seed`).

- new evaluations write only under results/final_pipeline/round2/ (gallery included), never in embeddings/;
- every control input is an evaluation of this round or one with damage = seed;
- the two control pairs compare the same true model against two different models, with the same removed rooms;
- the seed-42 curve stays where it was; other seeds read the graph branch from this round;
- jobs do not start without the unlock file.

Run: python -m pytest tests/test_multiseed_runs.py -v
"""

from __future__ import annotations

import pytest

from src.evaluation import fusion_curve as fc
from src.evaluation import multiseed_runs as r2
from src.graph import final_graph_configs as rc


def test_evaluations_write_only_under_round2():
    for split in r2.SPLITS:
        for e in r2.evals(split):
            p = r2.eval_paths(split, e["encoder"], e["cfg"], e["seed"], e["damage"])
            for k in ("PQ_DIR", "QV_DIR", "GALLERY_DIR"):
                assert p[k].startswith(str(r2.ROOT) + "/"), (k, p[k])
            assert p["DEST"].startswith("embeddings/graph/")      # read only: the checkpoint


def test_gallery_folders_are_unique_per_evaluation():
    for split in r2.SPLITS:
        dirs = [r2.eval_paths(split, e["encoder"], e["cfg"], e["seed"], e["damage"])["GALLERY_DIR"]
                for e in r2.evals(split)]
        assert len(dirs) == len(set(dirs))


def test_controls_pair_same_damage_and_differ_only_in_the_model():
    for split in r2.SPLITS:
        for c in r2.controls(split):
            (e1, c1, s1, d1), (e2, c2, s2, d2) = c["g1"], c["g2"]
            assert d1 == d2                                      # same removed rooms
            if "replica" in c["name"]:
                assert (e1, c1) == (e2, c2) and s1 != s2         # same config, another training
            else:
                assert e1 != e2 and c1 == c2                     # same change, another encoder


def test_control_inputs_are_round2_evals_or_own_damage():
    for split in r2.SPLITS:
        for i, c in enumerate(r2.controls(split)):
            p = r2.control_paths(split, i)
            for key, q in (("g1", p["G1_QVEC"]), ("g2", p["G2_QVEC"])):
                enc, cfg, seed, damage = c[key]
                if damage != seed:
                    assert q.startswith(str(r2.ROOT)), (c["name"], key)


def test_w_is_reevaluated_on_valid_for_every_replica():
    got = {(e["seed"], e["damage"]) for e in r2.evals("valid") if (e["encoder"], e["cfg"]) == r2.W}
    assert {(s, s) for s in rc.SEEDS} <= got


def test_graph_source_refuses_an_unknown_damage():
    with pytest.raises(ValueError):
        r2.graph_source("valid", "gcn", "ref", 42, 100042)


def test_gate_needs_the_unlock_files(tmp_path, monkeypatch):
    monkeypatch.setattr(r2, "GATES", {"valid": tmp_path / "v", "test": tmp_path / "t"})
    assert r2.gate("valid") and r2.gate("test")
    (tmp_path / "v").touch()
    assert not r2.gate("valid") and r2.gate("test")
    (tmp_path / "t").touch()
    assert not r2.gate("test")


def test_curve_seed42_unchanged_and_other_seeds_read_w_from_round2():
    p42 = fc.paths("frozen", "gat", "comb")
    assert p42["OUT"] == str(rc.root() / "curve" / "frozen" / "gat_comb" / "fusion_valid")
    p = fc.paths("frozen", "gat", "comb", seed=100042)
    assert p["OUT"].startswith(str(rc.root() / "curve" / "s100042") + "/")
    assert p["GRAPH_QVEC"].startswith(str(r2.ROOT))
    q = fc.paths("head", "gcn", "ref", seed=200042)
    assert "/s200042/" in q["GRAPH_QVEC"] and "s200042" in q["VISION_QVEC"]


def test_curve_other_seed_refuses_without_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(r2, "GATES", {"valid": tmp_path / "v", "test": tmp_path / "t"})
    with pytest.raises(SystemExit):
        fc.run(0, seed=100042)


def test_query_vectors_pin_the_gallery_written_by_this_evaluation(tmp_path):
    import argparse

    import numpy as np

    from src.graph.evaluation.graph_evaluate import query_vectors_context
    emb = np.eye(3, dtype=np.float32)
    base = dict(query_vectors_out=str(tmp_path / "qv"), save_dir="embeddings/graph/gat/x", baseline_hist=False,
                encoder="gat", variant="x")
    with_out = query_vectors_context(argparse.Namespace(**base, gallery_out=str(tmp_path / "g")), emb)
    assert with_out["gallery_vectors"]["path"] == str(tmp_path / "g" / "embeddings.npy")
    without = query_vectors_context(argparse.Namespace(**base), emb)     # historical behaviour
    assert without["gallery_vectors"]["path"] == "embeddings/graph/gat/x/embeddings.npy"
