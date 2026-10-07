
"""
CPU tests of the final graph pipeline: configuration table and flags (src/graph/final_graph_configs.py)
and selection rules (src/evaluation/graph_config_select.py). Synthetic data only.

Usage: python -m pytest tests/test_final_pipeline.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from src.graph import final_graph_configs as rc
from src.evaluation import graph_config_select as rs

ROOT_DIR = Path(__file__).resolve().parents[1]

# Configuration table, spelled independently of final_graph_configs: training flags as 03_train_gnn.sh appends them after the YAML.
PLAN_TRAIN_FLAGS = {
    "ref": [], "t01": ["--temperature", "0.1"], "t02": ["--temperature", "0.2"],
    "t05": ["--temperature", "0.5"], "noskip": ["--no-raw-skip"],
    "nosym": ["--flip-prob", "0", "--rot-prob", "0"], "lost": ["--lost-marker"],
    "l3": ["--num-layers", "3"], "h256": ["--hidden-dim", "256"], "pmm": ["--pooling", "mean_max"],
    "amax": ["--aggr", "max"], "aadd": ["--aggr", "add"], "hd1": ["--heads", "1"], "hd8": ["--heads", "8"],
}
ARCH_KEYS = ("encoder", "variant", "hidden_dim", "out_dim", "num_layers", "pooling", "dropout",
             "raw_skip", "heads", "attn_dropout", "aggr", "normalize", "drop_self_loops", "lost_marker")


# --- final_graph_configs: the table ---

def test_stage1_has_34_configurations():
    n = {enc: len(rc.stage1_configs(enc)) for enc in rc.ENCODERS}
    assert n == {"gcn": 10, "graph_sage": 12, "gat": 12}
    assert sum(n.values()) * len(rc.SEEDS) == 136


def test_every_change_is_one_factor_and_matches_the_plan():
    assert set(rc.OVERRIDES) == set(PLAN_TRAIN_FLAGS)
    for cfg, ov in rc.OVERRIDES.items():
        assert len(ov) <= 1 or cfg == "nosym"
    assert rc.FACTOR["t01"] == rc.FACTOR["t02"] == rc.FACTOR["t05"] == "temperature"
    assert rc.FACTOR["amax"] == rc.FACTOR["aadd"] == "aggr"
    assert rc.FACTOR["hd1"] == rc.FACTOR["hd8"] == "heads"
    assert len({rc.FACTOR[c] for c in ("noskip", "nosym", "lost", "l3", "h256", "pmm", "t01")}) == 7


def test_encoder_specific_changes_are_refused_elsewhere():
    with pytest.raises(ValueError):
        rc.config_overrides("gcn", "amax")
    with pytest.raises(ValueError):
        rc.config_overrides("graph_sage", "hd8")
    with pytest.raises(ValueError):
        rc.run_paths("gat", "aadd", 42)


def _bridge_flags(encoder: str) -> list[str]:
    """YAML->flag bridge of scripts/graph/_common.sh (the reference)."""
    out = subprocess.run(
        ["bash", "-c", f"source scripts/graph/_common.sh; train_flags_from_yaml configs/graph_models/{encoder}.yaml"],
        cwd=ROOT_DIR, capture_output=True, text=True, check=True)
    return out.stdout.split()


def _parse(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    return vars(module.parse_args())


@pytest.mark.parametrize("encoder", list(rc.ENCODERS))
def test_train_flags_equal_bridge_plus_recipe_plus_plan(encoder, monkeypatch):
    """Our rendering equals what 03_train_gnn.sh builds (YAML bridge + recipe + table)."""
    train_gnn = pytest.importorskip("src.graph.training.train_gnn")
    bridge = _bridge_flags(encoder)
    for cfg in rc.stage1_configs(encoder):
        for seed in (42, 300042):
            legacy = bridge + ["--variant", rc.variant(cfg, seed),
                               "--pair-mode", "asym_partial", "--selection-probe", "partial",
                               "--epochs", "600", "--patience", "0", "--shadow-patience", "10",
                               "--shadow-epochs", "150", "--seed", str(seed)] + PLAN_TRAIN_FLAGS[cfg]
            ours = rc.train_flags(encoder, cfg, seed)
            assert _parse(train_gnn, ours, monkeypatch) == _parse(train_gnn, legacy, monkeypatch), (encoder, cfg)


@pytest.mark.parametrize("encoder", list(rc.ENCODERS))
def test_eval_architecture_equals_training(encoder, monkeypatch):
    """The checkpoint is reloaded with the architecture it was trained with."""
    train_gnn = pytest.importorskip("src.graph.training.train_gnn")
    graph_evaluate = pytest.importorskip("src.graph.evaluation.graph_evaluate")
    for cfg in rc.stage1_configs(encoder):
        for split in ("valid", "test", "ctrl", "ctrltest"):
            tr = _parse(train_gnn, rc.train_flags(encoder, cfg, 100042), monkeypatch)
            ev = _parse(graph_evaluate, rc.eval_flags(encoder, cfg, 100042, split), monkeypatch)
            for k in ARCH_KEYS:
                assert ev[k] == tr[k], (encoder, cfg, k)
            assert ev["split"] == rc.eval_split(split)
            assert ev["seed"] == 42 and ev["num_queries"] == 2000      # query sample, not the training seed
            assert ev["gallery_names"] == "results/shared_gallery.json"
            assert ev["save_dir"].endswith(f"/{rc.encoder_key(encoder)}/{rc.variant(cfg, 100042)}")


def test_paths_and_damage_seed():
    p = rc.run_paths("graph_sage", "t02", 200042, "valid")
    assert p["DEST"] == "embeddings/graph/sage/rg_t02_s200042"
    assert p["TAG"] == "graph_sage_rg_t02_s200042"
    assert p["PQ_DIR"] == "results/final_pipeline/perquery/valid/s200042"
    assert p["PARTIAL_SEED"] == "200042"
    c = rc.run_paths("gat", "ref", 100042, "ctrl")
    assert c["PARTIAL_SEED"] == "42" and c["EVAL_SPLIT"] == "valid"
    assert c["QV_DIR"] == "results/final_pipeline/queryvec/valid/ctrl_p42"
    # second copy of W on the TEST, damage seed 42, its own folder
    t = rc.run_paths("gat", "comb", 100042, "ctrltest")
    assert t["PARTIAL_SEED"] == "42" and t["EVAL_SPLIT"] == "test"
    assert t["QV_DIR"] == "results/final_pipeline/queryvec/test/ctrl_p42"
    assert t["PQ_DIR"] == "results/final_pipeline/perquery/test/ctrl_p42"
    assert rc.run_paths("gat", "comb", 100042, "test")["QV_DIR"] == "results/final_pipeline/queryvec/test/s100042"
    s = rc.run_paths("gcn", "ref", 42, "valid", smoke=True)
    assert s["DEST"] == "embeddings/graph/gcn/rgsmoke_ref_s42" and s["EPOCHS"] == "2"
    assert s["ROOT"] == "results/final_pipeline_smoke"


def test_cli_refuses_unregistered_seed():
    with pytest.raises(SystemExit):
        rc.main(["paths", "--encoder", "gcn", "--cfg", "ref", "--seed", "7", "--split", "valid"])


def test_comb_reads_stage2(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        rc.config_overrides("gat", rc.COMB)
    st2 = tmp_path / "selection" / "stage2.json"
    st2.parent.mkdir(parents=True)
    st2.write_text(json.dumps({"combs": {"gat": {"cfgs": ["t02", "lost"],
                                                  "overrides": {"temperature": 0.2, "lost_marker": True}},
                                         "sage": None}}))
    assert rc.config_overrides("gat", rc.COMB) == {"temperature": 0.2, "lost_marker": True}
    assert "--lost-marker" in rc.train_flags("gat", rc.COMB, 42)
    with pytest.raises(ValueError):
        rc.config_overrides("graph_sage", rc.COMB)


# --- graph_config_select: rules on synthetic runs ---

NAMES = np.asarray([f"q{i:04d}" for i in range(400)])
SEEDS = rc.SEEDS


def make_runs(key, cfg, level, topo=0.6, n_params=1000, per_seed=None, noise=0.05, rng_seed=0):
    """seed -> RunData with AUC ~ level (+ per-seed offset) and topology ~ topo.

    Query difficulty (`base`) is shared by all configurations; noise has mean exactly 0 per seed
    (no clipping: values stay in [0, 1])."""
    base = np.random.default_rng(0).uniform(-0.2, 0.2, size=len(NAMES))
    rng = np.random.default_rng(zlib.crc32(f"{key}/{cfg}/{rng_seed}".encode()))

    def centred():
        x = rng.normal(0, noise, len(NAMES))
        return x - x.mean()

    runs = {}
    for i, s in enumerate(SEEDS):
        off = (per_seed or [0.0] * 4)[i]
        auc = level + off + base + centred()
        t = topo + base + centred()
        runs[s] = rs.RunData(key=key, cfg=cfg, seed=s, names=NAMES, auc=auc,
                             per_fraction={0.25: 0.8, 0.5: 0.4, 0.75: 0.1}, mrr_f0=0.97,
                             full_names=NAMES, full={"composition": t, "topology": t, "geometry": t},
                             best_epoch=599, epochs_run=600, n_params=n_params)
    return runs


def test_clearly_better_needs_ci_and_all_replicas():
    a, b = make_runs("sage", "ref", 0.50), make_runs("gat", "ref", 0.45)
    cmp = rs.paired(a, b)
    assert cmp["n"] == len(NAMES) and cmp["ci_lo"] > 0
    assert rs.clearly_better(cmp)
    # one replica worse -> tie even if the CI on the mean is above 0
    c = make_runs("sage", "t02", 0.50, per_seed=[0.08, 0.08, 0.08, -0.09])
    cmp2 = rs.paired(c, make_runs("gat", "ref", 0.48))
    assert cmp2["delta"] > 0 and min(cmp2["per_seed"].values()) < 0
    assert not rs.clearly_better(cmp2)
    # same level -> CI contains 0
    assert not rs.clearly_better(rs.paired(make_runs("gcn", "ref", 0.5), make_runs("gcn", "t01", 0.5)))


def test_paired_drops_nan_and_aligns_by_name():
    a, b = make_runs("gcn", "ref", 0.5), make_runs("gcn", "t01", 0.4)
    b[42].full["topology"] = b[42].full["topology"].copy()
    b[42].full["topology"][:10] = np.nan
    cmp = rs.paired(a, b, which="full", axis="topology")
    assert cmp["n"] == len(NAMES) - 10


def _summaries(runs):
    return {kc: rs.config_summary(r) for kc, r in runs.items()}


def test_netta_changes_best_per_factor_and_comb():
    runs = {("sage", "ref"): make_runs("sage", "ref", 0.45),
            ("sage", "t01"): make_runs("sage", "t01", 0.50),
            ("sage", "t02"): make_runs("sage", "t02", 0.55),
            ("sage", "noskip"): make_runs("sage", "noskip", 0.52),
            ("sage", "nosym"): make_runs("sage", "nosym", 0.45),
            ("sage", "amax"): make_runs("sage", "amax", 0.40)}
    cfgs = ["ref", "t01", "t02", "noskip", "nosym", "amax"]
    res = rs.netta_changes("sage", cfgs, runs, _summaries(runs))
    assert set(res["netta"]) == {"t01", "t02", "noskip"}
    assert res["kept"] == ["t02", "noskip"]
    assert res["overrides"] == {"temperature": 0.2, "raw_skip": False}


def test_netta_single_change_gives_no_comb():
    runs = {("gcn", "ref"): make_runs("gcn", "ref", 0.45),
            ("gcn", "t01"): make_runs("gcn", "t01", 0.50),
            ("gcn", "t02"): make_runs("gcn", "t02", 0.53)}
    res = rs.netta_changes("gcn", ["ref", "t01", "t02"], runs, _summaries(runs))
    assert res["kept"] == ["t02"] and res["overrides"] is None


def test_winner_without_ties():
    runs = {("sage", "ref"): make_runs("sage", "ref", 0.60),
            ("gat", "ref"): make_runs("gat", "ref", 0.45),
            ("gcn", "ref"): make_runs("gcn", "ref", 0.40)}
    ch = rs.choose_winner(runs, _summaries(runs))
    assert ch["winner"] == ("sage", "ref") and ch["ties"] == [] and ch["tiebreak"] is None
    assert ch["E"] == ("gat", "ref")


def test_tie_broken_by_topology():
    runs = {("sage", "ref"): make_runs("sage", "ref", 0.501, topo=0.55),
            ("gat", "t02"): make_runs("gat", "t02", 0.500, topo=0.70),
            ("gcn", "ref"): make_runs("gcn", "ref", 0.30)}
    ch = rs.choose_winner(runs, _summaries(runs))
    assert ch["top_by_score"] == ("sage", "ref")
    assert ch["ties"] == [("gat", "t02")]
    assert ch["winner"] == ("gat", "t02")
    assert ch["E"] == ("sage", "ref")              # best of another encoder than W


def test_tie_on_topology_broken_by_parameters_then_score():
    runs = {("sage", "h256"): make_runs("sage", "h256", 0.501, topo=0.60, n_params=5000),
            ("sage", "ref"): make_runs("sage", "ref", 0.500, topo=0.60, n_params=1000),
            ("gat", "ref"): make_runs("gat", "ref", 0.30)}
    ch = rs.choose_winner(runs, _summaries(runs))
    assert ch["winner"] == ("sage", "ref")
    assert ch["tiebreak"]["fewest_params"] == ["sage/ref"]
    runs[("sage", "ref")] = make_runs("sage", "ref", 0.500, topo=0.60, n_params=5000)
    ch2 = rs.choose_winner(runs, _summaries(runs))
    assert ch2["winner"] == ("sage", "h256")       # same parameters -> higher score


def _check_report(c3="PASS", mrr=(0.97, 0.985, 0.95)):
    return {"C1": {"status": "PASS"}, "C2": {"status": "PASS"}, "C3": {"status": c3},
            "C5": {"status": "FAIL", "per_alpha": {f"{a:g}": {"mrr": m} for a, m in zip((0, 0.5, 1), mrr)}}}


def test_fusion_check_lower_bound_only():
    assert rs.fusion_check_verdict(_check_report())["pass"]            # 0.985 > 0.98 is fine
    assert rs.fusion_check_verdict(_check_report(mrr=(0.97, 0.89, 0.95)))["failed"] == ["C5"]
    assert rs.fusion_check_verdict(_check_report(c3="FAIL"))["failed"] == ["C3"]
    v = rs.fusion_check_verdict(_check_report(c3="FAIL"), {"accepted": True})
    assert v["pass"] and v["c3_waived"]


def test_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "ROOT", str(tmp_path))
    assert rs.gate("test", "graph_sage", "ref", 42)                    # no final.json
    sel = tmp_path / "selection"
    sel.mkdir()
    (sel / "final.json").write_text(json.dumps({"W": {"encoder": "graph_sage", "cfg": "ref"}}))
    reasons = rs.gate("test", "graph_sage", "ref", 42)
    assert any("fusione valid" in r for r in reasons) and any("TEST_PREREGISTERED" in r for r in reasons)
    assert any("non e' il vincitore" in r for r in rs.gate("test", "gat", "lost", 42))   # not in TEST_EXTRA
    (tmp_path / "fusion" / "s42").mkdir(parents=True)
    (tmp_path / "fusion" / "s42" / "select_valid.json").write_text("{}")
    for c in rs.CONTROLS:
        (tmp_path / "controls" / c).mkdir(parents=True)
        (tmp_path / "controls" / c / "complementarity_valid.json").write_text("{}")
    (tmp_path / "TEST_PREREGISTERED").write_text("ok")
    assert rs.gate("test", "graph_sage", "ref", 42) == []
    # extra configs go to the test, seed 42 only
    assert rs.gate("test", "gat", "t05", 42) == []
    assert any("solo con il seed" in r for r in rs.gate("test", "gat", "t05", 100042))
    assert any("non e' il vincitore" in r for r in rs.gate("test", "gat", "lost", 42))
    assert any("non e' il vincitore" in r for r in rs.gate("ctrl", "gat", "t05", 100042))
    assert rs.gate("ctrl", "graph_sage", "ref", 42)                    # wrong replica
    assert any("100042" in r for r in rs.gate("ctrl", "graph_sage", "ref", 100042))
    (tmp_path / "fusion" / "s100042").mkdir()
    (tmp_path / "fusion" / "s100042" / "select_valid.json").write_text("{}")
    # fusion with the vision head reads the same embeddings.npy: it must come first
    assert any("con la head" in r for r in rs.gate("ctrl", "graph_sage", "ref", 100042))
    (tmp_path / "fusion_head" / "s100042").mkdir(parents=True)
    (tmp_path / "fusion_head" / "s100042" / "head_fusion_valid.json").write_text("{}")
    assert rs.gate("ctrl", "graph_sage", "ref", 100042) == []
    # ctrltest: only W s100042, only after the checks of its two test fusions
    assert any("non e' il vincitore" in r for r in rs.gate("ctrltest", "gat", "t05", 100042))
    assert any("ctrltest solo" in r for r in rs.gate("ctrltest", "graph_sage", "ref", 42))
    reasons = rs.gate("ctrltest", "graph_sage", "ref", 100042)
    assert any("fusione principale" in r for r in reasons) and any("con la head" in r for r in reasons)
    (tmp_path / "fusion" / "s100042" / "check_test_reset.json").write_text("{}")
    assert [r for r in rs.gate("ctrltest", "graph_sage", "ref", 100042) if "con la head" in r]
    (tmp_path / "fusion_head" / "s100042" / "check_test_reset.json").write_text("{}")
    assert rs.gate("ctrltest", "graph_sage", "ref", 100042) == []
    (tmp_path / "TEST_PREREGISTERED").unlink()
    assert any("TEST_PREREGISTERED" in r for r in rs.gate("ctrltest", "graph_sage", "ref", 100042))


def test_write_once(tmp_path):
    p = tmp_path / "x.json"
    rs._write_once(p, {"a": 1})
    with pytest.raises(SystemExit):
        rs._write_once(p, {"a": 2})
    assert json.loads(p.read_text())["a"] == 1


def test_shell_scripts_parse():
    for sh in sorted((ROOT_DIR / "scripts" / "final_pipeline").glob("*.sh")):
        subprocess.run(["bash", "-n", str(sh)], check=True)
