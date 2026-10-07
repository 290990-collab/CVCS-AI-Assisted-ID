"""RESET GRAPHS test, items B and C: outcome rules and paths. CPU, no heavy data."""

import pytest

from src.evaluation import fusion_head as fh
from src.evaluation import ensemble_control_test as tc


def ci(lo, hi):
    return {"mean": (lo + hi) / 2, "ci_lo": lo, "ci_hi": hi}


def test_outcome_rule_c():
    assert tc.outcome(ci(0.01, 0.02), ci(0.001, 0.02)).startswith("confermato")
    assert tc.outcome(ci(-0.02, -0.01), ci(-0.03, -0.001)).startswith("smentito")
    o = tc.outcome(ci(0.01, 0.02), ci(-0.001, 0.02))
    assert o.startswith("non distinguibile") and "sage/comb" in o and "seconda copia" not in o
    o = tc.outcome(ci(-0.01, 0.02), ci(-0.03, -0.01))
    assert "seconda copia di W" in o and "sage/comb" in o


def test_verdict_b():
    assert "confermato" in fh.test_verdict_b(ci(0.001, 0.01)) and "non" not in fh.test_verdict_b(ci(0.001, 0.01))
    assert "smentito" in fh.test_verdict_b(ci(-0.01, -0.001))
    assert "non confermato" in fh.test_verdict_b(ci(-0.001, 0.01))


def test_paths_controls_on_the_test():
    p = tc.paths()
    assert p["REPLICA_QVEC"].startswith("results/final_pipeline/queryvec/test/ctrl_p42/")
    assert p["REPLICA_QVEC"].endswith("graph_gat_rg_comb_s100042")
    assert p["CROSSENC_QVEC"] == "results/final_pipeline/queryvec/test/s42/graph_sage_rg_comb_s42"
    assert p["W42_QVEC"] == "results/final_pipeline/queryvec/test/s42/graph_gat_rg_comb_s42"
    for c in ("REPLICA", "CROSSENC"):
        assert p[f"{c}_DIR"].startswith("results/final_pipeline/controls_test/") and p[f"{c}_DIR"].endswith("fusion_test")
        assert p[f"{c}_SELECT_VALID"].startswith("results/final_pipeline/controls/")


def test_head_paths_test():
    p = fh.head_paths(42, "test")
    assert p["FUSION_DIR"].endswith("fusion_head/s42/fusion_test")
    assert p["SELECT_VALID_JSON"].endswith("fusion_head/s42/select_valid.json")
    assert p["MAIN_SELECT_JSON"].endswith("fusion/s42/select_test.json")


def test_weights_preregistered_match_valid_select():
    import json
    from pathlib import Path
    p = tc.paths()
    for ctrl, beta in tc.BETAS.items():
        assert float(json.loads(Path(p[f"{ctrl.upper()}_SELECT_VALID"]).read_text())["alpha_star"]) == beta
    assert float(json.loads(Path(fh.head_paths(42, "test")["SELECT_VALID_JSON"]).read_text())["alpha_star"]) == tc.ALPHA_HEAD


def test_near_ties_waiver_only_if_all_near_ties():
    from src.evaluation import near_ties as nt
    d = {"all_near_ties": False}
    with pytest.raises(ValueError):
        nt.waiver(d)
    d = {"all_near_ties": True, "alpha_endpoint": 1.0, "branch": "vision-head", "n_differing_total": 3,
         "max_gap": 1e-6, "method": "m", "per_fraction": {"0.25": {"n_differing": 3, "rows": []}}}
    w = nt.waiver(d)
    assert w["accepted"] is True and "rows" not in w["diagnosis"]["per_fraction"]["0.25"]
