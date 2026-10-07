"""Exploratory fusion curve: task list and paths. CPU, no heavy data."""

from src.evaluation import fusion_curve as fc
from src.evaluation import late_fusion as lf
from src.evaluation import graph_config_select as rs
from src.graph import final_graph_configs as rc


def test_tasks_cover_37_configs_twice():
    t = fc.tasks()
    assert len(fc.configs()) == 37 and len(t) == 74
    assert t[:37] == [("frozen", e, c) for e, c in fc.configs()]
    assert t[37:] == [("head", e, c) for e, c in fc.configs()]
    assert len({fc.paths(*x)["OUT"] for x in t}) == 74


def test_outputs_only_under_curve_root():
    for x in fc.tasks():
        assert fc.paths(*x)["OUT"].startswith(str(rc.root() / "curve") + "/")


def test_w_frozen_uses_the_main_fusion_inputs():
    p, m = fc.paths("frozen", "gat", "comb"), rs.fusion_paths(42, "valid")
    assert (p["GRAPH_QVEC"], p["VISION_QVEC"], p["VISION_PQ"]) == (m["GRAPH_QVEC"], m["VISION_QVEC"], m["VISION_PQ"])


def test_argv_reduced_grid_partial_only():
    a = lf.parse_args(fc.lf_argv(fc.paths("head", "gcn", "ref")))
    assert [float(x) for x in a.alphas] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert list(a.modes) == ["partial"] and [float(f) for f in a.fractions] == [0.25, 0.5, 0.75]


# --- reduced curve on the test ---

from src.evaluation import fusion_curve_test as ft  # noqa: E402
from src.evaluation import fusion_select as fs  # noqa: E402
import numpy as np  # noqa: E402


def test_test_curve_tasks_and_weights_preregistered():
    t = ft.tasks()
    assert len(t) == 10 and [x[0] for x in t] == ["frozen"] * 5 + ["head"] * 5
    assert [(e, c) for _, e, c in t[:5]] == [("gcn", "ref"), ("gat", "t05"), ("gat", "ref"), ("gat", "t01"), ("gat", "comb")]
    assert [ft.weight(*x) for x in t] == [0.8, 0.6, 0.6, 0.4, 0.4] * 2
    for x in t:
        p = ft.paths(*x)
        assert p["OUT"].startswith(str(rc.root() / "curve_test") + "/") and p["OUT"].endswith("/fusion_test")
        assert "/test/" in p["GRAPH_QVEC"] and "/test/" in p["VISION_QVEC"]


def test_test_curve_argv_fixed_weight_only():
    a = lf.parse_args(ft.lf_argv(ft.paths("frozen", "gat", "t05"), 0.6))
    assert a.split == "test" and [float(x) for x in a.alphas] == [0.0, 0.6, 1.0]
    assert list(a.modes) == ["partial"] and [float(f) for f in a.fractions] == [0.25, 0.5, 0.75]


def test_test_curve_w_frozen_uses_the_main_test_inputs():
    p, m = ft.paths("frozen", "gat", "comb"), rs.fusion_paths(42, "test")
    assert (p["GRAPH_QVEC"], p["VISION_QVEC"], p["VISION_PQ"]) == (m["GRAPH_QVEC"], m["VISION_QVEC"], m["VISION_PQ"])


def test_prediction_rule():
    rng = np.random.default_rng(0)
    up = 0.05 + 0.01 * rng.standard_normal(500)
    assert ft.prediction(up, +1, "x")["esito"] == "confermata"
    assert ft.prediction(-up, +1, "x")["esito"] == "contraria"
    assert ft.prediction(-up, -1, "x")["esito"] == "confermata"
    assert ft.prediction(rng.standard_normal(500), +1, "x")["esito"] == "non confermata"


def test_predictions_shape_on_synthetic_gains():
    """Gains built to follow the inverted U; head deltas as predicted."""
    rng = np.random.default_rng(1)
    names = np.array([f"q{i}" for i in range(400)])
    base_f = {("gcn", "ref"): 0.11, ("gat", "t05"): 0.21, ("gat", "ref"): 0.15, ("gat", "t01"): 0.06, ("gat", "comb"): 0.016}
    head_d = {("gcn", "ref"): -0.014, ("gat", "t05"): -0.03, ("gat", "ref"): 0.076, ("gat", "t01"): 0.067, ("gat", "comb"): 0.014}
    rows = {}
    for c in ft.CONFIGS:
        gf = base_f[c] + 0.002 * rng.standard_normal(400)
        rows[("frozen", c)] = {"_names": names, "_g": gf}
        rows[("head", c)] = {"_names": names[::-1], "_g": (gf + head_d[c])[::-1]}   # order differs: aligned by name
    p = ft.predictions(rows)
    assert all(x["esito"] == "confermata" for k in ("E1_falling_frozen", "E2_rising_frozen", "E3_head_vs_frozen", "E4")
               for x in p[k])
    assert p["shape"] == "curva a U rovesciata confermata sul test"
    rows[("frozen", ("gat", "t01"))]["_g"] = rows[("frozen", ("gat", "comb"))]["_g"] - 0.01
    p = ft.predictions(rows)
    assert p["shape"].startswith("forma non confermata") and "g_F(gat/t05) > g_F(gcn/ref)" in p["shape"]


def test_test_curve_endpoint_boundary_diagnosis_only_after_plain_fail(tmp_path, monkeypatch):
    import pytest
    import json
    from src.evaluation import fusion_curve as fc
    from src.evaluation import fusion_curve_test as ft
    from src.evaluation import fusion_select as fs
    monkeypatch.setattr(fc, "_branch_check", lambda fused, own: {0.75: {}})
    monkeypatch.setattr(fs, "c3_pass", lambda stats: False)
    plain, bnd = tmp_path / "nt.json", tmp_path / "nt_confine.json"
    assert not ft._endpoint_check(None, None, plain, bnd)["ok"]          # no diagnosis
    plain.write_text(json.dumps({"all_near_ties": False}))
    bnd.write_text(json.dumps({"all_near_ties": True, "boundary": True}))
    r = ft._endpoint_check(None, None, plain, bnd)
    assert r["ok"] and r["near_ties_boundary"]["all_near_ties"]
    bnd.write_text(json.dumps({"all_near_ties": True}))                    # not a --boundary run
    with pytest.raises(ValueError):
        ft._endpoint_check(None, None, plain, bnd)
    bnd.write_text(json.dumps({"all_near_ties": False, "boundary": True}))
    assert not ft._endpoint_check(None, None, plain, bnd)["ok"]
    assert "confine" in ft.paths("frozen", "gat", "t05")["NEAR_TIES_BOUNDARY"][0.0]
