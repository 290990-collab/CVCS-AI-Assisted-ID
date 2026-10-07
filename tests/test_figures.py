
"""
CPU smoke test for the report figures (`src/figures/`). Synthetic per-query files and figures in tmp.

Protects:
- the style always saves the three files (pdf, png, provenance), and provenance holds the files read and the figure numbers;
- the damage curve is paired (only queries present in all systems) and its AUC excludes f=0.0;
- the alpha figure stops if the mean recomputed from per-query files differs from the `fusion_select` json;
- the teaser rebuilds the ranking by putting the query back in place (`ret_rows` excludes it, `self_rr` says where it was)
  and picks the query by a declared rule.

Usage: python -m pytest tests/test_figures.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluation.perquery import gallery_sha1, write_npz
from src.evaluation.relevance import AXES
from src.evaluation.robustness_auc import AUC_FRACTIONS, fraction_path
from src.figures import (
    ablation_valid,
    alpha_valid,
    classes_valid,
    damage_kinds_valid,
    damage_test,
    style,
    teaser_valid,
)

NAMES = ["q0", "q1", "q2", "q3"]
CURVE = damage_test.CURVE_FRACTIONS          # (0.0, 0.25, 0.5, 0.75)


def _write_curve(prefix, strategy, values_by_frac, names=NAMES, split="valid"):
    """Per-query files of a synthetic config, one per fraction."""
    axes = list(AXES)
    for frac, vals in values_by_frac.items():
        q = len(names)
        meta = {
            "branch": "test", "run_tag": "sys", "mode": "partial",
            "partial_label": f"{strategy} f={frac}", "split": split,
            "exclude_self": True, "query_seed": 42,
            "gallery": {"n": 10, "sha1": gallery_sha1([f"g{i}" for i in range(10)]),
                        "source": "synthetic"},
            "axes": axes, "k_values": [10],
        }
        write_npz(fraction_path(prefix, frac, split, strategy), meta=meta, names=names,
                  qi=list(range(q)),
                  ndcg=np.full((len(axes), 1, q), np.nan, dtype=np.float32),
                  recall=np.full((len(axes), 1, q), np.nan, dtype=np.float32),
                  map_=np.full((len(axes), 1, q), np.nan, dtype=np.float32),
                  num_relevant=np.full((len(axes), q), -1, dtype=np.int32),
                  self_rr=np.asarray(vals, dtype=np.float32))


# --- style.py ---

def test_mean_ci_is_deterministic_and_brackets_the_mean():
    values = np.linspace(0.0, 1.0, 200)
    mean, lo, hi = style.mean_ci(values, b=500)
    assert mean == pytest.approx(values.mean())
    assert lo < mean < hi
    assert style.mean_ci(values, b=500) == (mean, lo, hi)      # same seed


def test_mean_ci_on_empty_is_nan():
    mean, lo, hi = style.mean_ci(np.zeros(0))
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_save_figure_writes_three_files_with_provenance(tmp_path):
    fig, ax = style.new_figure("column", height_in=1.5)
    ax.plot([0, 1], [0, 1])
    out = style.save_figure(fig, "smoke", sources=["a/b.npz"], notes=["auc 0.1234"],
                            out_dir=tmp_path)
    assert out["pdf"].exists() and out["png"].exists()
    text = out["sources"].read_text(encoding="utf-8")
    assert "a/b.npz" in text and "auc 0.1234" in text


def test_new_figure_uses_the_two_column_widths():
    fig, _ = style.new_figure("column")
    assert fig.get_size_inches()[0] == pytest.approx(style.COLUMN_WIDTH_IN)
    fig, _ = style.new_figure("text")
    assert fig.get_size_inches()[0] == pytest.approx(style.TEXT_WIDTH_IN)
    with pytest.raises(ValueError):
        style.new_figure("half")


# --- damage_test.py: four-system curve ---

def _two_systems(tmp_path, names_b=NAMES):
    """Two synthetic systems: `a` on all queries, `b` on `names_b`."""
    _write_curve(tmp_path / "a", "nowalls-random", {f: [1.0, 0.5, 0.25, 0.0] for f in CURVE})
    _write_curve(tmp_path / "b", "random",
                 {f: [0.5] * len(names_b) for f in CURVE}, names=names_b)
    return (
        damage_test.System("graph", str(tmp_path / "a"), "nowalls-random", "a", "a"),
        damage_test.System("baseline", str(tmp_path / "b"), "random", "b", "b"),
    )


def test_curve_keeps_only_the_queries_shared_by_all_systems(tmp_path, monkeypatch):
    monkeypatch.setattr(damage_test, "SYSTEMS", _two_systems(tmp_path, names_b=["q0", "q2"]))
    curves, names, sources = damage_test.load_systems(split="valid")
    assert names == ["q0", "q2"]                       # q1 and q3 missing in `b`
    assert curves["graph"][0.25].tolist() == [1.0, 0.25]
    assert len(sources) == 2 * len(CURVE)


def test_curve_auc_excludes_f0(tmp_path, monkeypatch):
    systems = _two_systems(tmp_path)
    monkeypatch.setattr(damage_test, "SYSTEMS", systems)
    curves, _, _ = damage_test.load_systems(split="valid")
    # f=0.0 is 1.0/0.5/0.25/0.0 like the others: it would not change the AUC if included, so check the mean by hand on an asymmetric case
    curves["graph"][0.0] = np.zeros(4)
    stats = damage_test.summarize(curves)
    assert stats["graph"]["auc"] == pytest.approx(np.mean([1.0, 0.5, 0.25, 0.0]))
    assert stats["graph"]["mean"][0.0] == pytest.approx(0.0)


def test_curve_fails_loudly_when_a_fraction_is_missing(tmp_path, monkeypatch):
    _write_curve(tmp_path / "a", "nowalls-random", {f: [1.0] * 4 for f in CURVE[:-1]})
    monkeypatch.setattr(damage_test, "SYSTEMS", (
        damage_test.System("graph", str(tmp_path / "a"), "nowalls-random", "a", "a"),))
    with pytest.raises(FileNotFoundError):
        damage_test.load_systems(split="valid")


# --- alpha_valid.py: curve over alpha ---

def _select_json(tmp_path, means, alpha_star=0.5, strategy="nowalls-random"):
    """Write a `fusion_select`-style json + the per-query files of its alphas."""
    fusion_dir = tmp_path / "fused"
    fusion_dir.mkdir(exist_ok=True)
    for alpha, value in means.items():
        _write_curve(fusion_dir / f"fusion_a{alpha:g}", strategy,
                     {f: [value] * 4 for f in (0.25, 0.5, 0.75)})
    path = tmp_path / "select.json"
    path.write_text(json.dumps({
        "alphas": sorted(means), "auc_means": {f"{a:g}": v for a, v in means.items()},
        "alpha_star": alpha_star, "split": "valid", "fusion_dir": str(fusion_dir),
        "oracle": {"oracle_mean": 0.55},
    }), encoding="utf-8")
    return alpha_valid.Pair("vision-graph", str(path), strategy, "p", "p")


def test_gain_is_measured_against_the_better_end(tmp_path):
    pair = _select_json(tmp_path, {0.0: 0.40, 0.5: 0.60, 1.0: 0.30})
    out = alpha_valid.load_pair(pair, with_ci=False, boot=200)
    assert out["best_end"] == pytest.approx(0.40)       # not the mean of the two
    assert out["gain"] == pytest.approx(0.20)
    assert out["oracle"] == pytest.approx(0.55)


def test_alpha_curve_stops_if_files_and_json_disagree(tmp_path):
    pair = _select_json(tmp_path, {0.0: 0.40, 0.5: 0.60, 1.0: 0.30})
    # per-query files say 0.60, json altered to 0.61: the figure would show a number not in the report
    path = tmp_path / "select.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["auc_means"]["0.5"] = 0.61
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="non coincide"):
        alpha_valid.load_pair(pair, with_ci=True, boot=200)


def test_alpha_curve_reads_the_bands_from_the_perquery_files(tmp_path):
    pair = _select_json(tmp_path, {0.0: 0.40, 0.5: 0.60, 1.0: 0.30})
    out = alpha_valid.load_pair(pair, with_ci=True, boot=200)
    assert out["mean"].tolist() == [0.40, 0.60, 0.30]
    # all queries equal: band degenerate, not absent
    assert out["lo"].tolist() == pytest.approx([0.40, 0.60, 0.30])
    assert len(out["sources"]) == 1 + 3 * 3


# --- teaser_valid.py: rebuilt ranking and query choice ---

def test_rank_of_self_inverts_the_reciprocal_rank():
    assert teaser_valid._rank_of_self(1.0) == 1
    assert teaser_valid._rank_of_self(0.2) == 5
    assert teaser_valid._rank_of_self(0.0) is None       # never returned in the top 100


def test_full_ranking_puts_the_query_back_where_it_was():
    rows = np.asarray([11, 12, 13, 14, -1])              # ranking without the query
    assert teaser_valid.full_ranking(rows, 1, 99, 3) == [99, 11, 12]
    assert teaser_valid.full_ranking(rows, 3, 99, 4) == [11, 12, 99, 13]
    # self_rr = 0: the plan did not return, the row stays that of the others only
    assert teaser_valid.full_ranking(rows, None, 99, 3) == [11, 12, 13]


def _systems(ranks):
    return {key: {"rank": list(values)} for key, values in ranks.items()}


def test_query_rule_takes_the_first_name_among_the_valid_ones():
    names = ["b", "a", "c"]
    systems = _systems({"fusion": [1, 1, 1], "vision": [1, 4, 2], "graph": [2, 7, 1]})
    index, rule = teaser_valid.choose_query(systems, names, forced=None)
    assert names[index] == "a"        # "b" already has vision first, "c" the graph
    assert "alfabetico" in rule


def test_query_rule_fails_when_nothing_qualifies_and_forced_name_is_checked():
    names = ["a", "b"]
    systems = _systems({"fusion": [2, 3], "vision": [1, 1], "graph": [1, 1]})
    with pytest.raises(ValueError, match="nessuna query"):
        teaser_valid.choose_query(systems, names, forced=None)
    with pytest.raises(ValueError, match="non fra le"):
        teaser_valid.choose_query(systems, names, forced="zz")
    index, rule = teaser_valid.choose_query(systems, names, forced="b")
    assert index == 1 and "imposta" in rule


# --- damage_kinds_valid.py: the three damages on the same queries ---

def test_damage_panel_pairs_the_strategies_and_averages_the_fractions(tmp_path):
    prefix = tmp_path / "cfg"
    _write_curve(prefix, "nowalls-random", {f: [1.0, 0.5, 0.0, 0.5] for f in AUC_FRACTIONS})
    _write_curve(prefix, "crop", {0.25: [1.0] * 2, 0.5: [0.5] * 2, 0.75: [0.0] * 2},
                 names=["q0", "q2"])
    curves, names, sources = damage_kinds_valid.load_panel(
        prefix, ("nowalls-random", "crop"), split="valid")
    assert names == ["q0", "q2"]                     # only queries present in both
    assert curves["crop"]["auc"] == pytest.approx(np.mean([1.0, 0.5, 0.0]))
    assert curves["nowalls-random"]["mean"][0.25] == pytest.approx(0.5)   # (1.0 + 0.0) / 2
    assert len(sources) == 2 * len(AUC_FRACTIONS)


# --- classes_valid.py: equivalence class size ---

def _write_full(path, num_relevant, names=NAMES):
    """Full-mode per-query files with given relevant counts."""
    axes = list(AXES)
    q = len(names)
    meta = {"branch": "test", "run_tag": "sys", "mode": "full", "split": "valid",
            "exclude_self": True, "query_seed": 42,
            "gallery": {"n": 100, "sha1": gallery_sha1(["g"]), "source": "synthetic"},
            "axes": axes, "k_values": [10]}
    write_npz(path, meta=meta, names=names, qi=list(range(q)),
              ndcg=np.full((len(axes), 1, q), np.nan, dtype=np.float32),
              recall=np.full((len(axes), 1, q), np.nan, dtype=np.float32),
              map_=np.full((len(axes), 1, q), np.nan, dtype=np.float32),
              num_relevant=np.asarray(num_relevant, dtype=np.int32))


def test_class_sizes_come_from_num_relevant(tmp_path):
    counts = [[500, 400, 300, 200], [3, 0, 1, 0], [-1] * 4]      # comp, topo, geom
    _write_full(tmp_path / "a.npz", counts)
    _write_full(tmp_path / "b.npz", counts)
    out = classes_valid.load_counts(tmp_path / "a.npz", check=tmp_path / "b.npz")
    assert out["counts"]["topology"].tolist() == [3, 0, 1, 0]
    assert int((out["counts"]["topology"] == 0).sum()) == 2       # singletons
    assert out["n_queries"] == 4 and out["gallery"] == 100


def test_class_sizes_refuse_two_systems_that_disagree(tmp_path):
    _write_full(tmp_path / "a.npz", [[5, 5, 5, 5], [1, 1, 1, 1], [-1] * 4])
    _write_full(tmp_path / "b.npz", [[5, 5, 5, 5], [1, 1, 2, 1], [-1] * 4])
    with pytest.raises(ValueError, match="non può dipendere dal sistema"):
        classes_valid.load_counts(tmp_path / "a.npz", check=tmp_path / "b.npz")


# --- ablation_valid.py: grouping of configurations ---

def test_ablation_groups_by_encoder_and_keeps_heads_apart():
    values = {
        "vision_alpha_gem_raw": 0.10, "vision_alpha_gem_whiten-train": 0.20,
        "vision_beta_mean_raw": 0.40, "vision_beta_mean_whiten-train": 0.60,
        "vision_beta_mean_head-conv": 0.99,
    }
    order, frozen, heads = ablation_valid.group(values)
    assert order == ["alpha", "beta"]                      # sorted by median
    assert [p for p, _ in heads["beta"]] == ["vision_beta_mean_head-conv"]
    assert all(not p.split("_", 3)[3].startswith("head") for e in frozen for p, _ in frozen[e])


def test_ablation_prefix_split_and_bad_prefix():
    assert ablation_valid.split_prefix("vision_pespatial_gem_whiten-train") == \
        ("pespatial", "gem", "whiten-train")
    assert ablation_valid.split_prefix("vision_tipsv2_gem448_raw")[1] == "gem448"
    with pytest.raises(ValueError, match="prefisso inatteso"):
        ablation_valid.split_prefix("vision_solo_due")
