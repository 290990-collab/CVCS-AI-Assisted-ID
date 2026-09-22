# tests/test_robustness_auc.py

"""
Smoke test CPU per l'AUC di robustezza del criterio A.5
(`src/evaluation/robustness_auc.py`). File per-query sintetici, niente dataset.

Esecuzione: python -m pytest tests/test_robustness_auc.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.perquery import gallery_sha1, write_npz
from src.evaluation.relevance import AXES
from src.evaluation.robustness_auc import (
    AUC_FRACTIONS,
    compare_auc,
    discover_prefixes,
    fraction_path,
    load_auc,
)

NAMES = ["q0", "q1", "q2", "q3"]


def _write(prefix, frac, self_rr, names=NAMES, meta_overrides=None, split="valid"):
    q = len(names)
    axes = list(AXES)
    shape = (len(axes), 1, q)
    meta = {
        "branch": "test", "run_tag": "sys", "mode": "partial",
        "partial_label": f"random f={frac}", "split": split,
        "exclude_self": True, "query_seed": 42,
        "gallery": {"n": 10, "sha1": gallery_sha1([f"g{i}" for i in range(10)]),
                    "source": "synthetic"},
        "axes": axes, "k_values": [10],
    }
    if meta_overrides:
        meta.update(meta_overrides)
    write_npz(fraction_path(prefix, frac, split), meta=meta, names=names,
              qi=list(range(q)),
              ndcg=np.full(shape, np.nan, dtype=np.float32),
              recall=np.full(shape, np.nan, dtype=np.float32),
              map_=np.full(shape, np.nan, dtype=np.float32),
              num_relevant=np.full((len(axes), q), -1, dtype=np.int32),
              self_rr=np.asarray(self_rr, dtype=np.float32))


def _write_config(prefix, values_by_frac, **kw):
    for frac, vals in values_by_frac.items():
        _write(prefix, frac, vals, **kw)


def test_file_name_matches_branch_convention(tmp_path):
    # I due rami scrivono `..._partial-random-f0.25_valid.npz` (status.md § 28).
    p = fraction_path(tmp_path / "graph_gcn_tau02", 0.25, "valid")
    assert p.name == "graph_gcn_tau02_partial-random-f0.25_valid.npz"
    assert fraction_path(tmp_path / "x", 0.5, "valid").name.endswith("-f0.5_valid.npz")


def test_auc_is_per_query_mean_over_fractions_excluding_f0(tmp_path):
    pre = tmp_path / "cfg"
    vals = {0.25: [1.0, 1.0, 0.5, 0.0], 0.5: [1.0, 0.5, 0.5, 0.0],
            0.75: [0.4, 0.0, 0.5, 0.2]}
    _write_config(pre, vals)
    _write(pre, 0.0, [0.0, 0.0, 0.0, 0.0])      # f=0.0 non deve entrare
    d = load_auc(pre)
    expected = np.mean([vals[f] for f in AUC_FRACTIONS], axis=0)
    np.testing.assert_allclose(d.auc, expected, rtol=1e-6)
    assert d.mean == pytest.approx(float(expected.mean()), rel=1e-6)


def test_query_missing_in_one_fraction_is_dropped(tmp_path):
    pre = tmp_path / "cfg"
    _write(pre, 0.25, [1.0, 1.0, 1.0, 1.0])
    _write(pre, 0.5, [1.0, np.nan, 1.0, 1.0])          # q1 saltata a f=0.5
    _write(pre, 0.75, [1.0, 1.0, 1.0], names=["q0", "q1", "q2"])   # q3 assente
    d = load_auc(pre)
    assert list(d.names) == ["q0", "q2"]
    assert d.n_ref == 4


def test_missing_fraction_raises(tmp_path):
    pre = tmp_path / "cfg"
    _write(pre, 0.25, [1.0] * 4)
    _write(pre, 0.5, [1.0] * 4)
    with pytest.raises(FileNotFoundError):
        load_auc(pre)


def test_fractions_with_different_gallery_are_rejected(tmp_path):
    pre = tmp_path / "cfg"
    _write(pre, 0.25, [1.0] * 4)
    _write(pre, 0.5, [1.0] * 4)
    _write(pre, 0.75, [1.0] * 4, meta_overrides={"gallery": {"n": 9, "sha1": "x"}})
    with pytest.raises(ValueError, match="gallery"):
        load_auc(pre)


def test_compare_hand_computed_delta_and_self_is_zero(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_config(a, {f: [1.0, 1.0, 1.0, 1.0] for f in AUC_FRACTIONS})
    _write_config(b, {f: [0.5, 0.5, 1.0, 0.0] for f in AUC_FRACTIONS})
    da, db = load_auc(a), load_auc(b)
    row = compare_auc(da, db)
    assert row["delta"] == pytest.approx(1.0 - 0.5, rel=1e-6)
    assert row["n_pairs"] == 4 and row["reliable"]
    same = compare_auc(da, da)
    assert same["delta"] == 0.0 and same["tie"]


def test_compare_rejects_exclude_self_mismatch(tmp_path):
    # È il caso reale di B.2: i file di § 24 hanno exclude_self=False, quelli nuovi True.
    a, b = tmp_path / "a", tmp_path / "b"
    _write_config(a, {f: [1.0] * 4 for f in AUC_FRACTIONS})
    _write_config(b, {f: [1.0] * 4 for f in AUC_FRACTIONS},
                  meta_overrides={"exclude_self": False})
    with pytest.raises(ValueError, match="exclude_self"):
        compare_auc(load_auc(a), load_auc(b))


def test_discover_keeps_only_complete_configs(tmp_path):
    _write_config(tmp_path / "full", {f: [1.0] * 4 for f in AUC_FRACTIONS})
    _write(tmp_path / "partial_only", 0.25, [1.0] * 4)
    found = [p.name for p in discover_prefixes([tmp_path])]
    assert found == ["full"]


# ----------------------------------------------------------------------
# Danni crop/patch (11 set 2026): stessa ricetta, altra famiglia di file.
# ----------------------------------------------------------------------

def _write_damage(prefix, strategy, values_by_frac, damage=None):
    # `_write` names files as random: write under a scratch prefix, then rename
    for frac, vals in values_by_frac.items():
        scratch = Path(prefix).parent / "__scratch"
        _write(scratch, frac, vals, meta_overrides={
            "partial_label": f"{strategy} f={frac}",
            **({"damage": damage} if damage is not None else {}),
        })
        fraction_path(scratch, frac, "valid").rename(fraction_path(prefix, frac, "valid", strategy))


def test_damage_file_names_follow_the_label_slug(tmp_path):
    assert fraction_path(tmp_path / "v", 0.25, "valid", "crop").name == \
        "v_partial-crop-f0.25_valid.npz"
    assert fraction_path(tmp_path / "v", 0.5, "test", strategy="patch").name == \
        "v_partial-patch-f0.5_test.npz"


def test_discover_and_load_are_per_strategy(tmp_path):
    _write_config(tmp_path / "rooms", {f: [1.0] * 4 for f in AUC_FRACTIONS})
    _write_damage(tmp_path / "boxy", "crop", {f: [0.5] * 4 for f in AUC_FRACTIONS})
    _write_damage(tmp_path / "grid", "patch", {0.25: [0.5] * 4})      # incompleta

    assert [p.name for p in discover_prefixes([tmp_path])] == ["rooms"]
    assert [p.name for p in discover_prefixes([tmp_path], strategy="crop")] == ["boxy"]
    assert discover_prefixes([tmp_path], strategy="patch") == []

    d = load_auc(tmp_path / "boxy", strategy="crop")
    assert d.strategy == "crop" and d.mean == pytest.approx(0.5)
    assert load_auc(tmp_path / "rooms").strategy == "random"
    with pytest.raises(FileNotFoundError):
        load_auc(tmp_path / "boxy")                                   # default = random


def test_patch_grid_mismatch_warns_but_compares(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write_damage(a, "patch", {f: [1.0] * 4 for f in AUC_FRACTIONS},
                  damage={"strategy": "patch", "patch_size": 14, "image_size": 224})
    _write_damage(b, "patch", {f: [0.5] * 4 for f in AUC_FRACTIONS},
                  damage={"strategy": "patch", "patch_size": 16, "image_size": 224})
    with pytest.warns(UserWarning, match="patch_size"):
        row = compare_auc(load_auc(a, strategy="patch"), load_auc(b, strategy="patch"))
    assert row["delta"] == pytest.approx(0.5)


def test_parse_args_strategy_defaults_to_random():
    from src.evaluation.robustness_auc import parse_args
    assert parse_args(["rank", "--dir", "x"]).strategy == "random"
    assert parse_args(["compare", "--a", "x", "--b", "y", "--strategy", "crop"]).strategy == "crop"
    with pytest.raises(SystemExit):
        parse_args(["rank", "--dir", "x", "--strategy", "semantic"])


# ----------------------------------------------------------------------
# Metro §38 (--robust): media per query delle AUC di nowalls-random, crop, patch.
# ----------------------------------------------------------------------

def _write_robust(prefix, by_strategy, damage=None):
    for strategy, values_by_frac in by_strategy.items():
        _write_damage(prefix, strategy, values_by_frac, damage=damage)


def test_robust_auc_is_per_query_mean_of_the_three_damages(tmp_path):
    from src.evaluation.robustness_auc import ROBUST_STRATEGIES, load_robust_auc
    assert set(ROBUST_STRATEGIES) == {"nowalls-random", "crop", "patch"}
    vals = {
        "nowalls-random": {f: [1.0, 0.5, 0.0, 0.2] for f in AUC_FRACTIONS},
        "crop":           {f: [0.4, 0.4, 0.4, 0.4] for f in AUC_FRACTIONS},
        "patch":          {0.25: [1.0, 1.0, 0.0, 0.0], 0.5: [0.5, 0.5, 0.0, 0.0],
                           0.75: [0.0, 0.0, 0.0, 0.0]},
    }
    _write_robust(tmp_path / "cfg", vals)
    d = load_robust_auc(tmp_path / "cfg")
    per_damage = [load_auc(tmp_path / "cfg", strategy=s).auc for s in ROBUST_STRATEGIES]
    np.testing.assert_allclose(d.auc, np.mean(per_damage, axis=0), rtol=1e-6)
    assert d.strategy == "+".join(ROBUST_STRATEGIES)
    assert set(d.components) == set(ROBUST_STRATEGIES)
    # stesse query in tutti i danni -> media per query = media delle tre AUC
    assert d.mean == pytest.approx(np.mean([c.mean for c in d.components.values()]))


def test_robust_needs_all_three_damages(tmp_path):
    from src.evaluation.robustness_auc import discover_robust_prefixes, load_robust_auc
    full = {s: {f: [0.5] * 4 for f in AUC_FRACTIONS} for s in ("nowalls-random", "crop", "patch")}
    _write_robust(tmp_path / "ok", full)
    _write_robust(tmp_path / "no_patch", {k: v for k, v in full.items() if k != "patch"})
    assert [p.name for p in discover_robust_prefixes([tmp_path])] == ["ok"]
    with pytest.raises(FileNotFoundError):
        load_robust_auc(tmp_path / "no_patch")


def test_robust_ignores_the_old_room_damage(tmp_path):
    """Il `random` storico (muri rimasti) non deve pesare sul metro."""
    from src.evaluation.robustness_auc import load_robust_auc
    base = {s: {f: [0.3] * 4 for f in AUC_FRACTIONS} for s in ("nowalls-random", "crop", "patch")}
    _write_robust(tmp_path / "a", base)
    _write_config(tmp_path / "a", {f: [1.0] * 4 for f in AUC_FRACTIONS})    # random altissimo
    _write_robust(tmp_path / "b", base)
    _write_config(tmp_path / "b", {f: [0.0] * 4 for f in AUC_FRACTIONS})    # random nullo
    row = compare_auc(load_robust_auc(tmp_path / "a"), load_robust_auc(tmp_path / "b"))
    assert row["delta"] == pytest.approx(0.0) and row["tie"]


def test_robust_compare_still_warns_on_patch_grid_mismatch(tmp_path):
    from src.evaluation.robustness_auc import load_robust_auc
    vals = {s: {f: [0.5] * 4 for f in AUC_FRACTIONS} for s in ("nowalls-random", "crop")}
    _write_robust(tmp_path / "a", vals)
    _write_robust(tmp_path / "b", vals)
    _write_damage(tmp_path / "a", "patch", {f: [0.5] * 4 for f in AUC_FRACTIONS},
                  damage={"strategy": "patch", "patch_size": 14, "image_size": 224})
    _write_damage(tmp_path / "b", "patch", {f: [0.5] * 4 for f in AUC_FRACTIONS},
                  damage={"strategy": "patch", "patch_size": 16, "image_size": 224})
    with pytest.warns(UserWarning, match="patch_size"):
        compare_auc(load_robust_auc(tmp_path / "a"), load_robust_auc(tmp_path / "b"))


def test_parse_args_robust_flag_keeps_strategy_contract():
    from src.evaluation.robustness_auc import parse_args
    a = parse_args(["rank", "--dir", "x", "--robust"])
    assert a.robust is True and a.strategy == "random"     # --strategy resta una stringa
    assert parse_args(["compare", "--a", "x", "--b", "y"]).robust is False
