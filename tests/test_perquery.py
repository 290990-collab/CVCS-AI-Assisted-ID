
"""
Smoke test CPU for per-query saving (`src/evaluation/perquery.py`) and its consumers: `random_floor.py`,
`significance.py`, `geometry_variants.py`, and the integration in `vision/evaluation/evaluate.py` /
`graph/evaluation/graph_evaluate.py`.

Main invariant: enabling per-query saving must never change a printed mean. Everything runs on CPU,
deterministic, without the real dataset (.mat) or downloaded weights: `RoomMeta` are built by hand.

Run: python -m pytest tests/test_perquery.py -v
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.data.rplan_metadata import RoomMeta
from src.evaluation.geometry_variants import geometry_sim_weighted
from src.evaluation.perquery import (
    PerQueryRecorder,
    SCHEMA_VERSION,
    gallery_sha1,
    load_perquery,
    write_npz,
)
from src.evaluation.random_floor import random_ret_rows
from src.evaluation.relevance import AXES, DISCRETE_AXES, GalleryAxes
from src.evaluation.significance import (
    check_compatible,
    compare,
    compare_self_rr,
    paired_self_rr,
    paired_values,
)
from src.graph.evaluation.axis_metrics import (
    accumulate_axes as graph_accumulate_axes,
    new_metrics as graph_new_metrics,
)
from src.vision.evaluation.evaluate import (
    _accumulate_axes as vision_accumulate_axes,
    _new_metrics as vision_new_metrics,
)

K_VALUES = (1, 5, 10)


# --- shared synthetic gallery: 6 plans, hand-written RoomMeta (no .mat) ---
# p0 and p5 are singletons (no other plan has the same histogram / typed adjacency): Recall/mAP undefined.
# p1/p2 and p3/p4 are two pairs identical in composition and topology.

def _meta(name, room_types, edges, boxes, footprint):
    return RoomMeta(
        name=name, split="test", room_types=tuple(room_types), edges=tuple(edges),
        boxes=tuple(boxes), footprint=tuple(footprint), entrance=None,
    )


def synthetic_gallery():
    metas = [
        _meta("p0", [0, 1, 2], [(0, 1, 1), (1, 2, 1)],
              [(0, 0, 20, 20), (20, 0, 40, 30), (0, 20, 30, 40)], (0, 0, 40, 40)),
        _meta("p1", [0, 0, 1], [(0, 1, 2), (0, 2, 2)],
              [(0, 0, 10, 10), (10, 0, 20, 10), (0, 10, 10, 20)], (0, 0, 20, 20)),
        _meta("p2", [0, 0, 1], [(0, 1, 2), (0, 2, 2)],
              [(0, 0, 12, 12), (12, 0, 24, 12), (0, 12, 12, 24)], (0, 0, 24, 24)),
        _meta("p3", [2, 2, 3], [(0, 1, 5), (1, 2, 5)],
              [(0, 0, 30, 10), (30, 0, 60, 10), (0, 10, 30, 20)], (0, 0, 60, 20)),
        _meta("p4", [2, 2, 3], [(0, 1, 5), (1, 2, 5)],
              [(0, 0, 25, 8), (25, 0, 50, 8), (0, 8, 25, 16)], (0, 0, 50, 16)),
        _meta("p5", [1, 3, 3], [(0, 2, 7), (1, 2, 7)],
              [(0, 0, 15, 15), (15, 0, 30, 15), (0, 15, 15, 30)], (0, 0, 30, 30)),
    ]
    names = [m.name for m in metas]
    return metas, names


# --- 1. numeric invariance: per-query on/off gives bit-identical means ---

def test_vision_full_perquery_toggle_does_not_change_metrics():
    """Enabling the recorder in the full vision branch leaves the accumulation unchanged (np.array_equal)."""
    metas, names = synthetic_gallery()
    axes = GalleryAxes(metas)
    n = len(metas)
    max_k = max(K_VALUES)

    def fake_ret_rows(qi):
        return [r for r in range(n) if r != qi][:max_k]

    metrics_off = vision_new_metrics(K_VALUES)
    skipped_off = {ax: 0 for ax in DISCRETE_AXES}
    for qi in range(n):
        vision_accumulate_axes(metrics_off, skipped_off, axes, qi, fake_ret_rows(qi),
                                K_VALUES, exclude_self=True)

    metrics_on = vision_new_metrics(K_VALUES)
    skipped_on = {ax: 0 for ax in DISCRETE_AXES}
    recorder = PerQueryRecorder(K_VALUES, max_k)
    for qi in range(n):
        skipped_before = dict(skipped_on)
        ret_rows = fake_ret_rows(qi)
        vision_accumulate_axes(metrics_on, skipped_on, axes, qi, ret_rows, K_VALUES,
                                exclude_self=True)
        recorder.add(name=names[qi], qi=qi, ret_rows=ret_rows, metrics=metrics_on,
                     skipped_before=skipped_before, skipped=skipped_on, axes=axes,
                     exclude_self=True)

    assert skipped_off == skipped_on
    assert skipped_off["composition"] >= 1  # p0 singleton: must be skipped
    for ax in AXES:
        for met in ("ndcg", "recall", "map"):
            for k in K_VALUES:
                a = np.array(metrics_off[ax][met][k])
                b = np.array(metrics_on[ax][met][k])
                assert np.array_equal(a, b), f"{ax}/{met}@{k}: il recorder ha alterato le medie"


def test_vision_partial_perquery_toggle_does_not_change_metrics():
    """Same invariant in partial mode (exclude_self=False, self_rr)."""
    metas, names = synthetic_gallery()
    axes = GalleryAxes(metas)
    n = len(metas)
    max_k = max(K_VALUES)

    def fake_ret_rows(qi):
        # even queries find self at rank 2, odd never
        others = [r for r in range(n) if r != qi]
        if qi % 2 == 0:
            rows = others[:1] + [qi] + others[1:]
        else:
            rows = others
        return rows[:max_k]

    def self_rr_of(ret_rows, qi):
        arr = np.asarray(ret_rows, dtype=int)
        hit = np.where(arr == qi)[0]
        rank = int(hit[0]) + 1 if len(hit) else None
        return 1.0 / rank if rank else 0.0

    metrics_off = vision_new_metrics(K_VALUES)
    skipped_off = {ax: 0 for ax in DISCRETE_AXES}
    for qi in range(n):
        vision_accumulate_axes(metrics_off, skipped_off, axes, qi, fake_ret_rows(qi),
                                K_VALUES, exclude_self=False)

    metrics_on = vision_new_metrics(K_VALUES)
    skipped_on = {ax: 0 for ax in DISCRETE_AXES}
    recorder = PerQueryRecorder(K_VALUES, max_k, with_self_rr=True)
    for qi in range(n):
        skipped_before = dict(skipped_on)
        ret_rows = fake_ret_rows(qi)
        vision_accumulate_axes(metrics_on, skipped_on, axes, qi, ret_rows, K_VALUES,
                                exclude_self=False)
        recorder.add(name=names[qi], qi=qi, ret_rows=ret_rows, metrics=metrics_on,
                     skipped_before=skipped_before, skipped=skipped_on, axes=axes,
                     exclude_self=False, self_rr=self_rr_of(ret_rows, qi))

    assert skipped_off == skipped_on
    for ax in AXES:
        for met in ("ndcg", "recall", "map"):
            for k in K_VALUES:
                a = np.array(metrics_off[ax][met][k])
                b = np.array(metrics_on[ax][met][k])
                assert np.array_equal(a, b), f"{ax}/{met}@{k}: il recorder ha alterato le medie"


def test_graph_full_perquery_toggle_does_not_change_metrics():
    """Same invariant in the graph branch (axis_metrics.accumulate_axes)."""
    metas, names = synthetic_gallery()
    axes = GalleryAxes(metas)
    n = len(metas)
    max_k = max(K_VALUES)

    def fake_ret_rows(qi):
        return [r for r in range(n) if r != qi][:max_k]

    metrics_off = graph_new_metrics(K_VALUES)
    skipped_off = {ax: 0 for ax in DISCRETE_AXES}
    for qi in range(n):
        graph_accumulate_axes(metrics_off, skipped_off, axes, qi, fake_ret_rows(qi),
                               K_VALUES, exclude_self=True)

    metrics_on = graph_new_metrics(K_VALUES)
    skipped_on = {ax: 0 for ax in DISCRETE_AXES}
    recorder = PerQueryRecorder(K_VALUES, max_k)
    for qi in range(n):
        skipped_before = dict(skipped_on)
        ret_rows = fake_ret_rows(qi)
        graph_accumulate_axes(metrics_on, skipped_on, axes, qi, ret_rows, K_VALUES,
                               exclude_self=True)
        recorder.add(name=names[qi], qi=qi, ret_rows=ret_rows, metrics=metrics_on,
                     skipped_before=skipped_before, skipped=skipped_on, axes=axes,
                     exclude_self=True)

    assert skipped_off == skipped_on
    for ax in AXES:
        for met in ("ndcg", "recall", "map"):
            for k in K_VALUES:
                a = np.array(metrics_off[ax][met][k])
                b = np.array(metrics_on[ax][met][k])
                assert np.array_equal(a, b), f"{ax}/{met}@{k}: il recorder ha alterato le medie"


# --- 2. nanmean of the saved file == printed mean (within on-disk float32) ---

def test_saved_perquery_nanmean_matches_printed_mean(tmp_path):
    metas, names = synthetic_gallery()
    axes = GalleryAxes(metas)
    n = len(metas)
    max_k = max(K_VALUES)

    metrics = vision_new_metrics(K_VALUES)
    skipped = {ax: 0 for ax in DISCRETE_AXES}
    recorder = PerQueryRecorder(K_VALUES, max_k)
    for qi in range(n):
        skipped_before = dict(skipped)
        ret_rows = [r for r in range(n) if r != qi][:max_k]
        vision_accumulate_axes(metrics, skipped, axes, qi, ret_rows, K_VALUES, exclude_self=True)
        recorder.add(name=names[qi], qi=qi, ret_rows=ret_rows, metrics=metrics,
                     skipped_before=skipped_before, skipped=skipped, axes=axes,
                     exclude_self=True)

    out = tmp_path / "perq.npz"
    recorder.write(out, meta={
        "branch": "vision", "run_tag": "test", "mode": "full", "partial_label": None,
        "split": "test", "exclude_self": True, "query_seed": 0,
        "gallery": {"n": n, "sha1": gallery_sha1(names), "source": "synthetic"},
    })
    data = load_perquery(out)

    # saved columns are float32 (write_npz casts), "metrics" is float64: tolerance = cast error, ~1e-7 relative
    for ax in AXES:
        for met in ("ndcg", "recall", "map"):
            if ax not in DISCRETE_AXES and met != "ndcg":
                continue
            for k in K_VALUES:
                values = metrics[ax][met][k]
                printed = float(np.mean(values)) if values else float("nan")
                row = getattr(data, met)[data.axis_index(ax), data.k_index(k)]
                saved_mean = float(np.nanmean(row))
                if np.isnan(printed):
                    assert np.isnan(saved_mean)
                else:
                    assert abs(saved_mean - printed) < 1e-5, f"{ax}/{met}@{k}: {saved_mean} vs {printed}"


# --- 3. alignment, NaN <=> num_relevant==0, dtype, axis_index/k_index ---

def test_saved_perquery_alignment_dtype_and_nan_semantics(tmp_path):
    metas, names = synthetic_gallery()
    axes = GalleryAxes(metas)
    n = len(metas)
    max_k = max(K_VALUES)

    metrics = vision_new_metrics(K_VALUES)
    skipped = {ax: 0 for ax in DISCRETE_AXES}
    recorder = PerQueryRecorder(K_VALUES, max_k)
    for qi in range(n):
        skipped_before = dict(skipped)
        ret_rows = [r for r in range(n) if r != qi][:max_k]
        vision_accumulate_axes(metrics, skipped, axes, qi, ret_rows, K_VALUES, exclude_self=True)
        recorder.add(name=names[qi], qi=qi, ret_rows=ret_rows, metrics=metrics,
                     skipped_before=skipped_before, skipped=skipped, axes=axes,
                     exclude_self=True)

    out = tmp_path / "perq.npz"
    recorder.write(out, meta={
        "branch": "vision", "run_tag": "test", "mode": "full", "partial_label": None,
        "split": "test", "exclude_self": True, "query_seed": 0,
        "gallery": {"n": n, "sha1": gallery_sha1(names), "source": "synthetic"},
    })
    data = load_perquery(out)

    # names/qi aligned row by row after save/load
    assert list(data.names) == names
    assert list(data.qi) == list(range(n))
    assert data.meta["schema_version"] == SCHEMA_VERSION

    # contract dtypes
    assert data.names.dtype.kind == "U"
    assert data.qi.dtype == np.int32
    assert data.ndcg.dtype == np.float32
    assert data.recall.dtype == np.float32
    assert data.map.dtype == np.float32
    assert data.num_relevant.dtype == np.int32

    # axis_index / k_index
    assert data.axis_index("geometry") == AXES.index("geometry")
    assert data.axis_index("composition") == AXES.index("composition")
    assert data.k_index(K_VALUES[1]) == 1
    with pytest.raises(KeyError):
        data.axis_index("nonexistent-axis")
    with pytest.raises(KeyError):
        data.k_index(99999)

    # NaN <=> num_relevant == 0 on discrete axes
    for ax in DISCRETE_AXES:
        a = data.axis_index(ax)
        for q in range(n):
            is_nan_row = bool(np.isnan(data.recall[a, :, q]).all())
            assert is_nan_row == (int(data.num_relevant[a, q]) == 0), (
                f"asse {ax}, query {q}: NaN={is_nan_row} ma num_relevant={data.num_relevant[a, q]}"
            )

    # ret_rows: padding = -1 where n_ret < max_k, nowhere else
    assert data.ret_rows is not None and data.n_ret is not None
    for q in range(n):
        n_ret = int(data.n_ret[q])
        row = data.ret_rows[q]
        assert (row[:n_ret] != -1).all()
        assert (row[n_ret:] == -1).all()


# --- 4. gallery_sha1: deterministic, sensitive to a single element ---

def test_gallery_sha1_deterministic_and_sensitive_to_single_element():
    names_a = ["100", "101", "102"]
    names_b = ["100", "101", "102"]
    names_c = ["100", "101", "999"]  # one differing element
    assert gallery_sha1(names_a) == gallery_sha1(names_b)
    assert gallery_sha1(names_a) != gallery_sha1(names_c)


# --- 5. random floor: distinct rows, never self, deterministic, seeds differ ---

def test_random_ret_rows_are_distinct_and_exclude_self():
    rng = random.Random(0)
    rows = random_ret_rows(rng, n_gallery=50, qi=7, max_k=10)
    assert len(rows) == len(set(rows))
    assert 7 not in rows


def test_random_ret_rows_deterministic_with_fixed_seed():
    r1 = random_ret_rows(random.Random(3), 50, 7, 10)
    r2 = random_ret_rows(random.Random(3), 50, 7, 10)
    assert r1 == r2


def test_random_ret_rows_differ_across_seeds():
    r1 = random_ret_rows(random.Random(1), 50, 7, 10)
    r2 = random_ret_rows(random.Random(2), 50, 7, 10)
    assert r1 != r2


def test_ideal_ranking_gives_ndcg_one():
    """A ranking sorted by true similarity (the IDCG itself) gives nDCG = 1.0 through the same accumulation."""
    metas, _ = synthetic_gallery()
    axes = GalleryAxes(metas)
    n = len(metas)
    qi = 0
    sims = axes.geometry_sim(qi)
    ideal_order = sorted((r for r in range(n) if r != qi), key=lambda r: -sims[r])

    metrics = graph_new_metrics(K_VALUES)
    skipped = {ax: 0 for ax in DISCRETE_AXES}
    graph_accumulate_axes(metrics, skipped, axes, qi, ideal_order, K_VALUES, exclude_self=True)

    for k in K_VALUES:
        assert metrics["geometry"]["ndcg"][k][-1] == pytest.approx(1.0, abs=1e-6)


# --- 6. geometry_sim_weighted: bitwise equal to geometry_sim for w=(1,1,1), different otherwise ---

def test_geometry_sim_weighted_bitwise_identical_for_equal_weights():
    metas, _ = synthetic_gallery()
    axes = GalleryAxes(metas)
    for qi in range(len(metas)):
        original = axes.geometry_sim(qi)
        weighted = geometry_sim_weighted(axes, qi, (1.0, 1.0, 1.0))
        assert np.array_equal(original, weighted), f"qi={qi}: w=(1,1,1) non e' bitwise identico"


def test_geometry_sim_weighted_differs_for_other_weights():
    metas, _ = synthetic_gallery()
    axes = GalleryAxes(metas)
    original = axes.geometry_sim(0)
    weighted = geometry_sim_weighted(axes, 0, (1.0, 0.0, 0.0))
    assert not np.array_equal(original, weighted), "il parametro dei pesi e' inerte"


# --- 7. significance: delta 0 on identical files, incompatibilities detected, NaN-AND pairing ---

def _write_minimal_perquery(path, names, geometry_ndcg_values, meta_overrides=None):
    """Minimal per-query file: single K=10, only the geometry axis; `geometry_ndcg_values[i]` may be None -> NaN."""
    q = len(names)
    axes_list = list(AXES)
    k_values = [10]
    ndcg = np.full((len(axes_list), 1, q), np.nan, dtype=np.float32)
    gi = axes_list.index("geometry")
    for i, v in enumerate(geometry_ndcg_values):
        ndcg[gi, 0, i] = np.nan if v is None else v
    recall = np.full((len(axes_list), 1, q), np.nan, dtype=np.float32)
    map_ = np.full((len(axes_list), 1, q), np.nan, dtype=np.float32)
    num_relevant = np.full((len(axes_list), q), -1, dtype=np.int32)

    meta = {
        "branch": "test", "run_tag": "sys", "mode": "full", "partial_label": None,
        "split": "test", "exclude_self": True, "query_seed": 0,
        "gallery": {"n": q, "sha1": gallery_sha1(names), "source": "synthetic"},
        "axes": axes_list, "k_values": k_values,
    }
    if meta_overrides:
        meta.update(meta_overrides)
    write_npz(path, meta=meta, names=names, qi=list(range(q)), ndcg=ndcg,
              recall=recall, map_=map_, num_relevant=num_relevant)


def test_significance_identical_files_give_zero_delta(tmp_path):
    names = ["a", "b", "c", "d", "e"]
    values = [0.9, 0.5, 0.7, 0.2, 0.6]
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_minimal_perquery(p1, names, values)
    _write_minimal_perquery(p2, names, values)

    data_a, data_b = load_perquery(p1), load_perquery(p2)
    check_compatible(data_a.meta, data_b.meta, k=10, allow_gallery_mismatch=False)
    result = compare(data_a, data_b, "geometry", "ndcg", 10)

    assert result["n_pairs"] == len(names)
    assert result["delta"] == 0.0
    assert result["ci_lo"] <= 0.0 <= result["ci_hi"]


def test_significance_meta_mismatch_raises(tmp_path):
    names = ["a", "b", "c"]
    values = [0.9, 0.5, 0.7]
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_minimal_perquery(p1, names, values)
    _write_minimal_perquery(p2, names, values, meta_overrides={"split": "valid"})

    data_a, data_b = load_perquery(p1), load_perquery(p2)
    with pytest.raises(ValueError):
        check_compatible(data_a.meta, data_b.meta, k=10, allow_gallery_mismatch=False)


def _write_partial_perquery(path, names, self_rr_values, meta_overrides=None):
    """Partial-mode per-query file: like `_write_minimal_perquery` plus `self_rr` (self reciprocal rank, partial only)."""
    q = len(names)
    axes_list = list(AXES)
    shape = (len(axes_list), 1, q)
    meta = {
        "branch": "test", "run_tag": "sys", "mode": "partial",
        "partial_label": "random f=0.75",
        "split": "valid", "exclude_self": False, "query_seed": 0,
        "gallery": {"n": q, "sha1": gallery_sha1(names), "source": "synthetic"},
        "axes": axes_list, "k_values": [10],
    }
    if meta_overrides:
        meta.update(meta_overrides)
    write_npz(path, meta=meta, names=names, qi=list(range(q)),
              ndcg=np.full(shape, np.nan, dtype=np.float32),
              recall=np.full(shape, np.nan, dtype=np.float32),
              map_=np.full(shape, np.nan, dtype=np.float32),
              num_relevant=np.full((len(axes_list), q), -1, dtype=np.int32),
              self_rr=np.asarray(self_rr_values, dtype=np.float32))


def test_self_rr_paired_delta_matches_hand_computation(tmp_path):
    """Paired delta on self_rr is the mean of differences, paired by name (B has the same queries reordered)."""
    names_a = ["a", "b", "c", "d"]
    rr_a = [1.0, 0.5, 0.25, 0.2]
    names_b = ["d", "c", "b", "a"]          # same set, reversed order
    rr_b = [0.1, 0.25, 0.5, 0.5]            # i.e. a=0.5, b=0.5, c=0.25, d=0.1

    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_partial_perquery(p1, names_a, rr_a)
    _write_partial_perquery(p2, names_b, rr_b,
                            meta_overrides={"gallery": {
                                "n": len(names_a),
                                "sha1": gallery_sha1(names_a),   # same gallery
                                "source": "synthetic"}})

    data_a, data_b = load_perquery(p1), load_perquery(p2)
    a, b, names = paired_self_rr(data_a, data_b)
    assert names == names_a                                  # A order, deterministic
    np.testing.assert_allclose(b, [0.5, 0.5, 0.25, 0.1], rtol=0, atol=1e-6)

    res = compare_self_rr(data_a, data_b)
    expected = float(np.mean(np.asarray(rr_a) - np.asarray([0.5, 0.5, 0.25, 0.1])))
    assert res["n_pairs"] == 4
    assert res["metric"] == "self_rr"
    assert res["k"] == "-"                                   # self_rr has no depth
    np.testing.assert_allclose(res["delta"], expected, rtol=0, atol=1e-9)


def test_self_rr_identical_files_give_zero_delta(tmp_path):
    names = ["a", "b", "c", "d", "e"]
    rr = [1.0, 0.5, 0.3333, 0.25, 0.2]
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_partial_perquery(p1, names, rr)
    _write_partial_perquery(p2, names, rr)

    res = compare_self_rr(load_perquery(p1), load_perquery(p2))
    assert res["delta"] == 0.0
    assert res["ci_lo"] <= 0.0 <= res["ci_hi"]


def test_self_rr_on_full_file_raises_with_actionable_message(tmp_path):
    """A full file has no self_rr: the error must say so, not blow up on None."""
    names = ["a", "b", "c"]
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_minimal_perquery(p1, names, [0.9, 0.5, 0.7])       # mode="full"
    _write_partial_perquery(p2, names, [1.0, 0.5, 0.25])

    with pytest.raises(ValueError, match="self_rr"):
        paired_self_rr(load_perquery(p1), load_perquery(p2))


def test_self_rr_different_masking_level_is_rejected(tmp_path):
    """Comparing f=0.5 with f=0.75 would measure masking level, not robustness: `partial_label` in STRICT_META_KEYS blocks it."""
    names = ["a", "b", "c"]
    rr = [1.0, 0.5, 0.25]
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_partial_perquery(p1, names, rr)
    _write_partial_perquery(p2, names, rr,
                            meta_overrides={"partial_label": "random f=0.5"})

    data_a, data_b = load_perquery(p1), load_perquery(p2)
    with pytest.raises(ValueError, match="partial_label"):
        check_compatible(data_a.meta, data_b.meta, k=10, allow_gallery_mismatch=False)


def test_significance_gallery_mismatch_requires_explicit_override(tmp_path):
    names_a = ["a", "b", "c"]
    names_b = ["a", "b", "x"]  # one differing query -> different sha1
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_minimal_perquery(p1, names_a, [0.9, 0.5, 0.7])
    _write_minimal_perquery(p2, names_b, [0.9, 0.5, 0.7])

    data_a, data_b = load_perquery(p1), load_perquery(p2)
    with pytest.raises(ValueError):
        check_compatible(data_a.meta, data_b.meta, k=10, allow_gallery_mismatch=False)
    check_compatible(data_a.meta, data_b.meta, k=10, allow_gallery_mismatch=True)  # does not raise


def test_significance_pairing_is_and_of_non_nan_in_both_files(tmp_path):
    names = ["a", "b", "c", "d"]
    values_a = [0.9, None, 0.7, 0.3]   # 'b' skipped in A
    values_b = [0.9, 0.5, None, 0.3]   # 'c' skipped in B (different position from A)
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_minimal_perquery(p1, names, values_a)
    _write_minimal_perquery(p2, names, values_b)

    data_a, data_b = load_perquery(p1), load_perquery(p2)
    a, b, kept = paired_values(data_a, data_b, "geometry", "ndcg", 10)

    assert kept == ["a", "d"]
    assert len(a) == 2 and len(b) == 2
    assert not np.isnan(a).any() and not np.isnan(b).any()


# --- 8. area_removed (crop/patch vision damages): optional field ---

def _write_area_perquery(path, names, area_removed):
    q = len(names)
    axes_list = list(AXES)
    shape = (len(axes_list), 1, q)
    meta = {
        "branch": "test", "run_tag": "sys", "mode": "partial",
        "partial_label": "crop f=0.5", "split": "valid", "exclude_self": True,
        "query_seed": 0, "axes": axes_list, "k_values": [10],
        "gallery": {"n": q, "sha1": gallery_sha1(names), "source": "synthetic"},
    }
    write_npz(path, meta=meta, names=names, qi=list(range(q)),
              ndcg=np.full(shape, np.nan, dtype=np.float32),
              recall=np.full(shape, np.nan, dtype=np.float32),
              map_=np.full(shape, np.nan, dtype=np.float32),
              num_relevant=np.full((len(axes_list), q), -1, dtype=np.int32),
              self_rr=np.ones(q, dtype=np.float32), area_removed=area_removed)


def test_area_removed_round_trip(tmp_path):
    p = tmp_path / "a.npz"
    _write_area_perquery(p, ["a", "b", "c"], [0.25, 0.5, 0.0])
    data = load_perquery(p)
    assert data.area_removed.dtype == np.float32
    np.testing.assert_array_equal(data.area_removed, np.float32([0.25, 0.5, 0.0]))
    assert data.meta["schema_version"] == SCHEMA_VERSION


def test_file_without_area_removed_loads_as_none(tmp_path):
    p = tmp_path / "a.npz"
    _write_partial_perquery(p, ["a", "b"], [1.0, 0.5])
    assert load_perquery(p).area_removed is None


def test_area_removed_shape_is_validated(tmp_path):
    with pytest.raises(ValueError, match="area_removed"):
        _write_area_perquery(tmp_path / "a.npz", ["a", "b", "c"], [0.25, 0.5])


def test_recorder_with_area_removed_requires_it_and_writes_it(tmp_path):
    metas, names = synthetic_gallery()
    axes = GalleryAxes(metas)
    max_k = max(K_VALUES)
    metrics = vision_new_metrics(K_VALUES)
    skipped = {ax: 0 for ax in DISCRETE_AXES}
    recorder = PerQueryRecorder(K_VALUES, max_k, with_self_rr=True, with_area_removed=True)
    rows = [1, 2, 3]
    skipped_before = dict(skipped)
    vision_accumulate_axes(metrics, skipped, axes, 0, rows, K_VALUES, exclude_self=True)
    with pytest.raises(ValueError, match="area_removed"):
        recorder.add(name=names[0], qi=0, ret_rows=rows, metrics=metrics,
                     skipped_before=skipped_before, skipped=skipped, axes=axes,
                     exclude_self=True, self_rr=1.0)
    recorder.add(name=names[0], qi=0, ret_rows=rows, metrics=metrics,
                 skipped_before=skipped_before, skipped=skipped, axes=axes,
                 exclude_self=True, self_rr=1.0, area_removed=0.4)
    p = tmp_path / "r.npz"
    recorder.write(p, meta={"mode": "partial", "partial_label": "crop f=0.5"})
    np.testing.assert_array_equal(load_perquery(p).area_removed, np.float32([0.4]))

    plain = PerQueryRecorder(K_VALUES, max_k, with_self_rr=True)
    plain.add(name=names[0], qi=0, ret_rows=rows, metrics=metrics,
              skipped_before=skipped_before, skipped=skipped, axes=axes,
              exclude_self=True, self_rr=1.0)
    q = tmp_path / "q.npz"
    plain.write(q, meta={"mode": "partial", "partial_label": "random f=0.5"})
    assert load_perquery(q).area_removed is None
