# tests/test_perquery.py

"""
Smoke test CPU per il salvataggio per-query (`src/evaluation/perquery.py`) e i
moduli che lo consumano: `random_floor.py`, `significance.py`,
`geometry_variants.py`, e l'integrazione in `vision/evaluation/evaluate.py` /
`graph/evaluation/graph_evaluate.py`.

Obiettivo: rompere l'invariante che conta di piu' -- attivare il salvataggio
per-query non deve MAI cambiare una media stampata -- prima che lo scopra un
run vero. Tutto qui gira su CPU, deterministico, senza dataset reale (.mat) ne'
pesi scaricati: i `RoomMeta` sono costruiti a mano.

Esecuzione: python -m pytest tests/test_perquery.py -v
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
from src.evaluation.significance import check_compatible, compare, paired_values
from src.graph.evaluation.axis_metrics import (
    accumulate_axes as graph_accumulate_axes,
    new_metrics as graph_new_metrics,
)
from src.vision.evaluation.evaluate import (
    _accumulate_axes as vision_accumulate_axes,
    _new_metrics as vision_new_metrics,
)

K_VALUES = (1, 5, 10)


# ----------------------------------------------------------------------
# Gallery sintetica condivisa: 6 piante, RoomMeta scritti a mano (niente .mat).
# p0 e p5 sono singleton (nessun'altra pianta ha lo stesso istogramma/adiacenza
# tipizzata) -> Recall/mAP indefiniti per loro, l'esatto caso limite richiesto.
# p1/p2 e p3/p4 formano due coppie identiche su composizione E topologia.
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# 1. Invarianza numerica: perquery ON/OFF -> stesse medie, bit-identiche.
# ----------------------------------------------------------------------

def test_vision_full_perquery_toggle_does_not_change_metrics():
    """Attivare il recorder nel ramo vision (full) non cambia una virgola
    dell'accumulo: stesso identico contenitore, confrontato con np.array_equal."""
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
    assert skipped_off["composition"] >= 1  # p0 singleton: deve essere stata saltata
    for ax in AXES:
        for met in ("ndcg", "recall", "map"):
            for k in K_VALUES:
                a = np.array(metrics_off[ax][met][k])
                b = np.array(metrics_on[ax][met][k])
                assert np.array_equal(a, b), f"{ax}/{met}@{k}: il recorder ha alterato le medie"


def test_vision_partial_perquery_toggle_does_not_change_metrics():
    """Stesso invariante in modalita' partial (exclude_self=False, self_rr)."""
    metas, names = synthetic_gallery()
    axes = GalleryAxes(metas)
    n = len(metas)
    max_k = max(K_VALUES)

    def fake_ret_rows(qi):
        # alterna: query pari ritrovano il self a rank 2, dispari mai.
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
    """Stesso invariante nel ramo graph (axis_metrics.accumulate_axes)."""
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


# ----------------------------------------------------------------------
# 2. nanmean del salvato == media stampata (entro il float32 su disco).
# ----------------------------------------------------------------------

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

    # Le colonne salvate sono float32 (write_npz le casta esplicitamente),
    # mentre "metrics" qui sopra e' ancora float64 (numpy default): il
    # confronto ammette quindi l'errore di rappresentazione del cast a
    # float32, ~1e-7 relativo, non un arbitrario "allclose".
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


# ----------------------------------------------------------------------
# 3. Allineamento, NaN <=> num_relevant==0, dtype, axis_index/k_index.
# ----------------------------------------------------------------------

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

    # allineamento riga per riga names/qi dopo save->load
    assert list(data.names) == names
    assert list(data.qi) == list(range(n))
    assert data.meta["schema_version"] == SCHEMA_VERSION

    # dtype del contratto
    assert data.names.dtype.kind == "U"
    assert data.qi.dtype == np.int32
    assert data.ndcg.dtype == np.float32
    assert data.recall.dtype == np.float32
    assert data.map.dtype == np.float32
    assert data.num_relevant.dtype == np.int32

    # axis_index / k_index corretti
    assert data.axis_index("geometry") == AXES.index("geometry")
    assert data.axis_index("composition") == AXES.index("composition")
    assert data.k_index(K_VALUES[1]) == 1
    with pytest.raises(KeyError):
        data.axis_index("nonexistent-axis")
    with pytest.raises(KeyError):
        data.k_index(99999)

    # NaN <=> num_relevant == 0 sugli assi discreti
    for ax in DISCRETE_AXES:
        a = data.axis_index(ax)
        for q in range(n):
            is_nan_row = bool(np.isnan(data.recall[a, :, q]).all())
            assert is_nan_row == (int(data.num_relevant[a, q]) == 0), (
                f"asse {ax}, query {q}: NaN={is_nan_row} ma num_relevant={data.num_relevant[a, q]}"
            )

    # ret_rows: padding = -1 dove n_ret < max_k, mai altrove
    assert data.ret_rows is not None and data.n_ret is not None
    for q in range(n):
        n_ret = int(data.n_ret[q])
        row = data.ret_rows[q]
        assert (row[:n_ret] != -1).all()
        assert (row[n_ret:] == -1).all()


# ----------------------------------------------------------------------
# 4. gallery_sha1: deterministico, sensibile a un solo elemento.
# ----------------------------------------------------------------------

def test_gallery_sha1_deterministic_and_sensitive_to_single_element():
    names_a = ["100", "101", "102"]
    names_b = ["100", "101", "102"]
    names_c = ["100", "101", "999"]  # un solo elemento diverso
    assert gallery_sha1(names_a) == gallery_sha1(names_b)
    assert gallery_sha1(names_a) != gallery_sha1(names_c)


# ----------------------------------------------------------------------
# 5. Floor casuale: righe distinte, mai il self, determinismo, seed diversi.
# ----------------------------------------------------------------------

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
    """Un ranking ordinato per similarita' vera (l'IDCG stesso) deve dare
    nDCG=1.0 esatto attraverso lo stesso accumulo usato dal floor/dai rami."""
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


# ----------------------------------------------------------------------
# 6. geometry_sim_weighted: bitwise identica a geometry_sim per w=(1,1,1);
#    diversa per altre pesature (altrimenti il parametro sarebbe inerte).
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# 7. significance: delta 0 su file identici, incompatibilita' rilevate,
#    appaiamento in AND sui NaN.
# ----------------------------------------------------------------------

def _write_minimal_perquery(path, names, geometry_ndcg_values, meta_overrides=None):
    """File per-query minimale: un solo K=10, solo l'asse geometry popolato,
    utile a isolare il comportamento di significance.py senza rifare tutto
    l'accumulo. `geometry_ndcg_values[i]` puo' essere None -> NaN."""
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


def test_significance_gallery_mismatch_requires_explicit_override(tmp_path):
    names_a = ["a", "b", "c"]
    names_b = ["a", "b", "x"]  # una sola query diversa -> sha1 diverso
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_minimal_perquery(p1, names_a, [0.9, 0.5, 0.7])
    _write_minimal_perquery(p2, names_b, [0.9, 0.5, 0.7])

    data_a, data_b = load_perquery(p1), load_perquery(p2)
    with pytest.raises(ValueError):
        check_compatible(data_a.meta, data_b.meta, k=10, allow_gallery_mismatch=False)
    check_compatible(data_a.meta, data_b.meta, k=10, allow_gallery_mismatch=True)  # non solleva


def test_significance_pairing_is_and_of_non_nan_in_both_files(tmp_path):
    names = ["a", "b", "c", "d"]
    values_a = [0.9, None, 0.7, 0.3]   # 'b' saltata in A
    values_b = [0.9, 0.5, None, 0.3]   # 'c' saltata in B (posizione diversa da A)
    p1, p2 = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_minimal_perquery(p1, names, values_a)
    _write_minimal_perquery(p2, names, values_b)

    data_a, data_b = load_perquery(p1), load_perquery(p2)
    a, b, kept = paired_values(data_a, data_b, "geometry", "ndcg", 10)

    assert kept == ["a", "d"]
    assert len(a) == 2 and len(b) == 2
    assert not np.isnan(a).any() and not np.isnan(b).any()
