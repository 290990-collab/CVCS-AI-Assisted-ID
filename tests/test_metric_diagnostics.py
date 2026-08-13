# tests/test_metric_diagnostics.py

"""
Smoke test CPU per `src/evaluation/metric_diagnostics.py`.

L'invariante che conta: la diagnostica NON deve produrre numeri suoi. Deve
descrivere esattamente cio' che l'accumulo delle metriche ha gia' fatto, letto
per due strade indipendenti (in linea da `_coverage` in evaluate.py, post-hoc dai
`num_relevant` del file per-query). Se le due strade divergono, una delle due sta
mentendo e la tabella del report con lei.

Gallery sintetica riusata da `tests/test_perquery.py` (6 piante, RoomMeta scritti
a mano): p0 e p5 sono singleton su entrambi gli assi discreti -> 4 query su 6
entrano in Recall/mAP, che e' il caso limite da coprire.

Esecuzione: python -m pytest tests/test_metric_diagnostics.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metric_diagnostics import (
    NO_RELEVANT_SET,
    axis_correlation,
    axis_diagnostics,
    run_diagnostics,
)
from src.evaluation.perquery import PerQueryRecorder, load_perquery
from src.evaluation.relevance import AXES, DISCRETE_AXES, GalleryAxes
from src.vision.evaluation.evaluate import (
    _accumulate_axes as vision_accumulate_axes,
    _coverage,
    _new_metrics as vision_new_metrics,
)
from tests.test_perquery import synthetic_gallery

K_VALUES = (1, 5)


# ----------------------------------------------------------------------
# Utility: una valutazione finta completa sulla gallery sintetica.
# ----------------------------------------------------------------------

def _run_eval(tmp_path, k_values=K_VALUES):
    """Accumula le metriche su tutte le query della gallery sintetica e scrive
    il file per-query. Ritorna (metrics, skipped, PerQueryData, n_query)."""
    metas, names = synthetic_gallery()
    axes = GalleryAxes(metas)
    max_k = max(k_values)

    metrics = vision_new_metrics(k_values)
    skipped = {ax: 0 for ax in DISCRETE_AXES}
    recorder = PerQueryRecorder(k_values, max_k)

    for qi in range(len(metas)):
        ret_rows = [r for r in range(len(metas)) if r != qi][:max_k]
        before = dict(skipped)
        vision_accumulate_axes(metrics, skipped, axes, qi, ret_rows, k_values,
                               exclude_self=True)
        recorder.add(name=names[qi], qi=qi, ret_rows=ret_rows, metrics=metrics,
                     skipped_before=before, skipped=skipped, axes=axes,
                     exclude_self=True)

    path = tmp_path / "diag.npz"
    recorder.write(path, meta={"branch": "test", "mode": "full", "split": "test",
                               "exclude_self": True, "gallery": {"n": len(metas)}})
    return metrics, skipped, load_perquery(path), len(metas)


# ----------------------------------------------------------------------
# 1. Le due strade devono dare la stessa copertura.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("axis", DISCRETE_AXES)
def test_coverage_live_matches_posthoc(tmp_path, axis, k):
    """`_coverage` (live, dalle lunghezze delle liste di accumulo) e
    `axis_diagnostics` (post-hoc, da num_relevant) misurano la stessa cosa."""
    metrics, _, data, _ = _run_eval(tmp_path)
    live = _coverage(metrics, axis, k)
    posthoc = axis_diagnostics(data, axis, k)["coverage"]
    assert live == pytest.approx(posthoc, abs=1e-12)


def test_coverage_counts_the_singletons(tmp_path):
    """p0 e p5 sono singleton su entrambi gli assi: 4 query su 6 nella media."""
    metrics, skipped, data, n_query = _run_eval(tmp_path)
    for axis in DISCRETE_AXES:
        d = axis_diagnostics(data, axis, k=1)
        assert d["n_query"] == n_query
        assert d["n_skipped"] == 2
        assert d["coverage"] == pytest.approx(4 / 6)
        # la diagnostica non puo' contraddire l'accumulo
        assert d["n_skipped"] == skipped[axis]
        assert _coverage(metrics, axis, 1) == pytest.approx(4 / 6)


# ----------------------------------------------------------------------
# 2. Il regime "Recall e' in realta' Precision@k".
# ----------------------------------------------------------------------

def test_precision_regime_switches_with_k(tmp_path):
    """Le query non-singleton hanno 1 solo rilevante: a k=1 sono in regime
    precision (min(k,#rel)=k), a k=5 no (min(k,#rel)=#rel = recall vero)."""
    _, _, data, _ = _run_eval(tmp_path)
    for axis in DISCRETE_AXES:
        assert axis_diagnostics(data, axis, k=1)["precision_regime"] == pytest.approx(1.0)
        assert axis_diagnostics(data, axis, k=5)["precision_regime"] == pytest.approx(0.0)


def test_median_relevant_excludes_skipped_queries(tmp_path):
    """La mediana della classe di equivalenza si calcola sulle query VALUTATE:
    includere le singleton (num_relevant=0) la falserebbe verso il basso."""
    _, _, data, _ = _run_eval(tmp_path)
    for axis in DISCRETE_AXES:
        assert axis_diagnostics(data, axis, k=1)["median_relevant"] == pytest.approx(1.0)


def test_continuous_axis_has_no_diagnostics(tmp_path):
    """La geometria non ha classe di equivalenza: num_relevant vale -1 ovunque e
    la diagnostica deve dire None, non inventare una copertura."""
    _, _, data, _ = _run_eval(tmp_path)
    a = data.axis_index("geometry")
    assert np.all(np.asarray(data.num_relevant[a]) == NO_RELEVANT_SET)
    assert axis_diagnostics(data, "geometry", k=1) is None
    assert set(run_diagnostics(data, k=1)) == set(DISCRETE_AXES)


# ----------------------------------------------------------------------
# 3. Correlazione fra assi.
# ----------------------------------------------------------------------

def test_axis_correlation_shape_and_diagonal():
    """Matrice [3,3] nell'ordine di AXES, diagonale a 1, simmetrica."""
    metas, _ = synthetic_gallery()
    axes = GalleryAxes(metas)
    corr, stats, n_pairs = axis_correlation(axes, num_queries=6)
    assert corr.shape == (len(AXES), len(AXES))
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)
    assert n_pairs == 6 * (len(metas) - 1)      # ogni query vs la gallery, self escluso
    assert set(stats) == set(AXES)


def test_axis_correlation_is_deterministic():
    """Stesso seed -> stesso risultato: la correlazione finisce nel report."""
    metas, _ = synthetic_gallery()
    axes = GalleryAxes(metas)
    a, sa, _ = axis_correlation(axes, num_queries=4, seed=7)
    b, sb, _ = axis_correlation(axes, num_queries=4, seed=7)
    assert np.array_equal(a, b)
    assert sa == sb


def test_axis_correlation_marginals_match_direct_computation():
    """Le statistiche marginali sono quelle dei gain veri, non un ricalcolo:
    confronto con le similarita' prese direttamente da GalleryAxes."""
    metas, _ = synthetic_gallery()
    axes = GalleryAxes(metas)
    _, stats, _ = axis_correlation(axes, num_queries=len(metas), seed=0)

    for ax in AXES:
        vals = []
        for qi in range(len(metas)):
            not_self = np.ones(len(metas), dtype=bool)
            not_self[qi] = False
            vals.append(axes.sim(ax, qi)[not_self])
        expected = np.concatenate(vals)
        assert stats[ax]["mean"] == pytest.approx(float(expected.mean()), abs=1e-6)
        assert stats[ax]["std"] == pytest.approx(float(expected.std()), abs=1e-6)
