
"""
CPU smoke test for the CSV results export (`src/evaluation/export_csv_historical.py`).
Synthetic per-query files in tmp: no real results, no dataset.

Protects:
- the CSV AUC is the mean over fractions 0.25/0.5/0.75 (f=0.0 excluded) and queries are paired across systems;
- guard against the `fusion_select` json: a recomputed mean differing from the published one stops the export;
- `num_relevant` is shared ground truth: systems reporting different values stop the export;
- NaN values (skipped queries) become empty cells, not zeros.

Usage: python -m pytest tests/test_export_csv_historical.py -v
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from src.evaluation import export_csv_historical
from src.evaluation.perquery import gallery_sha1, write_npz
from src.evaluation.relevance import AXES
from src.evaluation.robustness_auc import AUC_FRACTIONS, fraction_path

NAMES = ["q0", "q1", "q2", "q3"]
SPLIT = "valid"                      # dummy split: only the file suffix matters
GALLERY = {"n": 10, "sha1": gallery_sha1([f"g{i}" for i in range(10)]), "source": "synthetic"}


# --- synthetic data ---

def _meta(mode, label, axes, k_values):
    return {
        "branch": "test", "run_tag": "sys", "mode": mode, "partial_label": label,
        "split": SPLIT, "exclude_self": True, "query_seed": 42, "gallery": GALLERY,
        "axes": list(axes), "k_values": list(k_values),
    }


def _write_partial(prefix, strategy, self_rr_by_frac, names=NAMES):
    """One file per fraction with only `self_rr` (other matrices NaN)."""
    axes, q = list(AXES), len(names)
    for frac, values in self_rr_by_frac.items():
        write_npz(fraction_path(prefix, frac, SPLIT, strategy),
                  meta=_meta("partial", f"{strategy} f={frac}", axes, [10]),
                  names=names, qi=list(range(q)),
                  ndcg=np.full((len(axes), 1, q), np.nan, dtype=np.float32),
                  recall=np.full((len(axes), 1, q), np.nan, dtype=np.float32),
                  map_=np.full((len(axes), 1, q), np.nan, dtype=np.float32),
                  num_relevant=np.full((len(axes), q), -1, dtype=np.int32),
                  self_rr=np.asarray(values, dtype=np.float32))


def _write_full(path, ndcg_by_axis, num_relevant=None, names=NAMES):
    """A `full` run with per-axis nDCG@10; `recall`/`map` stay NaN."""
    axes, q = list(AXES), len(names)
    ndcg = np.stack([np.asarray(ndcg_by_axis[a], dtype=np.float32) for a in axes])[:, None, :]
    relevant = (np.full((len(axes), q), -1, dtype=np.int32) if num_relevant is None
                else np.asarray(num_relevant, dtype=np.int32))
    write_npz(path, meta=_meta("full", None, axes, [10]), names=names, qi=list(range(q)),
              ndcg=ndcg,
              recall=np.full((len(axes), 1, q), np.nan, dtype=np.float32),
              map_=np.full((len(axes), 1, q), np.nan, dtype=np.float32),
              num_relevant=relevant)


def _two_systems(tmp_path, self_rr_a=0.8, self_rr_b=0.4, names_b=NAMES):
    """Systems `a` and `b`: constant self_rr and nDCG@10, for hand checks."""
    _write_partial(tmp_path / "a", "nowalls-random",
                   {f: [self_rr_a] * len(NAMES) for f in (0.0,) + AUC_FRACTIONS})
    _write_partial(tmp_path / "b", "random",
                   {f: [self_rr_b] * len(names_b) for f in (0.0,) + AUC_FRACTIONS},
                   names=names_b)
    _write_full(tmp_path / f"a_full_{SPLIT}.npz", {a: [0.5] * len(NAMES) for a in AXES})
    _write_full(tmp_path / f"b_full_{SPLIT}.npz",
                {a: [0.25] * len(names_b) for a in AXES}, names=names_b)
    return {
        "a": dict(prefix=tmp_path / "a", strategy="nowalls-random",
                  full=tmp_path / f"a_full_{SPLIT}.npz"),
        "b": dict(prefix=tmp_path / "b", strategy="random",
                  full=tmp_path / f"b_full_{SPLIT}.npz"),
    }


def _read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


# --- pairing and AUC ---

def test_aligned_auc_keeps_only_the_queries_shared_by_all_systems(tmp_path):
    systems = _two_systems(tmp_path, names_b=["q0", "q2"])
    names, aligned = export_csv_historical.aligned_auc(systems, split=SPLIT)
    assert names == ["q0", "q2"]
    assert len(aligned["a"]["auc"]) == 2


def test_auc_in_the_csv_excludes_f0(tmp_path):
    systems = _two_systems(tmp_path)
    # f=0.0 set to zero: if it entered the AUC the mean would drop to 0.6
    _write_partial(tmp_path / "a", "nowalls-random",
                   {0.0: [0.0] * 4, 0.25: [0.8] * 4, 0.5: [0.8] * 4, 0.75: [0.8] * 4})
    names, aligned = export_csv_historical.aligned_auc(systems, split=SPLIT)
    path = export_csv_historical.export_summary(names, aligned, systems, SPLIT, tmp_path / "out")
    row = next(r for r in _read_csv(path) if r["system"] == "a")
    assert float(row["robustness_auc"]) == pytest.approx(0.8)
    assert float(row["self_rr_f0.25"]) == pytest.approx(0.8)


def test_damage_curve_is_long_format_with_a_row_per_fraction(tmp_path):
    systems = _two_systems(tmp_path)
    names, aligned = export_csv_historical.aligned_auc(systems, split=SPLIT)
    rows = _read_csv(export_csv_historical.export_damage_curve(names, aligned, systems, SPLIT, tmp_path / "out"))
    assert len(rows) == len(systems) * (len(AUC_FRACTIONS) + 1)   # +1 = f=0.0, outside the AUC
    for row in rows:
        assert float(row["ci_lo"]) <= float(row["mean_self_rr"]) <= float(row["ci_hi"])


def test_perquery_csv_has_one_row_per_query_and_all_systems(tmp_path):
    systems = _two_systems(tmp_path)
    names, aligned = export_csv_historical.aligned_auc(systems, split=SPLIT)
    rows = _read_csv(export_csv_historical.export_perquery_selfrr(names, aligned, SPLIT, tmp_path / "out"))
    assert [r["query"] for r in rows] == NAMES
    assert float(rows[0]["a_auc"]) == pytest.approx(0.8)
    assert float(rows[0]["b_f0.5"]) == pytest.approx(0.4)


# --- guards ---

def test_summary_stops_when_the_recomputed_mean_contradicts_the_json(tmp_path):
    systems = _two_systems(tmp_path)
    names, aligned = export_csv_historical.aligned_auc(systems, split=SPLIT)
    renamed = {"graph": systems["a"], "vision": systems["b"]}
    aligned = {"graph": aligned["a"], "vision": aligned["b"]}
    published = {"auc_means": {"0": 0.8, "1": 0.4}}
    export_csv_historical.export_summary(names, aligned, renamed, SPLIT, tmp_path / "ok", published)

    published["auc_means"]["0"] = 0.9                    # json no longer matches
    with pytest.raises(ValueError, match="ricalcolato"):
        export_csv_historical.export_summary(names, aligned, renamed, SPLIT, tmp_path / "ko", published)


def test_full_export_stops_when_num_relevant_differs_between_systems(tmp_path):
    systems = _two_systems(tmp_path)
    shared = np.full((len(AXES), len(NAMES)), 7, dtype=np.int32)
    other = shared.copy()
    other[0, 0] = 99
    _write_full(systems["a"]["full"], {a: [0.5] * 4 for a in AXES}, num_relevant=shared)
    _write_full(systems["b"]["full"], {a: [0.25] * 4 for a in AXES}, num_relevant=other)
    with pytest.raises(ValueError, match="num_relevant"):
        export_csv_historical.export_perquery_full(systems, SPLIT, tmp_path / "out")


def test_full_export_stops_when_the_queries_are_not_paired(tmp_path):
    systems = _two_systems(tmp_path, names_b=["q0", "q2"])
    with pytest.raises(ValueError, match="non appaiate"):
        export_csv_historical.export_perquery_full(systems, SPLIT, tmp_path / "out")


# --- NaN and empty cells ---

def test_skipped_queries_become_empty_cells_not_zeros(tmp_path):
    systems = _two_systems(tmp_path)
    values = {a: [0.5] * len(NAMES) for a in AXES}
    values[AXES[1]] = [np.nan] + [0.5] * (len(NAMES) - 1)
    _write_full(systems["a"]["full"], values)
    _write_full(systems["b"]["full"], {a: [0.25] * len(NAMES) for a in AXES})
    rows = _read_csv(export_csv_historical.export_perquery_full(systems, SPLIT, tmp_path / "out"))
    assert rows[0][f"a_ndcg10_{AXES[1]}"] == ""
    assert float(rows[1][f"a_ndcg10_{AXES[1]}"]) == pytest.approx(0.5)


def test_all_k_export_counts_the_valid_queries_per_metric(tmp_path):
    systems = _two_systems(tmp_path)
    rows = _read_csv(export_csv_historical.export_full_metrics_all_k(systems, SPLIT, tmp_path / "out"))
    ndcg = next(r for r in rows if r["system"] == "a" and r["metric"] == "ndcg")
    assert int(ndcg["n_valid"]) == len(NAMES) and float(ndcg["mean"]) == pytest.approx(0.5)
    recall = next(r for r in rows if r["system"] == "a" and r["metric"] == "recall")
    assert int(recall["n_valid"]) == 0 and recall["mean"] == ""   # all NaN: no zero


def test_alpha_sweep_reads_the_published_jsons(tmp_path, monkeypatch):
    path = tmp_path / "select.json"
    path.write_text(json.dumps({"auc_means": {"0": 0.4, "0.5": 0.6, "1": 0.3},
                                "alpha_star": 0.5, "reference_alpha": 0.0}))
    monkeypatch.setattr(export_csv_historical, "ALPHA_PAIRS", {"a-b": path})
    rows = _read_csv(export_csv_historical.export_alpha_sweep(tmp_path / "out"))
    assert [float(r["alpha"]) for r in rows] == [0.0, 0.5, 1.0]       # sorted
    assert [r["is_alpha_star"] for r in rows] == ["0", "1", "0"]


def test_perquery_values_are_not_rounded_so_ties_stay_ties(tmp_path):
    """Tied systems must stay equal in the CSV (1/3 rounded would make `a > b` true by 1e-13)."""
    third = 1.0 / 3.0
    _write_partial(tmp_path / "a", "nowalls-random",
                   {f: [third] * len(NAMES) for f in (0.0,) + AUC_FRACTIONS})
    _write_partial(tmp_path / "b", "random",
                   {f: [third] * len(NAMES) for f in (0.0,) + AUC_FRACTIONS})
    systems = {
        "a": dict(prefix=tmp_path / "a", strategy="nowalls-random", full=None),
        "b": dict(prefix=tmp_path / "b", strategy="random", full=None),
    }
    names, aligned = export_csv_historical.aligned_auc(systems, split=SPLIT)
    rows = _read_csv(export_csv_historical.export_perquery_selfrr(names, aligned, SPLIT, tmp_path / "out"))
    for row in rows:
        assert float(row["a_auc"]) == float(row["b_auc"])      # tie, not a win
        assert not float(row["a_auc"]) > float(row["b_auc"])
