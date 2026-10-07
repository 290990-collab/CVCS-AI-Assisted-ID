
"""
CPU tests of the secondary fusion with the vision head (src/evaluation/fusion_head.py).
Uses the real (sha1-pinned) head checkpoint; no dataset, no GPU.

Usage: python -m pytest tests/test_fusion_head.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import fusion_head as fh
from src.evaluation import fusion_select as fs
from src.evaluation import late_fusion as lf
from src.evaluation.query_vectors import QueryVectorData

RAW_DIM = 768


def _whitening(dim, seed=0):
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=(1, dim)).astype(np.float32)
    mat = rng.normal(size=(dim, dim)).astype(np.float32) / np.sqrt(dim)
    return mean, mat


def _qvec(head_meta, raw, final=None, mean=None, mat=None):
    n = len(raw)
    return QueryVectorData(meta={"head": head_meta}, names=np.array([str(i) for i in range(n)]),
                           qi=np.arange(n), vectors=raw, removed_ptr=np.zeros(n + 1, dtype=np.int32),
                           removed_idx=np.zeros(0, dtype=np.int32), vectors_final=final,
                           whiten_mean=mean, whiten_matrix=mat)


def test_head_is_the_pinned_one():
    assert fh.file_sha1(fh.HEAD_DIR / fh.HEAD_FILE) == fh.HEAD_SHA1
    out = fh.apply_head(np.random.default_rng(0).normal(size=(5, RAW_DIM)).astype(np.float32))
    assert out.shape == (5, fh.HEAD_OUT)


def test_patch_applies_head_before_whitening_and_restores():
    raw = np.random.default_rng(1).normal(size=(7, RAW_DIM)).astype(np.float32)
    mean, mat = _whitening(fh.HEAD_OUT)
    expected = lf.whiten(fh.apply_head(raw), mean, mat)
    orig = (lf.whiten, lf.check_branch, fs.check_branch)
    with fh.head_patch():
        got = lf.whiten(raw, mean, mat)
        assert lf.check_branch is fs.check_branch and lf.check_branch is not orig[1]
    assert np.array_equal(got, expected)
    assert (lf.whiten, lf.check_branch, fs.check_branch) == orig
    assert np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("meta", [None, {"enabled": True, "file": "head.pt"}, {"enabled": False, "file": fh.HEAD_FILE}])
def test_only_the_preregistered_head_is_accepted(meta):
    raw = np.zeros((2, RAW_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="head"):
        fh._check_vision_head_meta({0.25: _qvec(meta, raw)})
    ok = fh._check_vision_head_meta({0.25: _qvec(dict(fh.HEAD_META), raw)})
    assert ok[0.25].meta["head"] is None          # what the main fusion's checks expect


def test_query_final_check(monkeypatch):
    raw = np.random.default_rng(2).normal(size=(6, RAW_DIM)).astype(np.float32)
    mean, mat = _whitening(fh.HEAD_OUT, 3)
    final = lf.whiten(fh.apply_head(raw), mean, mat)
    good = {0.5: _qvec(dict(fh.HEAD_META), raw, final, mean, mat)}
    monkeypatch.setattr(lf, "load_branch_qvecs", lambda *a, **k: good)
    assert fh.check_query_final("x", "valid", [0.5])["0.5"] <= fh.QUERY_TOL
    bad = {0.5: _qvec(dict(fh.HEAD_META), raw, final + 1e-3, mean, mat)}
    monkeypatch.setattr(lf, "load_branch_qvecs", lambda *a, **k: bad)
    with pytest.raises(ValueError, match="differ"):
        fh.check_query_final("x", "valid", [0.5])


def test_four_outcomes_of_59H():
    assert fh.outcome(False, 0.1, 0.2) == "il guadagno dipendeva dal vision non allenato sul danno"
    assert fh.outcome(True, -0.2, -0.1) == "il guadagno resta ma si riduce"
    assert fh.outcome(True, -0.1, 0.1) == "il guadagno non dipende dall'allenamento del vision"
    assert fh.outcome(True, 0.1, 0.2) == "il guadagno cresce con la head"


def test_paths_are_new_folders():
    p = fh.head_paths(42, "valid")
    for k in ("VISION_QVEC", "VISION_PQ", "FUSION_DIR", "SELECT_JSON", "COMPARE_JSON", "HEAD_JSON"):
        assert "/vision_head/" in p[k] or "/fusion_head/" in p[k]
    assert p["MAIN_FUSION_DIR"].endswith("fusion/s42/fusion_valid")


def test_qvec_accepts_whitening_after_the_head(tmp_path):
    # with a head the whitening is fit on the head output (256), not on the raw vector (768)
    from src.evaluation.query_vectors import load_qvec, write_qvec
    raw = np.zeros((2, RAW_DIM), dtype=np.float32)
    mean, mat = _whitening(fh.HEAD_OUT)
    kw = dict(names=["a", "b"], qi=[0, 1], vectors=raw, removed=[[0], []],
              vectors_final=np.zeros((2, fh.HEAD_OUT), dtype=np.float32), whiten_mean=mean, whiten_matrix=mat)
    write_qvec(tmp_path / "h.npz", meta={"head": dict(fh.HEAD_META)}, **kw)
    assert load_qvec(tmp_path / "h.npz").whiten_mean.shape == (1, fh.HEAD_OUT)
    with pytest.raises(ValueError, match="whiten_mean"):      # without a head: unchanged check
        write_qvec(tmp_path / "n.npz", meta={"head": None}, **kw)
