
"""
Smoke test CPU of the second-round reading rules (`src/evaluation/multiseed_analysis.py`).

- G: "above the training noise" needs the same sign in 4/4 replicas and |mean| > 2 sd (ddof=1);
- A: the equivalence margin is checked before "beyond"/"below";
- B/C: both CI > 0 -> confirmed, both < 0 -> refuted, else the controls not beaten are named;
- pooling averages per query over the replicas after aligning by name;
- test: a weight on disk different from the registered one stops the reading; the reading is refused while a
  precondition is missing; on the test there is only the single full reading.

Run: python -m pytest tests/test_multiseed_analysis.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import multiseed_analysis as ra


def ci(lo, hi):
    return {"mean": (lo + hi) / 2, "ci_lo": lo, "ci_hi": hi}


def test_noise_rule():
    assert ra.noise_rule([0.010, 0.011, 0.012, 0.013])["verdict"] == "sopra il rumore fra training"
    assert ra.noise_rule([0.010, 0.011, -0.001, 0.013])["verdict"] == "entro il rumore fra training"
    big_sd = ra.noise_rule([0.001, 0.02, 0.001, 0.02])         # same sign, |mean| < 2 sd
    assert big_sd["same_sign"] and big_sd["verdict"] == "entro il rumore fra training"
    assert abs(big_sd["sd"] - np.std([0.001, 0.02, 0.001, 0.02], ddof=1)) < 1e-12


def test_equivalence_order():
    assert ra.equivalence(ci(-0.004, 0.004)) == "equivalente all'effetto d'insieme"
    assert ra.equivalence(ci(0.001, 0.004)) == "equivalente all'effetto d'insieme"   # inside the margin first
    assert ra.equivalence(ci(0.001, 0.006)) == "oltre l'effetto d'insieme"
    assert ra.equivalence(ci(-0.006, -0.001)) == "sotto l'effetto d'insieme"
    assert ra.equivalence(ci(-0.006, 0.002)) == "non concludente"


def test_beyond():
    assert ra.beyond(ci(0.01, 0.02), ci(0.001, 0.02)) == "confermato su 4 repliche"
    assert ra.beyond(ci(-0.02, -0.01), ci(-0.02, -0.001)) == "smentito"
    assert ra.beyond(ci(0.01, 0.02), ci(-0.001, 0.02)) == "non distinguibile da l'altro encoder (SAGE)"
    assert ra.beyond(ci(0.01, 0.02), ci(0.01, 0.02), where="") == "confermato"


def test_pooled_aligns_by_name():
    ref = np.array(["a", "b", "c"])
    per = {1: (ref, np.array([1.0, 2.0, 3.0])), 2: (np.array(["c", "a", "b"]), np.array([30.0, 10.0, 20.0]))}
    assert np.allclose(ra.pooled(per, ref), [5.5, 11.0, 16.5])


def test_check_weight_stops_on_a_different_weight():
    ra.check_weight("x", 0.4, 0.4)
    with pytest.raises(ValueError, match="pre-registrato"):
        ra.check_weight("x", 0.5, 0.4)
    assert ra.TEST_WEIGHTS["v2"] == {42: 0.6, 100042: 0.6, 200042: 0.5, 300042: 0.5}


def test_test_reading_refused_without_preconditions(monkeypatch):
    monkeypatch.setattr(ra, "test_preconditions", lambda: ["manca qualcosa"])
    with pytest.raises(SystemExit, match="rifiutata"):
        ra.analyse_test()


def test_single_test_reading_only():
    with pytest.raises(SystemExit, match="una sola lettura"):
        ra.main(["test", "--geometry"])
