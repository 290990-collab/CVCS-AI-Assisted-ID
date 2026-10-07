
"""
CPU smoke test for the final CSVs and figures (`src/evaluation/export_csv_final.py`, `src/figures/final_results.py`,
`--setup reset` of teaser and pipeline). Synthetic CSVs in tmp.

Protects:
- the export never overwrites an existing folder;
- the five figures are drawn from the CSVs and write pdf, png and provenance;
- `--setup storico` stays the default of teaser and pipeline.

Usage: python -m pytest tests/test_final_outputs.py -v
"""

from __future__ import annotations

import csv
import json
import sys

import pytest

from src.evaluation import export_csv_final as ecr
from src.figures import pipeline, final_results as rr, teaser_valid

SEEDS = ("42", "100042", "200042", "300042")


def _write(path, rows):
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _fake_csvs(d):
    systems = ("graph_W", "layoutgkn", "vision_frozen", "vision_head", "hist_baseline", "fusion_main", "fusion_head")
    _write(d / "test_summary.csv", [dict(system=s, robustness_auc=0.5) for s in systems])
    _write(d / "test_damage_curve.csv", [dict(system=s, fraction=f, mean_self_rr=0.9 - f, ci_lo=0.85 - f,
                                              ci_hi=0.95 - f) for s in systems for f in (0.0, 0.25, 0.5, 0.75)])
    for split in ("valid", "test"):
        _write(d / f"{split}_seeds.csv", [dict(seed=s, gain_main=0.015, gain_main_ci_lo=0.01, gain_main_ci_hi=0.02,
                                               gain_head=0.03, gain_head_ci_lo=0.025, gain_head_ci_hi=0.035)
                                          for s in SEEDS])
        _write(d / f"{split}_curve.csv", [dict(vision=v, encoder="gat", cfg=c, graph_auc=g, vision_auc=0.4, gain=0.1,
                                               ci_lo=0.09, ci_hi=0.11)
                                          for v in ("frozen", "head") for c, g in (("ref", 0.5), ("comb", 0.9))])
    rows = [("valid", "g_fusion_main (vs replica)"), ("valid", "g_control_replica"), ("valid", "g_control_crossenc"),
            ("test", "g_F_frozen"), ("test", "g_C_rep"), ("test", "g_C_enc"), ("test", "g_F_H"),
            ("test", "D_rep"), ("test", "D_enc")]
    _write(d / "controls_gain.csv", [dict(split=s, row=r, value=0.02, ci_lo=0.01, ci_hi=0.03) for s, r in rows])
    _write(d / "valid_alpha_sweep.csv", [dict(fusion=f"{k}/s{s}", alpha=a / 10, robustness_auc=0.9 + a / 1000,
                                              is_alpha_star=int(a == 4))
                                         for k in ("fusion_main", "fusion_head") for s in SEEDS for a in range(11)])
    hist = d / "hist_select.json"
    hist.write_text(json.dumps({"alpha_star": 0.6, "auc_means": {"0": 0.45, "0.6": 0.63, "1": 0.39}}))
    return hist


def test_reset_figures_draw_from_the_csvs_and_write_provenance(tmp_path, monkeypatch):
    d = tmp_path / "csv"
    d.mkdir()
    hist = _fake_csvs(d)
    monkeypatch.setattr(rr, "CSV_DIR", d)
    monkeypatch.setattr(rr, "HISTORICAL_SELECT", hist)
    out = tmp_path / "fig"
    for lang in ("it", "en"):
        monkeypatch.setattr(sys, "argv", ["final_results", "all", "--lang", lang, "--out-dir", str(out / lang)])
        rr.main()
        for name, _ in rr.FIGURES.values():
            for ext in ("pdf", "png", "sources.txt"):
                assert (out / lang / f"{name}.{ext}").exists()
    assert "controls_gain.csv" in (out / "it" / "f_final_controls.sources.txt").read_text()


def test_export_refuses_an_existing_folder(tmp_path, monkeypatch):
    out = tmp_path / "csv_final"
    out.mkdir()
    monkeypatch.setattr(sys, "argv", ["export_csv_final", "--out-dir", str(out)])
    with pytest.raises(SystemExit):
        ecr.main()


def test_historical_setup_is_the_default_of_teaser_and_pipeline():
    assert teaser_valid.SETUPS["storico"] is teaser_valid.SYSTEMS
    assert pipeline.SETUPS["storico"] == (pipeline.SELECT, pipeline.GRAPH_QVEC)
    assert "gat_rg_comb_s42" in pipeline.SETUPS["reset"][1]
    assert all("fusion_head/s42" in prefix for _, prefix, _, _ in teaser_valid.SETUPS["reset"])
