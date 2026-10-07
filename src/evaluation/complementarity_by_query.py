"""
Per-query breakdown of the vision complementarity gain over the ensemble effect (seed 42, descriptive).

Reads only per-query files already on disk. Complementarity gain (CG) of a fusion vision + W, per query:
    g(X)  = AUC(fusion X at its weight) - AUC(best branch of X)
    CG    = g(vision + W) - g(control),   control = the ensemble control with the larger mean gain
            among W + second copy of W and W + sage/comb (both are also written)
CG > 0: the vision adds more than a second graph model would.

Groups:
    level      fraction of rooms removed (0.25 / 0.5 / 0.75)
    twins      gallery plans with the same room graph as the complete query plan (`num_relevant`, topology axis)
    graph_ok   W alone ranks the original first at every level; uses W + sage/comb as control
               (its best branch is W s42, whereas for W + second copy it is the copy)
    rooms      rooms of the complete plan (RPLAN metadata)

Each number is a paired mean over the group with a bootstrap CI 95% (B = 10000, seed 0).

Usage (CPU, under a minute; the RPLAN metadata index takes most of it):
    python -m src.evaluation.complementarity_by_query --out-dir results/csv_complementarity
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.evaluation import fusion_select as fs
from src.evaluation.late_fusion import fused_full_path
from src.evaluation.significance import bootstrap_ci

ROOT = Path("results/final_pipeline")
FRACTIONS = (0.25, 0.5, 0.75)
VISIONS = {"frozen": "fusion/s42", "head": "fusion_head/s42", "head_v2": "fusion_head_v2/s42"}
CONTROLS = {"split": {"valid": "controls", "test": "controls_test"}, "names": ("replica", "crossenc")}
TWIN_BINS = ((0, 0, "0"), (1, 9, "1-9"), (10, 99, "10-99"), (100, 10**9, "100+"))
ROOM_BINS = ((1, 5, "≤5"), (6, 7, "6-7"), (8, 99, "≥8"))


def fusion_dirs(split: str) -> dict[str, tuple[Path, str]]:
    """{name: (fusion dir, pair)} of the five seed-42 fusions on `split`."""
    out = {v: (ROOT / d / f"fusion_{split}", "vision-graph") for v, d in VISIONS.items()}
    for c in CONTROLS["names"]:
        out[c] = (ROOT / CONTROLS["split"][split] / c / f"fusion_{split}", "graph-graph")
    return out


def select_json(fdir: Path, split: str) -> Path:
    return fdir.parent / f"select_{split}.json"


def load_fusion(fdir: Path, split: str, pair: str) -> dict:
    """Per-query AUC and per-level self RR at the selected weight and at the best branch."""
    sel = json.loads(select_json(fdir, split).read_text())
    # select json stores the old path (results/reset_graphs, symlink to final_pipeline)
    if Path(sel["fusion_dir"]).resolve() != fdir.resolve() or sel["split"] != split:
        raise ValueError(f"{select_json(fdir, split)} does not describe {fdir}")
    star = float(sel["alpha_star"])
    datas, comp = fs._fusion_endpoints(fdir, split, star, fs.pair_spec(pair)["fused"], pair)
    return {"star": star, "comp": comp, "fused": datas[star], "best": datas[comp],
            "sha": {d.meta["gallery"]["sha1"] for d in datas.values()}}


def gains(f: dict, names) -> dict:
    """g per query: AUC and each level, aligned on `names`."""
    out = {"auc": fs.align_by_name(names, f["fused"].names, f["fused"].auc)
           - fs.align_by_name(names, f["best"].names, f["best"].auc)}
    for x in FRACTIONS:
        out[x] = (fs.align_by_name(names, f["fused"].names, f["fused"].per_fraction[x])
                  - fs.align_by_name(names, f["best"].names, f["best"].per_fraction[x]))
    return out


def twins(fdir: Path, split: str, star: float, names) -> np.ndarray:
    """Gallery plans in the exact topology class of each complete query (axis 1 of `num_relevant`)."""
    z = np.load(fused_full_path(fdir, star, split), allow_pickle=True)
    axes = json.loads(str(z["meta"]))["axes"]
    return fs.align_by_name(names, z["names"], z["num_relevant"][axes.index("topology")])


def n_rooms(names) -> np.ndarray:
    from src.data.rplan_metadata import load_metadata
    return np.asarray([len(load_metadata(str(n)).room_types) for n in names], dtype=float)


def stat(x: np.ndarray) -> tuple[float, float, float]:
    lo, hi = bootstrap_ci(np.asarray(x, dtype=float))
    return float(np.mean(x)), lo, hi


def analyse(split: str) -> list[dict]:
    fus = {k: load_fusion(d, split, pair) for k, (d, pair) in fusion_dirs(split).items()}
    if len(set().union(*(f["sha"] for f in fus.values()))) != 1:
        raise ValueError("the fusions use different galleries")
    names = fus["head"]["fused"].names
    g = {k: gains(f, names) for k, f in fus.items()}
    # best branch = W s42: alpha 0 for the vision fusions, alpha 1 for W + sage/comb
    for k, want in (*((v, 0.0) for v in VISIONS), ("crossenc", 1.0)):
        if fus[k]["comp"] != want:
            raise ValueError(f"{k}: best branch alpha={fus[k]['comp']:g}, expected {want:g} (W s42)")
    w_auc = fs.align_by_name(names, fus["head"]["best"].names, fus["head"]["best"].auc)
    tw = twins(fusion_dirs(split)["head"][0], split, fus["head"]["star"], names)
    rooms = n_rooms(names)
    ctrl = max(CONTROLS["names"], key=lambda c: float(np.mean(g[c]["auc"])))

    groups = [("all", "all", np.ones(len(names), bool))]
    groups += [("twins", lab, (tw >= lo) & (tw <= hi)) for lo, hi, lab in TWIN_BINS]
    groups += [("graph_ok", lab, m) for lab, m in (("W perfect", w_auc >= 1 - 1e-9), ("W not perfect", w_auc < 1 - 1e-9))]
    groups += [("rooms", lab, (rooms >= lo) & (rooms <= hi)) for lo, hi, lab in ROOM_BINS]

    rows = []
    for v in VISIONS:
        for level in ("auc", *FRACTIONS):
            sel_groups = groups if level == "auc" else groups[:1]       # per-level rows: all queries only
            for kind, lab, m in sel_groups:
                if not m.any():
                    continue
                c_cg = "crossenc" if kind == "graph_ok" else ctrl
                r = {"split": split, "vision": v, "level": "AUC" if level == "auc" else f"f={level:g}",
                     "group": kind, "bin": lab, "n": int(m.sum()), "control_cg": c_cg}
                gf = g[v][level][m]
                r["g_fusion"], r["g_fusion_lo"], r["g_fusion_hi"] = stat(gf)
                for c in CONTROLS["names"]:
                    r[f"g_{c}"] = float(np.mean(g[c][level][m]))
                    r[f"D_{c}"], r[f"D_{c}_lo"], r[f"D_{c}_hi"] = stat(gf - g[c][level][m])
                r["CG"], r["CG_lo"], r["CG_hi"] = r[f"D_{c_cg}"], r[f"D_{c_cg}_lo"], r[f"D_{c_cg}_hi"]
                r["W_auc_mean"] = float(np.mean(w_auc[m]))
                rows.append(r)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--out-dir", default="results/csv_complementarity")
    p.add_argument("--splits", nargs="+", default=["valid", "test"])
    a = p.parse_args()
    out = Path(a.out_dir) / "complementarity_by_query.csv"
    if out.exists():
        raise SystemExit(f"!! {out} esiste gia': niente sovrascritture")
    rows = [r for s in a.splits for r in analyse(s)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    srcs = sorted({str(d) for s in a.splits for d, _ in fusion_dirs(s).values()})
    (out.parent / "_sources.txt").write_text(
        "complementarity_by_query.csv — python -m src.evaluation.complementarity_by_query\n"
        "per-query files read (seed 42, weights from each select json):\n" + "\n".join(f"  {s}" for s in srcs) + "\n")
    print(f"[complementarity_by_query] scritto {out} ({len(rows)} righe)")


if __name__ == "__main__":
    main()
