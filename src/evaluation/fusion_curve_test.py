"""Test-split fusion gain vs graph strength: reduced curve on five graphs (seed 42).

Graphs chosen on the valid by strength only: gcn/ref, gat/t05, gat/ref, gat/t01, W (gat/comb). Each fused with the
frozen vision (F) and the vision with head (H), weight fixed from the valid curve ({0, w, 1}, no grid on the test);
same fusion code as the valid curve (`fusion_curve`, i.e. `late_fusion.run`, via `fusion_head.head_patch` for H).

Gain g per query = AUC(w) - AUC(best branch of the pair, higher mean AUC).
Predictions (paired differences of per-query gains, CI 95% in the predicted direction):
  E1 (falling side, frozen)  g_F(gat/t05) > g_F(gat/ref) > g_F(gat/t01) > g_F(W)
  E2 (rising side, frozen)   g_F(gat/t05) > g_F(gcn/ref)
  E3 (head vs frozen)        g_H - g_F > 0 for gat/ref, gat/t01, W; < 0 for gcn/ref
  E4                         (g_H - g_F)(gat/ref) > (g_H - g_F)(W)
Shape: E1 and E2 all confirmed means inverted-U curve confirmed on the test.

`--seed S` runs the same 10 fusions on the other training seeds (damage seed = S), weights = alpha* of the valid
curve of the same seed (curve/s<S>/summary_valid.json). Graph inputs via `round2.graph_source`. Outputs under
curve_test/s<S>/. Gate: TEST_PREREGISTERED_R2.

Usage (CPU):
    python -m src.evaluation.fusion_curve_test list
    python -m src.evaluation.fusion_curve_test run --index 0        # 0..9 (SLURM array)
    python -m src.evaluation.fusion_curve_test summary              # after the 10 runs
    python -m src.evaluation.fusion_curve_test run --index 0 --seed 100042     # second round
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from src.evaluation import fusion_curve as fc
from src.evaluation import fusion_head as fh
from src.evaluation import fusion_select as fs
from src.evaluation import late_fusion as lf
from src.evaluation import graph_config_select as rs
from src.evaluation.robustness_auc import load_auc
from src.graph import final_graph_configs as rc

SEED = 42
SPLIT = "test"
VISIONS = fc.VISIONS
FRACTIONS = fc.FRACTIONS
RULE = "status.md §62 E"
W = ("gat", "comb")
# (encoder, cfg) -> vision weight (frozen, head), from the valid curve
PREREG = {
    ("gcn", "ref"): (0.8, 0.8),
    ("gat", "t05"): (0.6, 0.6),
    ("gat", "ref"): (0.6, 0.6),
    ("gat", "t01"): (0.4, 0.4),
    W: (0.4, 0.4),
}
CONFIGS = tuple(PREREG)
NAMES = {("gcn", "ref"): "gcn/ref", ("gat", "t05"): "gat/t05", ("gat", "ref"): "gat/ref",
         ("gat", "t01"): "gat/t01", W: "W"}


def root(smoke: bool = False, seed: int = SEED) -> Path:
    base = rc.root(smoke) / "curve_test"
    return base if int(seed) == SEED else base / f"s{int(seed)}"


def tasks() -> list[tuple[str, str, str]]:
    """(vision, encoder, cfg); index = position (SLURM array 0..9)."""
    return [(v, enc, cfg) for v in VISIONS for enc, cfg in CONFIGS]


def weight(vision: str, encoder: str, cfg: str) -> float:
    return PREREG[(encoder, cfg)][VISIONS.index(vision)]


def paths(vision: str, encoder: str, cfg: str, smoke: bool = False, seed: int = SEED) -> dict:
    seed = int(seed)
    g = rc.run_paths(encoder, cfg, seed, SPLIT, smoke)
    if seed == SEED:
        gq, gp = f"{g['QV_DIR']}/{g['TAG']}", f"{g['PQ_DIR']}/{g['TAG']}"
    else:
        if smoke:
            raise ValueError("second-round test curve: no smoke mode")
        from src.evaluation import multiseed_runs as round2
        gq, gp = round2.graph_source(SPLIT, encoder, cfg, seed, seed)
    if vision == "frozen":
        vq, vp = rs.vision_prefixes(seed, SPLIT)
    else:
        vh = fh.vision_head_dirs(seed, SPLIT, smoke)
        vq, vp = f"{vh['QV_DIR']}/{fh.VISION_HEAD_TAG}", f"{vh['PQ_DIR']}/{fh.VISION_HEAD_TAG}"
    out = root(smoke, seed) / vision / f"{g['KEY']}_{cfg}"
    return {"GRAPH_QVEC": gq, "GRAPH_PQ": gp,
            "VISION_QVEC": vq, "VISION_PQ": vp, "OUT": str(out / f"fusion_{SPLIT}"),
            "NEAR_TIES": {a: str(out / f"near_ties_a{a:g}_{SPLIT}.json") for a in (0.0, 1.0)},
            # near_ties --boundary: a query crossing the rank-100 border
            "NEAR_TIES_BOUNDARY": {a: str(out / f"near_ties_a{a:g}_confine_{SPLIT}.json") for a in (0.0, 1.0)}}


def check_weights(smoke: bool = False, seed: int = SEED) -> None:
    """Weights must equal the alpha* of the same-seed valid curve (summary_valid.json)."""
    rows = json.loads((fc.root(smoke, seed) / "summary_valid.json").read_text())["rows"]
    got = {(r["vision"], r["encoder"], r["cfg"]): float(r["alpha_star"]) for r in rows}
    for v, enc, cfg in tasks():
        if abs(got[(v, enc, cfg)] - weight(v, enc, cfg)) > 1e-9:
            raise ValueError(f"{v} {enc}/{cfg}: valid alpha* {got[(v, enc, cfg)]:g}, "
                             f"pre-registered {weight(v, enc, cfg):g}")


def lf_argv(p: dict, w: float) -> list[str]:
    return ["run", "--split", SPLIT, "--gallery-names", "results/shared_gallery.json",
            "--vision-qvec", p["VISION_QVEC"], "--graph-qvec", p["GRAPH_QVEC"],
            "--fractions", *[f"{f:g}" for f in FRACTIONS],
            "--alphas", *[f"{a:g}" for a in sorted({0.0, w, 1.0})],
            "--modes", "partial", "--out", p["OUT"]]


def run(index: int, smoke: bool = False, seed: int = SEED) -> None:
    vision, encoder, cfg = tasks()[index]
    if int(seed) == SEED:
        reasons = rs.gate("test", encoder, cfg, SEED, smoke)
    else:
        from src.evaluation import multiseed_runs as round2
        reasons = round2.gate(SPLIT)
    if reasons:
        raise SystemExit("!! test non consentito: " + "; ".join(reasons))
    check_weights(smoke, seed)
    p = paths(vision, encoder, cfg, smoke, seed)
    out = Path(p["OUT"])
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"!! {out} non e' vuota: niente sovrascritture")
    w = weight(vision, encoder, cfg)
    print(f"[fusion_curve_test] {index}: {vision} · {encoder} {cfg} · seed {seed} · peso {w:g} -> {out}")
    args = lf.parse_args(lf_argv(p, w))
    if vision == "head":
        fh.check_query_final(args.vision_qvec, args.split, args.fractions)
        with fh.head_patch():
            lf.run(args)
    else:
        lf.run(args)


# --- summary ---

def _endpoint_check(fused, own, near_ties_json, boundary_json=None) -> dict:
    """Endpoint check (alpha=0 vs graph, alpha=1 vs vision); a failure passes only if `near_ties` shows every differing query is a near-tie.

    The `near_ties --boundary` (rank-100 border) file is read only if the plain diagnosis exists and fails; both are recorded.
    """
    stats = fc._branch_check(fused, own)
    ok = fs.c3_pass(list(stats.values()))
    res = {"per_fraction": stats, "c3_pass": ok, "near_ties": None, "near_ties_boundary": None}
    if not ok:
        nt = Path(near_ties_json)
        if nt.exists():
            res["near_ties"] = {"file": str(nt), "all_near_ties": bool(json.loads(nt.read_text())["all_near_ties"])}
        nb = Path(boundary_json) if boundary_json else None
        if res["near_ties"] and not res["near_ties"]["all_near_ties"] and nb and nb.exists():
            d = json.loads(nb.read_text())
            if not d.get("boundary"):
                raise ValueError(f"{nb}: not a --boundary diagnosis")
            res["near_ties_boundary"] = {"file": str(nb), "all_near_ties": bool(d["all_near_ties"])}
    res["ok"] = ok or any(bool(res[k] and res[k]["all_near_ties"]) for k in ("near_ties", "near_ties_boundary"))
    return res


def one(vision: str, encoder: str, cfg: str, smoke: bool = False, seed: int = SEED) -> dict:
    p = paths(vision, encoder, cfg, smoke, seed)
    w = weight(vision, encoder, cfg)
    datas = {a: load_auc(lf.fusion_prefix(p["OUT"], a), SPLIT, strategy=lf.FUSED_STRATEGY)
             for a in sorted({0.0, w, 1.0})}
    names = datas[0.0].names
    for a, d in datas.items():
        if not np.array_equal(d.names, names):
            raise ValueError(f"{p['OUT']} alpha={a:g}: query names differ from alpha=0")
    means = {a: d.mean for a, d in datas.items()}
    comp = fs.reference_alpha(means)
    graph = load_auc(p["GRAPH_PQ"], SPLIT, strategy=lf.GRAPH_STRATEGY)
    vis = load_auc(p["VISION_PQ"], SPLIT, strategy=lf.VISION_STRATEGY)
    checks = {"alpha0_vs_graph": _endpoint_check(datas[0.0], graph, p["NEAR_TIES"][0.0],
                                                 p["NEAR_TIES_BOUNDARY"][0.0]),
              "alpha1_vs_vision": _endpoint_check(datas[1.0], vis, p["NEAR_TIES"][1.0],
                                                  p["NEAR_TIES_BOUNDARY"][1.0])}
    return {"vision": vision, "encoder": encoder, "cfg": cfg, "name": NAMES[(encoder, cfg)],
            "fusion_dir": p["OUT"], "weight": w, "n": int(len(names)),
            "auc_means": {f"{a:g}": m for a, m in means.items()},
            "graph_auc": means[0.0], "vision_auc": means[1.0], "best_component": comp,
            "gain": fs._mean_ci(datas[w].auc - datas[comp].auc), "checks": checks,
            "_names": names, "_g": datas[w].auc - datas[comp].auc}


def prediction(diff: np.ndarray, sign: int, label: str) -> dict:
    """Paired difference with CI; confirmed iff the CI lies entirely on the predicted side."""
    ci = fs._mean_ci(diff)
    if (sign > 0 and ci["ci_lo"] > 0) or (sign < 0 and ci["ci_hi"] < 0):
        esito = "confermata"
    elif (sign > 0 and ci["ci_hi"] < 0) or (sign < 0 and ci["ci_lo"] > 0):
        esito = "contraria"
    else:
        esito = "non confermata"
    return {"label": label, "predicted_sign": "+" if sign > 0 else "-", **ci, "esito": esito}


def predictions(rows: dict) -> dict:
    """E1-E4 on the per-query gains; `rows[(vision, (enc, cfg))]` -> one()."""
    ref_names = rows[("frozen", W)]["_names"]
    g = {k: fs.align_by_name(ref_names, r["_names"], r["_g"]) for k, r in rows.items()}
    gF = {c: g[("frozen", c)] for c in CONFIGS}
    gH = {c: g[("head", c)] for c in CONFIGS}
    t05, ref, t01, gcn = ("gat", "t05"), ("gat", "ref"), ("gat", "t01"), ("gcn", "ref")
    e1 = [prediction(gF[a] - gF[b], +1, f"g_F({NAMES[a]}) > g_F({NAMES[b]})")
          for a, b in ((t05, ref), (ref, t01), (t01, W))]
    e2 = [prediction(gF[t05] - gF[gcn], +1, "g_F(gat/t05) > g_F(gcn/ref)")]
    e3 = [prediction(gH[c] - gF[c], s, f"g_H - g_F {'>' if s > 0 else '<'} 0 per {NAMES[c]}")
          for c, s in ((ref, +1), (t01, +1), (W, +1), (gcn, -1))]
    e4 = [prediction((gH[ref] - gF[ref]) - (gH[W] - gF[W]), +1, "(g_H - g_F)(gat/ref) > (g_H - g_F)(W)")]
    shape_ok = all(p["esito"] == "confermata" for p in e1 + e2)
    held = [p["label"] for p in e1 + e2 if p["esito"] == "confermata"]
    return {"E1_falling_frozen": e1, "E2_rising_frozen": e2, "E3_head_vs_frozen": e3, "E4": e4,
            "shape": ("curva a U rovesciata confermata sul test" if shape_ok else
                      "forma non confermata per intero; reggono: " + ("; ".join(held) or "nessuna")),
            "descriptive_head_t05": fs._mean_ci(gH[t05] - gF[t05])}


def summary(smoke: bool = False, seed: int = SEED) -> dict:
    check_weights(smoke, seed)
    rows, missing = {}, []
    for v, enc, cfg in tasks():
        out = Path(paths(v, enc, cfg, smoke, seed)["OUT"])
        w = weight(v, enc, cfg)
        if any(len(list(out.glob(f"fusion_a{a:g}_partial-*_{SPLIT}.npz"))) != len(FRACTIONS)
               for a in sorted({0.0, w, 1.0})):
            missing.append(f"{v}/{enc}/{cfg}")
            continue
        rows[(v, (enc, cfg))] = one(v, enc, cfg, smoke, seed)
    if missing:
        raise SystemExit(f"!! fusioni mancanti ({len(missing)}): {missing}")
    bad = [f"{v}/{NAMES[c]} {k}" for (v, c), r in rows.items() for k, ch in r["checks"].items() if not ch["ok"]]
    if bad:
        for b in bad:
            print(f"!! controllo fallito: {b} (diagnosi: python -m src.evaluation.near_ties ...)")
        raise SystemExit("!! controlli falliti: nessun risultato scritto, prima la diagnosi")
    pred = predictions(rows)
    out_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows.values()]
    return {"rule": RULE if int(seed) == SEED else "status.md §65 D (secondo giro)", "seed": int(seed),
            "split": SPLIT, "fractions": list(FRACTIONS),
            "weights": {f"{v}/{NAMES[c]}": weight(v, *c) for v in VISIONS for c in CONFIGS},
            "written": datetime.now().isoformat(timespec="seconds"),
            "rows": out_rows, "predictions": pred}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="fusion_curve_test")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("--index", type=int, required=True)
    s = sub.add_parser("summary")
    for x in (r, s):
        x.add_argument("--smoke", action="store_true")
        x.add_argument("--seed", type=int, default=SEED, choices=list(rc.SEEDS))
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.cmd == "list":
        for i, (v, enc, cfg) in enumerate(tasks()):
            print(i, v, enc, cfg, weight(v, enc, cfg), paths(v, enc, cfg)["OUT"])
    elif a.cmd == "run":
        run(a.index, a.smoke, a.seed)
    else:
        out = root(a.smoke, a.seed) / f"summary_{SPLIT}.json"
        if out.exists():
            raise SystemExit(f"!! {out} esiste gia': niente sovrascritture")
        res = summary(a.smoke, a.seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, sort_keys=True))
        p = res["predictions"]
        for key in ("E1_falling_frozen", "E2_rising_frozen", "E3_head_vs_frozen", "E4"):
            for x in p[key]:
                print(f"  {key}: {x['label']}: {x['mean']:+.4f} [{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}] → {x['esito']}")
        print(f"FORMA (§62 E): {p['shape']}")
        print(f"[fusion_curve_test] scritto {out}")


if __name__ == "__main__":
    main()
