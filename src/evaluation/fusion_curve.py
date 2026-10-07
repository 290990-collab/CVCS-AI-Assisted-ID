"""Exploratory curve: fusion gain as a function of graph strength.

The 37 graph configs (seed 42, valid, already evaluated) are each fused with two visions:
  - "frozen": `pespatial/gem/whiten`, as the main fusion;
  - "head":   vision with the frozen head + whitening fit on train.
Fusion via `late_fusion.run` (for "head" through `fusion_head.head_patch`), reduced grid alpha in {0, 0.2, ..., 1},
removed rooms only.

Per (vision, config): graph strength = AUC at alpha=0; alpha* by the central-tie rule on the reduced grid;
gain = AUC(alpha*) - AUC(better branch), paired bootstrap CI. Changes nothing in W, the main fusion or the test.

`--seed S` runs the same curve on the other training seeds (damage seed = S). Graph inputs via
`round2.graph_source` (W re-evaluated, other 36 configs from stage 1); vision frozen / head of replica S.
Outputs under curve/s<S>/. Gate: results/final_pipeline/ROUND2_PREREGISTERED.

Usage (CPU):
    python -m src.evaluation.fusion_curve list
    python -m src.evaluation.fusion_curve run --index 0        # 0..73 (SLURM array)
    python -m src.evaluation.fusion_curve summary              # after all 74 runs
    python -m src.evaluation.fusion_curve run --index 0 --seed 100042     # second round
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from src.evaluation import fusion_head as fh
from src.evaluation import fusion_select as fs
from src.evaluation import late_fusion as lf
from src.evaluation import graph_config_select as rs
from src.evaluation.robustness_auc import compare_auc, load_auc
from src.graph import final_graph_configs as rc

SEED = 42
SPLIT = "valid"
VISIONS = ("frozen", "head")
CURVE_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
FRACTIONS = (0.25, 0.5, 0.75)   # AUC fractions
MAX_K = 100
RULE = "status.md §59.E (ESPLORATIVA)"


def root(smoke: bool = False, seed: int = SEED) -> Path:
    base = rc.root(smoke) / "curve"
    return base if int(seed) == SEED else base / f"s{int(seed)}"


def configs() -> list[tuple[str, str]]:
    """The 37 graph configs: per encoder the stage-1 table, then `comb`."""
    return [(enc, cfg) for enc in rc.ENCODERS for cfg in (*rc.stage1_configs(enc), rc.COMB)]


def tasks() -> list[tuple[str, str, str]]:
    """(vision, encoder, cfg); index = position (SLURM array 0..73)."""
    return [(v, enc, cfg) for v in VISIONS for enc, cfg in configs()]


def paths(vision: str, encoder: str, cfg: str, smoke: bool = False, seed: int = SEED) -> dict:
    seed = int(seed)
    g = rc.run_paths(encoder, cfg, seed, SPLIT, smoke)
    if seed == SEED:
        gq, gp = f"{g['QV_DIR']}/{g['TAG']}", f"{g['PQ_DIR']}/{g['TAG']}"
    else:
        if smoke:
            raise ValueError("second-round curve: no smoke mode")
        from src.evaluation import multiseed_runs as round2
        gq, gp = round2.graph_source(SPLIT, encoder, cfg, seed, seed)
    if vision == "frozen":
        vq, vp = rs.vision_prefixes(seed, SPLIT)
    elif vision == "head":
        vh = fh.vision_head_dirs(seed, SPLIT, smoke)
        vq, vp = f"{vh['QV_DIR']}/{fh.VISION_HEAD_TAG}", f"{vh['PQ_DIR']}/{fh.VISION_HEAD_TAG}"
    else:
        raise ValueError(f"unknown vision {vision!r} (expected one of {VISIONS})")
    return {"GRAPH_QVEC": gq, "GRAPH_PQ": gp, "VISION_QVEC": vq, "VISION_PQ": vp,
            "OUT": str(root(smoke, seed) / vision / f"{g['KEY']}_{cfg}" / f"fusion_{SPLIT}")}


def lf_argv(p: dict) -> list[str]:
    return ["run", "--split", SPLIT, "--gallery-names", "results/shared_gallery.json",
            "--vision-qvec", p["VISION_QVEC"], "--graph-qvec", p["GRAPH_QVEC"],
            "--fractions", *[f"{f:g}" for f in FRACTIONS],
            "--alphas", *[f"{a:g}" for a in CURVE_ALPHAS],
            "--modes", "partial", "--out", p["OUT"]]


def run(index: int, smoke: bool = False, seed: int = SEED) -> None:
    if int(seed) != SEED:
        from src.evaluation import multiseed_runs as round2
        reasons = round2.gate(SPLIT)
        if reasons:
            raise SystemExit("!! " + "; ".join(reasons))
    vision, encoder, cfg = tasks()[index]
    p = paths(vision, encoder, cfg, smoke, seed)
    out = Path(p["OUT"])
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"!! {out} non e' vuota: niente sovrascritture")
    print(f"[fusion_curve] {index}: {vision} · {encoder} {cfg} · seed {seed} -> {out}")
    args = lf.parse_args(lf_argv(p))
    if vision == "head":
        fh.check_query_final(args.vision_qvec, args.split, args.fractions)
        with fh.head_patch():
            lf.run(args)
    else:
        lf.run(args)


# --- summary ---

def _branch_check(fused, own) -> dict:
    """Per fraction: self_rr of a fusion endpoint vs the branch's own per-query."""
    out = {}
    for f in fused.per_fraction:
        a = fs.align_by_name(own.names, fused.names, fused.per_fraction[f])
        out[f"{f:g}"] = fs.c3_stats(a, own.per_fraction[f], MAX_K)
    return out


def one(vision: str, encoder: str, cfg: str, smoke: bool = False, seed: int = SEED) -> dict:
    p = paths(vision, encoder, cfg, smoke, seed)
    datas = {a: load_auc(lf.fusion_prefix(p["OUT"], a), SPLIT, strategy=lf.FUSED_STRATEGY)
             for a in CURVE_ALPHAS}
    names = datas[0.0].names
    for a, d in datas.items():
        if not np.array_equal(d.names, names):
            raise ValueError(f"{p['OUT']} alpha={a:g}: query names differ from alpha=0")
    means = {a: d.mean for a, d in datas.items()}
    best = max(CURVE_ALPHAS, key=lambda a: means[a])
    ties = fs.ties_from_comparisons({a: compare_auc(datas[best], datas[a]) for a in CURVE_ALPHAS})
    star = fs.central_alpha(ties, means)
    ref = fs.reference_alpha(means)
    graph = load_auc(p["GRAPH_PQ"], SPLIT, strategy=lf.GRAPH_STRATEGY)
    vis = load_auc(p["VISION_PQ"], SPLIT, strategy=lf.VISION_STRATEGY)
    return {
        "vision": vision, "encoder": encoder, "cfg": cfg, "fusion_dir": p["OUT"],
        "n": int(len(names)), "auc_means": {f"{a:g}": m for a, m in means.items()},
        "graph_auc": means[0.0], "vision_auc": means[1.0],
        "best": best, "ties": ties, "alpha_star": star, "reference_alpha": ref,
        "gain": compare_auc(datas[star], datas[ref]),
        "gain_best": compare_auc(datas[best], datas[ref]),
        "check_alpha0_vs_graph": _branch_check(datas[0.0], graph),
        "check_alpha1_vs_vision": _branch_check(datas[1.0], vis),
        "_per_query_gain": (names, datas[star].auc - datas[ref].auc),
    }


def summary(smoke: bool = False, seed: int = SEED) -> dict:
    rows, missing = [], []
    for v, enc, cfg in tasks():
        out = Path(paths(v, enc, cfg, smoke, seed)["OUT"])
        if any(len(list(out.glob(f"fusion_a{a:g}_partial-*_{SPLIT}.npz"))) != len(FRACTIONS)
               for a in CURVE_ALPHAS):
            missing.append(f"{v}/{enc}/{cfg}")
            continue
        rows.append(one(v, enc, cfg, smoke, seed))
    if missing:
        raise SystemExit(f"!! fusioni mancanti ({len(missing)}): {missing[:5]} ...")
    by = {(r["vision"], r["encoder"], r["cfg"]): r for r in rows}
    curves = {}
    for v in VISIONS:
        rv = [r for r in rows if r["vision"] == v]
        curves[v] = {"spearman_gain_vs_graph_auc": fs.spearman([r["graph_auc"] for r in rv],
                                                               [r["gain"]["delta"] for r in rv]),
                     "n_configs": len(rv)}
    head_minus_frozen = {}
    for enc, cfg in configs():
        fr, hd = by[("frozen", enc, cfg)], by[("head", enc, cfg)]
        n_f, g_f = fr["_per_query_gain"]
        n_h, g_h = hd["_per_query_gain"]
        d = fs.align_by_name(n_f, n_h, g_h) - g_f
        head_minus_frozen[f"{rc.encoder_key(enc)}_{cfg}"] = fs._mean_ci(d)
    for r in rows:
        r.pop("_per_query_gain")
    res = {"rule": RULE if int(seed) == SEED else "status.md §64 D (secondo giro)", "seed": int(seed),
           "split": SPLIT, "alphas": list(CURVE_ALPHAS),
           "written": datetime.now().isoformat(timespec="seconds"),
           "rows": rows, "curves": curves, "gain_head_minus_frozen": head_minus_frozen}
    return res


def preflight(smoke: bool = False, seed: int = SEED) -> None:
    """Check every graph qvec is present and its gallery sha1 unchanged."""
    shared = lf.load_shared_names("results/shared_gallery.json")
    bad = []
    for enc, cfg in configs():
        p = paths("frozen", enc, cfg, smoke, seed)
        try:
            gq = lf.load_branch_qvecs(p["GRAPH_QVEC"], SPLIT, FRACTIONS, lf.GRAPH_STRATEGY)
            ref = gq[FRACTIONS[0]]
            lf.load_graph_gallery(ref.meta["gallery_vectors"]["path"], shared,
                                  ref.meta["gallery_vectors"]["sha1"], f"graph {enc} {cfg}")
            load_auc(p["GRAPH_PQ"], SPLIT, strategy=lf.GRAPH_STRATEGY)
        except Exception as e:  # noqa: BLE001 - report every config, then fail
            bad.append(f"{enc} {cfg}: {e}")
    for v in VISIONS:
        p = paths(v, *configs()[0], smoke, seed)
        lf.load_branch_qvecs(p["VISION_QVEC"], SPLIT, FRACTIONS, lf.VISION_STRATEGY)
        load_auc(p["VISION_PQ"], SPLIT, strategy=lf.VISION_STRATEGY)
    if bad:
        raise SystemExit("!! preflight FALLITO:\n" + "\n".join(bad))
    print(f"[fusion_curve] preflight OK (seed {seed}): {len(configs())} graph (qvec, per-query, sha1 gallery) + 2 vision")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="fusion_curve")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    pf = sub.add_parser("preflight")
    r = sub.add_parser("run")
    r.add_argument("--index", type=int, required=True)
    s = sub.add_parser("summary")
    s.add_argument("--out-json", default=None)
    for x in (r, s, pf):
        x.add_argument("--smoke", action="store_true")
        x.add_argument("--seed", type=int, default=SEED, choices=list(rc.SEEDS))
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.cmd == "list":
        for i, t in enumerate(tasks()):
            print(i, *t)
    elif a.cmd == "preflight":
        preflight(a.smoke, a.seed)
    elif a.cmd == "run":
        run(a.index, a.smoke, a.seed)
    else:
        res = summary(a.smoke, a.seed)
        out = Path(a.out_json or root(a.smoke, a.seed) / f"summary_{SPLIT}.json")
        if out.exists():
            raise SystemExit(f"!! {out} esiste gia': niente sovrascritture")
        out.write_text(json.dumps(res, indent=2, sort_keys=True))
        print(f"[fusion_curve] scritto {out}")


if __name__ == "__main__":
    main()
