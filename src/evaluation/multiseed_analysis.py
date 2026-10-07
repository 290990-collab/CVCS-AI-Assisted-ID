"""Second-round analysis of the graph study: items A-E, H, I with the between-training noise rule G.

Inputs (all on disk, read only):
  - main fusions of the 4 replicas: fusion/s<S> (frozen vision) and fusion_head/s<S> (vision with head),
    weight from their select json;
  - graph + graph controls: seed 42 controls/{replica,crossenc}, the other replicas and C
    round2/controls/<name>, weight from their select json;
  - the curve on 4 seeds: curve/[s<S>/]summary_valid.json (alpha* per config) and its fusion folders.

Per query g = AUC(fusion at its weight) - AUC(best branch of its pair); D = g(true fusion) - g(control).
Pooled over replicas: the 2000 queries are the same in every replica, so a per-query quantity is averaged
over the 4 replicas, then bootstrapped (B = 10000, seed 0, CI 95%).

  A  frozen vision + W vs the two controls, per replica and pooled; equivalence margin DELTA: CI inside
     [-DELTA, +DELTA] -> equivalent, CI > 0 -> beyond, CI < 0 -> below, else inconclusive (checked in this order).
  B  vision with head + W, same controls: both CI > 0 -> confirmed, both < 0 -> refuted, else not
     distinguishable from (named); number of replicas with CI > 0.
  C  gat/t05 and gat/ref (seed 42) fused with each vision (curve fusion, alpha* of the valid curve) vs
     their second copy and the same change on SAGE; decisive: gat/t05 + frozen and gat/ref + head.
  D  the curve pooled over 4 seeds: per (vision, config) mean gain; Spearman gain vs graph strength;
     the E1-E4 predictions on the pooled gains and per seed; the weights the test will use (alpha* per seed).
  E  W vs sage/comb: Delta AUC per replica (same removed rooms) and pooled.
  G  an effect with 4 replicas is above the between-training noise iff same sign in 4/4 replicas and
     |mean| > 2 x sd (ddof=1) of the 4 per-replica means.
  H  head v2: robustness on the three damages vs frozen and current head, fusion with W vs the
     current head's (per replica and pooled), D vs the controls of A (as B).
  I  the 6 geometry weightings on W, frozen vision and vision with head, full plan, seed 42 (descriptive).
     On the test the vision with head has no full-plan evaluation yet.

Test: the same items with every weight fixed from the valid (checked against the values in TEST_WEIGHTS:
the code stops if one differs), plus the stated predictions, each reported on its own. Seed-42 controls of
A/B on the test = controls_test/; C uses the seed-42 test curve; D the test curve of the 4 seeds; I adds
head v2 (descriptive). Refuses to read if a check of the fusions is missing or failed.

Usage (CPU, minutes):
    python -m src.evaluation.multiseed_analysis valid              # -> results/final_pipeline/round2/analysis_valid.json
    python -m src.evaluation.multiseed_analysis valid --geometry   # -> results/final_pipeline/round2/geometry_valid.json
    python -m src.evaluation.multiseed_analysis valid --head-v2    # -> results/final_pipeline/round2/head_v2_valid.json
    python -m src.evaluation.multiseed_analysis test               # -> results/final_pipeline/round2/analysis_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from src.evaluation import fusion_curve as fc
from src.evaluation import fusion_curve_test as fct
from src.evaluation import fusion_head as fh
from src.evaluation import fusion_select as fs
from src.evaluation import late_fusion as lf
from src.evaluation import graph_config_select as rs
from src.evaluation import multiseed_runs as r2
from src.evaluation.robustness_auc import load_auc
from src.graph import final_graph_configs as rc

RULE = "status.md §64"
DELTA = 0.005            # equivalence margin of A
SEEDS = tuple(rc.SEEDS)  # 42, 100042, 200042, 300042
CONTROLS = ("replica", "crossenc")
CONTROL_NAMES = {"replica": "la seconda copia", "crossenc": "l'altro encoder (SAGE)"}
# test weights, fixed from the valid; the reading stops if a fusion on disk used another one
TEST_WEIGHTS = {
    "frozen": {42: 0.4, 100042: 0.4, 200042: 0.4, 300042: 0.4},
    "head": {42: 0.3, 100042: 0.4, 200042: 0.4, 300042: 0.4},
    "v2": {42: 0.6, 100042: 0.6, 200042: 0.5, 300042: 0.5},
    "control": {"replica": {42: 0.5, 100042: 0.5, 200042: 0.6, 300042: 0.5},
                "crossenc": {42: 0.6, 100042: 0.5, 200042: 0.6, 300042: 0.6}},
    "mid": {("gat_t05", "replica"): 0.6, ("gat_ref", "replica"): 0.4,
            ("gat_t05", "crossenc"): 0.4, ("gat_ref", "crossenc"): 0.4},
}
P_H2_TOL = 0.02          # |test - valid| of the head-v2 fusion, per replica


def check_weight(label: str, got: float, expected: float) -> None:
    if abs(float(got) - float(expected)) > 1e-9:
        raise ValueError(f"{label}: peso {got:g} sul disco, {expected:g} pre-registrato (status.md §65)")


# --- one fusion as per-query gains ---

def _star(select_json, fdir, split: str) -> float:
    sel = json.loads(Path(select_json).read_text())
    if sel.get("split") != split or Path(str(sel.get("fusion_dir", ""))).resolve() != Path(fdir).resolve():
        raise ValueError(f"{select_json} does not describe {fdir} on the {split}")
    return float(sel["alpha_star"])


def gains(fdir, star: float, pair: str, split: str) -> dict:
    """Per-query g of one fusion at weight `star` (best branch = higher mean AUC)."""
    datas, comp = fs._fusion_endpoints(fdir, split, star, fs.pair_spec(pair)["fused"], pair)
    sha = {d.meta["gallery"]["sha1"] for d in datas.values()}
    if len(sha) != 1:
        raise ValueError(f"{fdir}: endpoints with different galleries")
    d = datas[float(star)]
    return {"dir": str(fdir), "weight": float(star), "best_component": comp, "names": d.names,
            "g": d.auc - datas[comp].auc, "auc": d.mean, "auc_q": d.auc, "sha1": sha.pop()}


def main_fusion(seed: int, vision: str, split: str = "valid") -> dict:
    """W + vision of replica `seed`: frozen (fusion/), head (fusion_head/), head v2 (fusion_head_v2/)."""
    if vision == "v2":
        from src.evaluation import fusion_head_v2 as fv
        p = fv.head_paths(seed, split)
    else:
        p = fh.head_paths(seed, split) if vision == "head" else rs.fusion_paths(seed, split)
    sel = p["SELECT_JSON"]
    fdir = p["FUSION_DIR"]
    return gains(fdir, _star(sel, fdir, split), "vision-graph", split)


def control(seed: int, kind: str, split: str = "valid") -> dict:
    """graph + graph control of W for replica `seed`."""
    if int(seed) == 42:
        d = rc.root() / ("controls" if split == "valid" else "controls_test") / kind
        fdir, sel = d / f"fusion_{split}", d / f"select_{split}.json"
    else:
        names = [c["name"] for c in r2.controls(split)]
        p = r2.control_paths(split, names.index(f"A_{kind}_s{int(seed)}"))
        fdir, sel = p["DIR"], p["SELECT"]
    return gains(fdir, _star(sel, fdir, split), "graph-graph", split)


def mid_control(name: str, kind: str, split: str = "valid") -> dict:
    names = [c["name"] for c in r2.controls(split)]
    p = r2.control_paths(split, names.index(f"C_{kind}_{name}"))
    return gains(p["DIR"], _star(p["SELECT"], p["DIR"], split), "graph-graph", split)


def curve_summary(seed: int, split: str = "valid") -> dict:
    root = fc.root(seed=seed) if split == "valid" else fct.root(seed=seed)
    return json.loads((root / f"summary_{split}.json").read_text())


def curve_row(summary: dict, vision: str, enc: str, cfg: str) -> dict:
    for r in summary["rows"]:
        if (r["vision"], r["encoder"], r["cfg"]) == (vision, enc, cfg):
            return r
    raise KeyError(f"{vision}/{enc}/{cfg} not in the curve summary")


def curve_fusion(seed: int, vision: str, enc: str, cfg: str, summary: dict | None = None,
                 split: str = "valid") -> dict:
    row = curve_row(summary or curve_summary(seed, split), vision, enc, cfg)
    if split == "valid":
        return gains(row["fusion_dir"], float(row["alpha_star"]), "vision-graph", split)
    w = fct.weight(vision, enc, cfg)                       # pre-registered
    if abs(float(row["weight"]) - w) > 1e-9:
        raise ValueError(f"curve s{seed} {vision}/{enc}/{cfg}: weight {row['weight']} != pre-registered {w}")
    return gains(row["fusion_dir"], w, "vision-graph", split)


# --- rules ---

def aligned(ref_names, x: dict) -> np.ndarray:
    return fs.align_by_name(ref_names, x["names"], x["g"])


def pooled(per_seed: dict, ref_names) -> np.ndarray:
    """Per-query mean over the replicas of {seed: (names, values)}."""
    return np.mean([fs.align_by_name(ref_names, n, v) for n, v in per_seed.values()], axis=0)


def noise_rule(means) -> dict:
    """G: same sign in every replica and |mean| > 2 sd (ddof=1) of the per-replica means."""
    x = np.asarray(list(means), dtype=float)
    m, sd = float(x.mean()), float(x.std(ddof=1))
    same = bool(np.all(x > 0) or np.all(x < 0))
    above = same and abs(m) > 2 * sd
    return {"per_replica": [float(v) for v in x], "mean": m, "sd": sd, "same_sign": same,
            "n_positive": int((x > 0).sum()), "verdict": ("sopra il rumore fra training" if above
                                                          else "entro il rumore fra training")}


def equivalence(ci: dict, delta: float = DELTA) -> str:
    """A: the four outcomes, checked in order."""
    if -delta <= ci["ci_lo"] and ci["ci_hi"] <= delta:
        return "equivalente all'effetto d'insieme"
    if ci["ci_lo"] > 0:
        return "oltre l'effetto d'insieme"
    if ci["ci_hi"] < 0:
        return "sotto l'effetto d'insieme"
    return "non concludente"


def beyond(d_rep: dict, d_enc: dict, where: str = "su 4 repliche") -> str:
    """B and C: both CI > 0 / both CI < 0 / otherwise not distinguishable from the named controls."""
    pos = {"replica": d_rep["ci_lo"] > 0, "crossenc": d_enc["ci_lo"] > 0}
    neg = {"replica": d_rep["ci_hi"] < 0, "crossenc": d_enc["ci_hi"] < 0}
    if all(pos.values()):
        return f"confermato {where}".strip()
    if all(neg.values()):
        return "smentito"
    return "non distinguibile da " + " e ".join(CONTROL_NAMES[k] for k in CONTROLS if not pos[k])


# --- items ---

def item_vs_controls(vision: str, split: str = "valid") -> dict:
    """A (frozen) or B (head): D per replica and pooled, for both controls."""
    per_seed, sha, ref_names = {}, set(), None
    raw = {k: {} for k in CONTROLS}
    for s in SEEDS:
        F = main_fusion(s, vision, split)
        if split == "test":
            check_weight(f"fusione {vision} s{s}", F["weight"], TEST_WEIGHTS[vision][s])
        ref_names = F["names"] if ref_names is None else ref_names
        row = {"fusion": {k: F[k] for k in ("dir", "weight", "best_component", "auc")},
               "g_F": fs._mean_ci(F["g"])}
        sha.add(F["sha1"])
        for k in CONTROLS:
            C = control(s, k, split)
            if split == "test":
                check_weight(f"controllo {k} s{s}", C["weight"], TEST_WEIGHTS["control"][k][s])
            sha.add(C["sha1"])
            d = F["g"] - fs.align_by_name(F["names"], C["names"], C["g"])
            raw[k][s] = (F["names"], d)
            row[k] = {"control": {kk: C[kk] for kk in ("dir", "weight", "best_component", "auc")},
                      "g_C": fs._mean_ci(C["g"]), "D": fs._mean_ci(d)}
        per_seed[s] = row
    if len(sha) != 1:
        raise ValueError(f"different galleries: {sha}")
    out = {"per_replica": {str(s): r for s, r in per_seed.items()}, "gallery_sha1": sha.pop()}
    for k in CONTROLS:
        D = fs._mean_ci(pooled(raw[k], ref_names))
        out[f"D_{k}_pooled"] = D
        out[f"G_{k}"] = noise_rule(per_seed[s][k]["D"]["mean"] for s in SEEDS)
        out[f"n_replicas_ci_pos_{k}"] = sum(per_seed[s][k]["D"]["ci_lo"] > 0 for s in SEEDS)
        if vision == "frozen":
            out[f"outcome_{k}"] = equivalence(D)
    if vision in ("head", "v2"):
        out["outcome"] = beyond(out["D_replica_pooled"], out["D_crossenc_pooled"])
    return out


def item_c(split: str = "valid") -> dict:
    """C: graphs of strength similar to the vision, seed 42, both visions."""
    summary = curve_summary(42, split)
    out = {}
    for name, (cfg, _) in r2.MID.items():
        ctrl = {k: mid_control(name, k, split) for k in CONTROLS}
        if split == "test":
            for k in CONTROLS:
                check_weight(f"controllo C {name} {k}", ctrl[k]["weight"], TEST_WEIGHTS["mid"][(name, k)])
        for vision in fc.VISIONS:
            F = curve_fusion(42, vision, *cfg, summary, split)
            row = {"fusion": {k: F[k] for k in ("dir", "weight", "best_component", "auc")},
                   "g_F": fs._mean_ci(F["g"]),
                   "decisive": (name, vision) in (("gat_t05", "frozen"), ("gat_ref", "head"))}
            Ds = {}
            for k in CONTROLS:
                C = ctrl[k]
                if C["sha1"] != F["sha1"]:
                    raise ValueError(f"C {name}/{vision}/{k}: different galleries")
                Ds[k] = fs._mean_ci(F["g"] - fs.align_by_name(F["names"], C["names"], C["g"]))
                row[k] = {"control": {kk: C[kk] for kk in ("dir", "weight", "best_component", "auc")},
                          "g_C": fs._mean_ci(C["g"]), "D": Ds[k]}
            row["outcome"] = beyond(Ds["replica"], Ds["crossenc"], where="")
            out[f"{name}+{vision}"] = row
    return out


def item_d(split: str = "valid") -> dict:
    """D: the curve pooled over 4 seeds (valid: 37 configs, descriptive; test: the 5 test configs, decides)."""
    sums = {s: curve_summary(s, split) for s in SEEDS}
    per = {}       # (vision, enc, cfg) -> {seed: gains}
    for v, enc, cfg in (fc.tasks() if split == "valid" else fct.tasks()):
        per[(v, enc, cfg)] = {s: curve_fusion(s, v, enc, cfg, sums[s], split) for s in SEEDS}
    ref_names = per[("frozen", *fct.W)][42]["names"]
    rows = []
    for (v, enc, cfg), by in per.items():
        g = pooled({s: (x["names"], x["g"]) for s, x in by.items()}, ref_names)
        graph = [curve_row(sums[s], v, enc, cfg)["graph_auc"] for s in SEEDS]
        rows.append({"vision": v, "encoder": enc, "cfg": cfg, "graph_auc_mean": float(np.mean(graph)),
                     "alpha_star": {str(s): by[s]["weight"] for s in SEEDS},
                     "gain_pooled": fs._mean_ci(g), "_g": g,
                     "gain_per_seed": {str(s): float(np.mean(by[s]["g"])) for s in SEEDS}})
    curves = {}
    for v in fc.VISIONS:
        rv = [r for r in rows if r["vision"] == v]
        curves[v] = {"spearman_pooled": fs.spearman([r["graph_auc_mean"] for r in rv],
                                                    [r["gain_pooled"]["mean"] for r in rv])}
        if split == "valid":
            curves[v]["spearman_per_seed"] = {str(s): sums[s]["curves"][v]["spearman_gain_vs_graph_auc"]
                                              for s in SEEDS}
    by_key = {(r["vision"], (r["encoder"], r["cfg"])): r for r in rows}
    pooled_pred = fct.predictions({(v, c): {"_names": ref_names, "_g": by_key[(v, c)]["_g"]}
                                   for v in fc.VISIONS for c in fct.CONFIGS})
    pooled_pred["shape"] = pooled_pred["shape"].replace(
        "sul test", "sul valid, media dei 4 seed" if split == "valid" else "sul test, media dei 4 seed")
    seed_pred, held = {}, {}
    for s in SEEDS:
        p = fct.predictions({(v, c): {"_names": per[(v, *c)][s]["names"], "_g": per[(v, *c)][s]["g"]}
                             for v in fc.VISIONS for c in fct.CONFIGS})
        seed_pred[str(s)] = {k: [x["esito"] for x in p[k]] for k in ("E1_falling_frozen", "E2_rising_frozen",
                                                                     "E3_head_vs_frozen", "E4")}
        for k in ("E1_falling_frozen", "E2_rising_frozen", "E3_head_vs_frozen", "E4"):
            for x in p[k]:
                held.setdefault(x["label"], 0)
                held[x["label"]] += x["esito"] == "confermata"
    e12 = {}
    for k in ("E1_falling_frozen", "E2_rising_frozen"):
        for i, x in enumerate(pooled_pred[k]):
            means = []
            for s in SEEDS:
                a, b = _pred_pair(k, i)
                means.append(float(np.mean(per[("frozen", *a)][s]["g"]) - np.mean(per[("frozen", *b)][s]["g"])))
            e12[x["label"]] = noise_rule(means)
    weights = {str(s): {f"{v}/{fct.NAMES[c]}": per[(v, *c)][s]["weight"] for v in fc.VISIONS for c in fct.CONFIGS}
               for s in SEEDS}
    for r in rows:
        r.pop("_g")
    return {"rows": rows, "curves": curves, "predictions_pooled": pooled_pred,
            "predictions_per_seed": seed_pred, "n_seeds_confirmed": held, "G_E1_E2": e12,
            ("weights_for_test" if split == "valid" else "weights_used"): weights}


def _pred_pair(key: str, i: int):
    t05, ref, t01, gcn = ("gat", "t05"), ("gat", "ref"), ("gat", "t01"), ("gcn", "ref")
    if key == "E1_falling_frozen":
        return ((t05, ref), (ref, t01), (t01, fct.W))[i]
    return (t05, gcn)


def item_e(split: str = "valid") -> dict:
    """E: W vs sage/comb, per replica (stage-1 evaluations, same removed rooms) and pooled."""
    raw, per = {}, {}
    for s in SEEDS:
        a = {}
        for key, (enc, cfg) in (("W", r2.W), ("E", r2.E)):
            if split == "valid":
                p = rc.run_paths(enc, cfg, s, split)
                prefix = f"{p['PQ_DIR']}/{p['TAG']}"
            else:   # test: sage/comb s100042-300042 and W s100042 from the second-round evals
                prefix = r2.graph_source(split, enc, cfg, s, s)[1]
            a[key] = load_auc(prefix, split, strategy=lf.GRAPH_STRATEGY)
        d = a["W"].auc - fs.align_by_name(a["W"].names, a["E"].names, a["E"].auc)
        raw[s] = (a["W"].names, d)
        per[str(s)] = {"auc_W": a["W"].mean, "auc_E": a["E"].mean, "delta": fs._mean_ci(d)}
    ref = raw[42][0]
    D = fs._mean_ci(pooled(raw, ref))
    verdict = ("W più robusto" if D["ci_lo"] > 0 else "sage/comb più robusto" if D["ci_hi"] < 0 else "pari")
    return {"per_replica": per, "delta_pooled": D, "outcome": verdict,
            "G": noise_rule(per[str(s)]["delta"]["mean"] for s in SEEDS)}


GEOMETRY_SYSTEMS = {   # full plan, seed 42; split -> label -> per-query file
    "valid": {"W": None,
              "vision_congelato": "results/perquery/vision_full_valid_B/vision_pespatial_gem_whiten-train_full_valid.npz",
              "vision_head": ("results/perquery/vision_full_valid_B/"
                              "vision_pespatial_gem_head-nowalls-conv+whiten-train_full_valid.npz")},
}
GEOMETRY_SYSTEMS["test"] = {
    "W": None,
    "vision_congelato": "results/perquery/vision_test_B/vision_pespatial_gem_whiten-train_full_test.npz",
    "vision_head": "results/vision_head_v1/perquery/test/vision_pespatial_gem_head-nowalls-conv+whiten-train_full_test.npz",
    "vision_head_v2": ("results/vision_head_v2/perquery/test/"
                       "vision_pespatial_gem_whiten-train+qhead-v2_full_test.npz")}        # descriptive
GEOMETRY_PAIRS = (("W", "vision_congelato"), ("W", "vision_head"), ("vision_head", "vision_congelato"))
GEOMETRY_PAIRS_V2 = (("W", "vision_head_v2"), ("vision_head_v2", "vision_congelato"))      # test, descriptive


def item_i(split: str = "valid", k: int = 10) -> dict:
    """I: the 6 geometry weightings on the final systems, full plan (descriptive)."""
    from src.data.rplan_metadata import load_metadata
    from src.evaluation import geometry_variants as gv
    from src.evaluation.perquery import load_perquery
    from src.evaluation.relevance import GalleryAxes
    shared = lf.load_shared_names("results/shared_gallery.json")
    axes = GalleryAxes([load_metadata(n) for n in shared])
    files = dict(GEOMETRY_SYSTEMS[split])
    p = rc.run_paths(*r2.W, 42, split)
    files["W"] = f"{p['PQ_DIR']}/{p['TAG']}_full_{split}.npz"
    data, check = {}, {}
    for label, path in files.items():
        d = load_perquery(path)
        gv.check_gallery(axes, d, label)
        if d.meta.get("gallery", {}).get("sha1") not in (None, json.loads(
                Path("results/shared_gallery.json").read_text())["sha1"]):
            raise ValueError(f"{label}: gallery sha1 differs from the shared gallery")
        data[label] = d
        check[label] = {"file": path, "recomputed_111": gv.ndcg_geometry_weighted(axes, d, gv.BASELINE_WEIGHTS, k),
                        "stored": gv.stored_geometry_ndcg(d, k)}
    pq = {(tuple(w), lab): gv.perquery_ndcg_geometry(axes, d, w, k)
          for w in gv.DEFAULT_WEIGHTS for lab, d in data.items()}
    means = {",".join(f"{x:g}" for x in w): {lab: float(pq[(tuple(w), lab)][1].mean()) for lab in data}
             for w in gv.DEFAULT_WEIGHTS}
    pairs = {}
    for a, b in GEOMETRY_PAIRS + (GEOMETRY_PAIRS_V2 if "vision_head_v2" in data else ()):
        rows = [{"weights": list(w), **gv.paired_delta(*pq[(tuple(w), a)], *pq[(tuple(w), b)])}
                for w in gv.DEFAULT_WEIGHTS]
        pairs[f"{a} vs {b}"] = {"rows": rows, "verdict": gv.claim_verdict(rows)}
    return {"rule": "status.md §64 I (criterio di §22, descrittivo)" + (", §65 P-I" if split == "test" else ""),
            "split": split, "k": k, "seed": 42,
            "written": datetime.now().isoformat(timespec="seconds"), "check_baseline": check,
            "ndcg_geometry_means": means, "pairs": pairs,
            "notes": ["righe non confrontabili fra pesature (ogni pesatura e' una GT diversa)",
                      "(0,0,1) e' la pesatura piu' circolare per il graph, (1,1,0) la meno"]}


V2_PREFIX = "vision_pespatial_gem_whiten-train+qhead-v2"
V2_DIR = Path("results/vision_head_v2/perquery")
DAMAGE_PREFIXES = {   # three-damage evaluations already on disk (valid, seed 42)
    "valid": {"vision_congelato": "results/perquery/vision_damage_valid_B/vision_pespatial_gem_whiten-train",
              "vision_head": "results/perquery/vision_damage_valid_B/vision_pespatial_gem_head-nowalls-conv+whiten-train"},
    "test": {"vision_congelato": "results/perquery/vision_test_B/vision_pespatial_gem_whiten-train",
             "vision_head": "results/vision_head_v1/perquery/test/vision_pespatial_gem_head-nowalls-conv+whiten-train"},
}
FULL_FILES = {
    "valid": {"vision_congelato": "results/perquery/vision_full_valid_B/vision_pespatial_gem_whiten-train_full_valid.npz",
              "vision_head": ("results/perquery/vision_full_valid_B/"
                              "vision_pespatial_gem_head-nowalls-conv+whiten-train_full_valid.npz")},
    "test": {k: v for k, v in GEOMETRY_SYSTEMS["test"].items() if k in ("vision_congelato", "vision_head")},
}


def item_h(split: str = "valid") -> dict:
    """H: head v2 - (1) robustness on three damages, (2) fusion with W vs the current head,
    (3) beyond the ensemble effect (controls of A)."""
    from src.evaluation.perquery import load_perquery
    from src.evaluation.robustness_auc import compare_auc, load_robust_auc
    prefixes = {"vision_head_v2": str(V2_DIR / split / V2_PREFIX), **DAMAGE_PREFIXES[split]}
    rob = {lab: load_robust_auc(pre, split) for lab, pre in prefixes.items()}
    v2 = rob["vision_head_v2"]
    m1 = {"auc": {lab: {"media_tre_danni": float(d.auc.mean()),
                        **{s: float(c.auc.mean()) for s, c in d.components.items()}} for lab, d in rob.items()}}
    for other in ("vision_congelato", "vision_head"):
        o = rob[other]
        m1[f"v2_minus_{other}"] = {"media_tre_danni": compare_auc(v2, o),
                                   **{s: compare_auc(v2.components[s], o.components[s]) for s in v2.components}}
    m1["prediction_not_worse_than_frozen"] = {
        "rule": "media dei tre danni: CI di v2 - congelato non tutto sotto 0 (§64 H)",
        "holds": not (m1["v2_minus_vision_congelato"]["media_tre_danni"]["ci_hi"] < 0)}
    full_files = {"vision_head_v2": str(V2_DIR / split / f"{V2_PREFIX}_full_{split}.npz"), **FULL_FILES[split]}
    full = {}
    for lab, path in full_files.items():
        d = load_perquery(path)
        full[lab] = {ax: float(d.ndcg[d.axis_index(ax), d.k_index(10)].mean()) for ax in d.meta["axes"]}

    per, raw = {}, {}
    for s in SEEDS:
        F2, F1 = main_fusion(s, "v2", split), main_fusion(s, "head", split)
        if split == "test":
            check_weight(f"fusione v2 s{s}", F2["weight"], TEST_WEIGHTS["v2"][s])
            check_weight(f"fusione head s{s}", F1["weight"], TEST_WEIGHTS["head"][s])
        if F2["sha1"] != F1["sha1"]:
            raise ValueError(f"s{s}: the two fusions use different galleries")
        d = F2["auc_q"] - fs.align_by_name(F2["names"], F1["names"], F1["auc_q"])
        raw[s] = (F2["names"], d)
        per[str(s)] = {"fusion_v2": {k: F2[k] for k in ("dir", "weight", "best_component", "auc")},
                       "fusion_head": {k: F1[k] for k in ("dir", "weight", "best_component", "auc")},
                       "g_v2": fs._mean_ci(F2["g"]), "g_head": fs._mean_ci(F1["g"]),
                       "auc_v2_minus_head": fs._mean_ci(d)}
    m2 = {"per_replica": per, "auc_v2_minus_head_pooled": fs._mean_ci(pooled(raw, raw[42][0])),
          "G": noise_rule(per[str(s)]["auc_v2_minus_head"]["mean"] for s in SEEDS),
          "note": "fusione v2 con W del secondo giro; fusione con la head attuale con W della fase 1 (stessi "
                  "checkpoint e stanze tolte, differenze solo nei pareggi)"}
    return {"rule": "status.md §64 H, §64.2", "split": split, "written": datetime.now().isoformat(timespec="seconds"),
            "measure1_robustness": m1, "full_plan_ndcg10": full, "measure2_fusion_vs_head": m2,
            "measure3_vs_ensemble": item_vs_controls("v2", split)}


# --- test ---

def _passed(path) -> bool:
    try:
        return bool(json.loads(Path(path).read_text())["pass"])
    except (OSError, KeyError, ValueError):
        return False


def test_preconditions() -> list[str]:
    """Everything the single test reading needs, checked before loading any number."""
    from src.evaluation import fusion_head_v2 as fv
    bad = list(r2.gate("test"))
    for c in r2.controls("test"):
        p = r2.control_paths("test", r2.controls("test").index(c))
        if not _passed(p["CHECK_RESET"]):
            bad.append(f"controllo {c['name']}: {p['CHECK_RESET']} mancante o FAIL")
        if not Path(p["SELECT"]).exists():
            bad.append(f"controllo {c['name']}: manca {p['SELECT']}")
    for s in SEEDS:
        hp = fv.head_paths(s, "test")
        if not _passed(hp["CHECK_RESET_JSON"]):
            bad.append(f"fusione v2 s{s}: {hp['CHECK_RESET_JSON']} mancante o FAIL")
        if not Path(hp["SELECT_JSON"]).exists():
            bad.append(f"fusione v2 s{s}: manca {hp['SELECT_JSON']}")
        f = fct.root(seed=s) / "summary_test.json"
        if not f.exists():
            bad.append(f"curva del test s{s}: manca {f} (fusion_curve_test summary --seed {s})")
    for lab, pre in DAMAGE_PREFIXES["test"].items():
        if not list(Path(pre).parent.glob(Path(pre).name + "_partial-*_test.npz")):
            bad.append(f"{lab}: nessun file dei tre danni sul test ({pre})")
    for lab, path in GEOMETRY_SYSTEMS["test"].items():
        if path and not Path(path).exists():
            bad.append(f"{lab}: manca {path}")
    v2_pre = V2_DIR / "test" / V2_PREFIX
    if len(list(v2_pre.parent.glob(v2_pre.name + "_partial-*_test.npz"))) != 9:
        bad.append(f"head v2: non ci sono i 9 file dei tre danni sul test ({v2_pre})")
    return bad


def _p(pid: str, label: str, ok: bool, detail) -> dict:
    return {"id": pid, "label": label, "esito": "confermata" if ok else "non confermata", "detail": detail}


def _ci_txt(ci: dict) -> str:
    return f"{ci['mean']:+.4f} [{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}]"


def predictions_test(res: dict) -> list[dict]:
    """The stated predictions, each on its own."""
    A, B, C, D, E, H, I = (res[k] for k in ("A_frozen_vs_ensemble", "B_head_vs_ensemble", "C_mid_strength",
                                            "D_curve_4_seeds", "E_W_vs_sage", "H_head_v2", "I_geometry"))
    m1, m2, m3 = H["measure1_robustness"], H["measure2_fusion_vs_head"], H["measure3_vs_ensemble"]
    out = []
    # first set
    for k in CONTROLS:
        out.append(_p(f"A-{k}", f"A: vision congelato equivalente all'effetto d'insieme ({CONTROL_NAMES[k]}, ±{DELTA})",
                      A[f"outcome_{k}"].startswith("equivalente"), A[f"outcome_{k}"]))
    out.append(_p("B", "B: vision con head attuale oltre l'effetto d'insieme, confermato su 4 repliche",
                  B["outcome"].startswith("confermato"), B["outcome"]))
    for key in ("gat_t05+frozen", "gat_ref+head"):
        d = C[key]["replica"]["D"]
        out.append(_p(f"C-{key}", f"C: {key} oltre la seconda copia", d["ci_lo"] > 0, _ci_txt(d)))
    e12 = D["predictions_pooled"]["E1_falling_frozen"] + D["predictions_pooled"]["E2_rising_frozen"]
    out.append(_p("D", "D: U rovesciata confermata (E1 ed E2 sulla media dei 4 seed)",
                  all(x["esito"] == "confermata" for x in e12), D["predictions_pooled"]["shape"]))
    out.append(_p("E", "E: W più robusto di sage/comb", E["outcome"] == "W più robusto", _ci_txt(E["delta_pooled"])))
    out.append(_p("H1-§64", "H (1): v2 non peggiore del congelato (media dei tre danni)",
                  m1["prediction_not_worse_than_frozen"]["holds"], m1["v2_minus_vision_congelato"]["media_tre_danni"]))
    for other in ("vision_congelato", "vision_head"):
        c = m1[f"v2_minus_{other}"]["media_tre_danni"]
        out.append(_p(f"H1-{other}", f"H (1): v2 meglio di {other} sulla media dei tre danni (CI > 0)",
                      c["ci_lo"] > 0, c))
    c = m2["auc_v2_minus_head_pooled"]
    out.append(_p("H2", "H (2): la v2 migliora la fusione con W (CI > 0)", c["ci_lo"] > 0, _ci_txt(c)))
    out.append(_p("H3", "H (3): v2 oltre l'effetto d'insieme, confermato su 4 repliche",
                  m3["outcome"].startswith("confermato"), m3["outcome"]))
    # added after the valid
    for k in CONTROLS:
        d = A[f"D_{k}_pooled"]
        out.append(_p(f"P-A-{k}", f"P-A: congelato non oltre l'effetto d'insieme ({CONTROL_NAMES[k]})",
                      not d["ci_lo"] > 0, _ci_txt(d)))
    for key in ("gat_t05+frozen", "gat_ref+head"):
        d = C[key]["crossenc"]["D"]
        out.append(_p(f"P-C-{key}", f"P-C: {key} oltre anche l'altro encoder", d["ci_lo"] > 0, _ci_txt(d)))
    for other in ("vision_congelato", "vision_head"):
        for dmg, c in m1[f"v2_minus_{other}"].items():
            if dmg == "media_tre_danni":
                continue
            out.append(_p(f"P-H1-{other}-{dmg}", f"P-H1: v2 meglio di {other} su {dmg}", c["ci_lo"] > 0, c))
    for s, r in res["H_valid_reference"].items():
        t = m2["per_replica"][s]["fusion_v2"]["auc"]
        out.append(_p(f"P-H2-s{s}", f"P-H2: fusione v2 s{s} entro ±{P_H2_TOL} dal valid",
                      abs(t - r) <= P_H2_TOL, {"test": t, "valid": r, "diff": t - r}))
    for pair in ("W vs vision_congelato", "W vs vision_head"):
        for row in I["pairs"][pair]["rows"]:
            out.append(_p(f"P-I-{pair}-{row['weights']}", f"P-I: {pair}, pesatura {row['weights']}",
                          row["ci_lo"] > 0, {k: row[k] for k in ("delta", "ci_lo", "ci_hi")}))
    above, within = "sopra il rumore fra training", "entro il rumore fra training"
    for name, g, want in (("B-replica", B["G_replica"], above), ("B-crossenc", B["G_crossenc"], above),
                          ("H2", m2["G"], above), ("H3-replica", m3["G_replica"], above),
                          ("H3-crossenc", m3["G_crossenc"], above),
                          ("A-replica", A["G_replica"], within), ("A-crossenc", A["G_crossenc"], within)):
        out.append(_p(f"P-G-{name}", f"P-G: {name} {want}", g["verdict"] == want, g))
    return out


def analyse_test() -> dict:
    bad = test_preconditions()
    if bad:
        for b in bad:
            print(f"!! {b}")
        raise SystemExit("!! lettura del test rifiutata: prima tutti i job, i controlli e le scelte (§65)")
    res = {"rule": "status.md §65 (lettura unica del test del secondo giro)", "split": "test",
           "delta_equivalence": DELTA, "seeds": list(SEEDS), "weights_preregistered": {
               k: ({str(kk): vv for kk, vv in v.items()} if k != "control" else
                   {c: {str(kk): vv for kk, vv in w.items()} for c, w in v.items()}) if k != "mid" else
               {f"{a}/{b}": w for (a, b), w in v.items()} for k, v in TEST_WEIGHTS.items()},
           "written": datetime.now().isoformat(timespec="seconds"),
           "A_frozen_vs_ensemble": item_vs_controls("frozen", "test"),
           "B_head_vs_ensemble": item_vs_controls("head", "test"),
           "C_mid_strength": item_c("test"),
           "D_curve_4_seeds": item_d("test"),
           "E_W_vs_sage": item_e("test"),
           "H_head_v2": item_h("test"),
           "H_valid_reference": {str(s): main_fusion(s, "v2", "valid")["auc"] for s in SEEDS},
           "I_geometry": item_i("test"),
           "notes": ["quarta lettura del test (18 set, 1 ott, 5 ott, questa)",
                     "seed 42 dei controlli A/B: controls_test/ (§62 C), già letti il 5 ott",
                     "C: le fusioni vision + graph di gat/t05 e gat/ref (seed 42) sono quelle di §62 E, già lette",
                     "fusioni principale e con la head attuale: W della fase 1; controlli nuovi e fusione v2: W del "
                     "secondo giro dove rivalutato (stessi checkpoint, differenze solo nei pareggi)",
                     "previsioni P-*: aggiunte in §65 dopo il valid, prima dei job del test"]}
    res["predictions"] = predictions_test(res)
    return res


def analyse_valid() -> dict:
    return {"rule": RULE, "split": "valid", "delta_equivalence": DELTA, "seeds": list(SEEDS),
            "written": datetime.now().isoformat(timespec="seconds"),
            "A_frozen_vs_ensemble": item_vs_controls("frozen"),
            "B_head_vs_ensemble": item_vs_controls("head"),
            "C_mid_strength": item_c(),
            "D_curve_4_seeds": item_d(),
            "E_W_vs_sage": item_e(),
            "notes": ["G: sd con ddof=1 sulle 4 medie per replica",
                      "W rivalutato nel secondo giro per i controlli; le fusioni principali usano la valutazione "
                      "di fase 1 (stessi checkpoint e stanze tolte; differenze solo nei pareggi quasi esatti)"]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="multiseed_analysis")
    ap.add_argument("split", choices=["valid", "test"])
    ap.add_argument("--geometry", action="store_true", help="item I only -> round2/geometry_<split>.json")
    ap.add_argument("--head-v2", action="store_true", help="item H only -> round2/head_v2_<split>.json")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.split == "test" and (a.geometry or a.head_v2):
        raise SystemExit("!! sul test una sola lettura: `multiseed_analysis test` (tutte le voci in un file)")
    kind = "geometry" if a.geometry else "head_v2" if a.head_v2 else "analysis"
    out = r2.ROOT / f"{kind}_{a.split}.json"
    if out.exists():
        raise SystemExit(f"!! {out} esiste gia': niente sovrascritture")
    if a.split == "test":
        res = analyse_test()
    else:
        res = item_i(a.split) if a.geometry else item_h(a.split) if a.head_v2 else analyse_valid()
    out.write_text(json.dumps(res, indent=2, sort_keys=True, default=_jsonable))
    print(f"[multiseed_analysis] scritto {out}")
    if a.split == "test":
        for x in res["predictions"]:
            print(f"  {x['esito']:>14} · {x['label']}")


def _jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON serialisable: {type(x)}")


if __name__ == "__main__":
    main()
