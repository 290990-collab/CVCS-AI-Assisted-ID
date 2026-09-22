# src/evaluation/fusion_select.py

"""
Validity checks and choice of alpha for the late fusion (CPU only). The rules
are the ones pre-registered in status.md §49 (16 Sep 2026), applied literally.

Two commands:

- `check`: validity controls, each with an explicit PASS/FAIL (or DESCRIPTIVE).
  If one fails, stop and diagnose before reading any result.
    C1  same removed rooms in the two branches, query by query (from the qvec files)
    C2  every query of the branch per-query files paired in the fused files + gallery sha1
    C3  alpha=1 reproduces the vision per-query of the SAME job, alpha=0 the graph
        one: for every f at most C3_MAX_DIFF self_rr differ and every rank jump is
        <= C3_MAX_JUMP; AUC delta with `compare_auc`. Full plan vs historical
        files: descriptive.
    C4  same-job branch files vs historical ones (descriptive)
    C5  MRR at f=0.0 in [C5_LOW, C5_HIGH] for every alpha

- `select`:
    (1) `load_auc` for every alpha (strategy nowalls-random), identical names;
    (2) best = highest mean AUC; tie = paired CI 95% of (best - alpha) contains 0;
    (3) alpha* = median of the ties; with two central values the one closer to
        0.5; if equidistant from 0.5, the one with higher mean AUC;
    (4) reference = the branch (alpha=0 or alpha=1) with higher MEAN AUC, compared
        query by query; "helps" iff alpha* not in {0, 1} and ci_lo > 0; delta <
        NOISE_DELTA is declared below the training noise of the graph;
    (5) oracle (DESCRIPTIVE, not a ceiling): per query and f the best self_rr of
        alpha=0 and alpha=1, then AUC; reports alpha* - oracle and how many queries
        alpha* exceeds it on;
    (6) full plan: nDCG@10 C/T/G for {0, alpha*, 1}, paired deltas
        (`significance.compare`), per-axis oracle;
    (7) writes results/fusion/select_<split>.json.
  `--fixed-alpha-from results/fusion/select_valid.json` skips (2)-(3) (the test).

Ensemble-effect control (status.md §50, 17 Sep 2026): `--pair graph-graph` runs
`check` and `select` on the graph + graph fusion (files labelled `random`, first
run = beta=1 via --graph-qvec/--graph-perquery, replica = beta=0 via
--graph2-qvec/--graph2-perquery). Differences fixed by §50: C5 has ONLY the lower
bound; select writes results/fusion/select_graphgraph_<split>.json; valid only.

- `complementarity` (§50 rule): per query, G_F = AUC_F(alpha*) - AUC of the
  component of F with higher mean AUC; G_C = AUC_C(beta*) - same for C;
  D = mean(G_F - G_C), paired by NAME (the two fusions have different labels),
  bootstrap CI (B=10000, seed 0). CI > 0 complementarity · contains 0 not
  distinguishable · CI < 0 explained by the ensemble effect. It lives here
  because it reuses the select rules (`reference_alpha`, `load_auc`, naming) and
  reads the two select json files this module writes.

Control «different models, same information» (status.md §51, 17 Sep 2026):
`--pair vision-vision` (pespatial = gamma=1 via --vision-qvec/--vision-perquery,
radio = gamma=0 via --vision2-qvec/--vision2-perquery), C5 lower bound only,
select -> results/fusion/select_visionvision_<split>.json, valid only.
`complementarity --control-pair vision-vision` applies the FOUR-way rule of §51
on D2 = mean(G_F - G_V) (margin NEGLIGIBLE_DELTA = 0.04) and adds the §51
descriptives (gain scale, Spearman of the components, full-plan V(gamma*) -
pespatial). The three-way rule of §50 for graph-graph is unchanged. Same command
(not a twin) because the pairing, the component rule and the checks are the same;
only the control, the rule and the descriptives change.

Usage:

    python -m src.evaluation.fusion_select check --split valid \\
        --fusion-dir results/perquery/fusion_valid \\
        --vision-qvec results/queryvec/valid/vision_pespatial_gem_whiten-train \\
        --graph-qvec  results/queryvec/valid/graph_gat_asymrob \\
        --vision-perquery results/perquery/fusion_branches_valid/vision_pespatial_gem_whiten-train \\
        --graph-perquery  results/perquery/fusion_branches_valid/graph_gat_asymrob
    python -m src.evaluation.fusion_select select --split valid \\
        --fusion-dir results/perquery/fusion_valid
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.evaluation.gallery_join import load_shared_names
from src.evaluation.late_fusion import (
    ALPHA_NOMINAL_NOTE,
    DEFAULT_ALPHAS,
    DEFAULT_FRACTIONS,
    FUSED_STRATEGY,
    GRAPH_STRATEGY,
    GRAPHGRAPH_STRATEGY,
    PAIRS,
    VISIONVISION_STRATEGY,
    VISION_STRATEGY,
    check_branch,
    check_cross_branch,
    check_query_rows,
    file_pair,
    fused_full_path,
    fused_partial_path,
    fusion_prefix,
    join_queries,
    load_branch_qvecs,
    spread_path,
)
from src.evaluation.perquery import gallery_sha1, load_perquery
from src.evaluation.relevance import AXES
from src.evaluation.robustness_auc import AUC_FRACTIONS, compare_auc, fraction_path, load_auc
from src.evaluation.significance import bootstrap_ci
from src.evaluation.significance import compare as significance_compare
from src.evaluation.significance import paired_self_rr

# Thresholds fixed in status.md §49 BEFORE any number.
C3_MAX_DIFF = 6          # self_rr differing per f (out of 2000), measure of §37
C3_MAX_JUMP = 2          # max rank jump per query
C5_LOW, C5_HIGH = 0.965, 0.980
NOISE_DELTA = 0.04       # training-to-training noise of the graph AUC (§47)
NEGLIGIBLE_DELTA = 0.04  # §51 equivalence margin: negligible robustness difference (§47)
FULL_K = 10


# ----------------------------------------------------------------------
# Pure rules (tested in tests/test_late_fusion.py).
# ----------------------------------------------------------------------

def ties_from_comparisons(comparisons: dict) -> list[float]:
    """Alphas whose paired CI 95% of (best - alpha) contains 0, sorted.

    Args:
        comparisons: alpha -> dict with `ci_lo`, `ci_hi` (as `compare_auc`).
    """
    return sorted(float(a) for a, c in comparisons.items()
                  if c["ci_lo"] <= 0.0 <= c["ci_hi"])


def central_alpha(ties: list[float], means: dict) -> float:
    """alpha* = median of the ties (§49).

    Odd count: the middle value. Even count: of the two central values the one
    closer to 0.5; if equidistant from 0.5 (compared after rounding to 9
    decimals: 0.3 and 0.7 are float-unequal distances), the one with higher mean
    AUC; if even that is equal, the smaller alpha (not pre-registered, flagged
    by `select`).
    """
    if not ties:
        raise ValueError("empty tie set: the best alpha is always tied with itself")
    ties = sorted(float(a) for a in ties)
    n = len(ties)
    if n % 2 == 1:
        return ties[n // 2]
    lo, hi = ties[n // 2 - 1], ties[n // 2]
    d_lo, d_hi = round(abs(lo - 0.5), 9), round(abs(hi - 0.5), 9)
    if d_lo != d_hi:
        return lo if d_lo < d_hi else hi
    if means[lo] != means[hi]:
        return lo if means[lo] > means[hi] else hi
    return lo


def reference_alpha(means: dict) -> float:
    """The single branch with higher mean AUC: 0.0 (graph) or 1.0 (vision).
    Exact equality (not pre-registered) -> 0.0, flagged by `select`."""
    return 1.0 if means[1.0] > means[0.0] else 0.0


def verdict(alpha_star: float, cmp: dict) -> dict:
    """Decision rule of §49 on the paired comparison alpha* - reference."""
    helps = alpha_star not in (0.0, 1.0) and cmp["ci_lo"] > 0.0
    below_noise = bool(cmp["delta"] < NOISE_DELTA)
    text = "la fusione aiuta" if helps else "la fusione non aiuta"
    if below_noise:
        text += f" (delta {cmp['delta']:+.4f} < {NOISE_DELTA}: sotto il rumore fra training del graph)"
    return {"helps": bool(helps), "below_noise": below_noise, "text": text}


# Declarations of status.md §49 («Scelte di implementazione e dichiarazioni»),
# printed and saved with every select output.
VALID_NOTE = ("verdetto ottimista: α* scelto fra 11 sulle stesse query su cui si misura il Δ; "
              "conferma = test con α* fisso")
TEST_NOTE = "riferimento ricalcolato sulle medie del test (conservativo)"
FULL_NOTE = ("Δ per-asse sulla pianta intera descrittivi: etichette derivate dall'input del "
             "graph (circolarità)")
VERDICT_NOTES = (
    "asimmetria del danno: il vision conserva la sagoma del buco, il graph no (perde le stanze, "
    "le superstiti restano nella posizione assoluta)",
    f"valido per questo checkpoint graph; Δ < {NOISE_DELTA} sotto il rumore fra training (§47)",
)


GRAPHGRAPH_NOTES = (
    "controllo dell'effetto d'insieme (§50): questo verdetto non decide nulla, decide "
    "`complementarity`",
    "controllo MINIMO: due training identici sono l'insieme meno diverso (errori più correlati); "
    "esclude l'effetto d'insieme fra training ripetuti, non fra modelli diversi con la stessa "
    "informazione (es. gat + sage): non misurato (correzione del 17 set a §50)",
)
# Declarations of §50 printed with every complementarity output.
COMPLEMENTARITY_NOTES = (
    "ottimismo simmetrico: β*, come α*, scelto fra 11 sulle stesse query",
    "controllo MINIMO: due training identici sono l'insieme meno diverso (errori più correlati); "
    "esclude l'effetto d'insieme fra training ripetuti, non fra modelli diversi con la stessa "
    "informazione (es. gat + sage): non misurato (correzione del 17 set a §50)",
    "solo stanze tolte · solo valid (il test non si consuma)",
    "guadagni e non valori assoluti: i componenti del controllo sono più forti di quelli della "
    "fusione vera",
)
VISIONVISION_NOTES = (
    "controllo «modelli diversi, stessa informazione» (§51): questo verdetto non decide nulla, "
    "decide `complementarity --control-pair vision-vision`",
    "due ViT sono meno diversi fra loro di un ViT e una GNN: anche la complementarità lascia un "
    "residuo sulla quantità di diversità (informazione e modello non si separano del tutto)",
)
# Declarations of §51 printed with every vision-vision complementarity output.
VISIONVISION_COMPLEMENTARITY_NOTES = (
    "due ViT sono meno diversi fra loro di un ViT e una GNN → anche l'esito «complementarità» "
    "lascia un residuo sulla quantità di diversità (limite strutturale: informazione e modello "
    "non si separano del tutto)",
    "il componente migliore del controllo è più debole di quello della fusione vera → più margine "
    "di guadagno, il controllo parte favorito",
    "ottimismo simmetrico: γ*, come α*, scelto fra 11 sulle stesse query",
    "solo valid, solo stanze tolte",
    f"margine di equivalenza ±{NEGLIGIBLE_DELTA}: soglia di differenza trascurabile di robustezza (§47)",
)
VISIONVISION_VERDICTS = {
    "complementarity": "complementarità: vision + graph guadagna più di due modelli diversi con la "
                       "stessa informazione",
    "refuted": "complementarità smentita: due modelli diversi con la stessa informazione guadagnano di più",
    "equivalent": "equivalenti: la diversità dei modelli basta (conclusione negativa con prova)",
    "inconclusive": "non conclusivo: CI contiene 0 ma è più largo del margine",
}
COMPLEMENTARITY_VERDICTS = {
    "complementarity": "complementarità: vision porta più dell'effetto d'insieme fra due training "
                       "identici del graph",
    "indistinguishable": "non distinguibile dall'effetto d'insieme",
    "ensemble": "guadagno di vision + graph spiegato dall'effetto d'insieme (complementarità smentita)",
}


def split_notes(split: str) -> list[str]:
    """Split-dependent declaration of §49."""
    return [VALID_NOTE] if split == "valid" else [TEST_NOTE]


def tie_set_shape(alphas, ties, best: float) -> dict:
    """How the tie set looks on the grid: reduced to {best}? contiguous?

    Contiguous = the ties are consecutive values of the sorted grid `alphas`.
    """
    grid = sorted(float(a) for a in alphas)
    pos = sorted(grid.index(float(a)) for a in ties)
    contiguous = bool(pos) and pos == list(range(pos[0], pos[0] + len(pos)))
    return {"only_best": [float(a) for a in ties] == [float(best)], "contiguous": contiguous}


def oracle_auc(per_fraction_0: dict, per_fraction_1: dict,
               fractions=AUC_FRACTIONS) -> np.ndarray:
    """Per query: mean over f of max(self_rr alpha=0, self_rr alpha=1). DESCRIPTIVE."""
    return np.mean(np.stack([np.maximum(per_fraction_0[f], per_fraction_1[f])
                             for f in fractions]), axis=0)


def oracle_summary(auc_star: np.ndarray, auc_oracle: np.ndarray) -> dict:
    """alpha* against the oracle, same query order. Not a ceiling: summing the
    similarities can beat both branches on the same query."""
    return {
        "oracle_mean": float(np.mean(auc_oracle)) if len(auc_oracle) else float("nan"),
        "star_mean": float(np.mean(auc_star)) if len(auc_star) else float("nan"),
        "delta_star_minus_oracle": float(np.mean(auc_star - auc_oracle)) if len(auc_star) else float("nan"),
        "n_star_exceeds_oracle": int(np.sum(auc_star > auc_oracle)),
        "n": int(len(auc_star)),
    }


def rank_from_rr(rr: np.ndarray, max_k: int) -> np.ndarray:
    """Self rank from reciprocal rank; rr=0 (self not in the top max_k) -> max_k + 1."""
    rr = np.asarray(rr, dtype=float)
    ranks = np.full(rr.shape, max_k + 1, dtype=int)
    hit = rr > 0
    ranks[hit] = np.rint(1.0 / rr[hit]).astype(int)
    return ranks


def c3_stats(rr_a: np.ndarray, rr_b: np.ndarray, max_k: int) -> dict:
    """Paired self_rr of two runs that must coincide: differing count and rank jumps."""
    rr_a, rr_b = np.asarray(rr_a), np.asarray(rr_b)
    diff = rr_a != rr_b
    jumps = np.abs(rank_from_rr(rr_a, max_k) - rank_from_rr(rr_b, max_k))[diff]
    return {
        "n": int(len(rr_a)),
        "n_diff": int(diff.sum()),
        "max_jump": int(jumps.max()) if len(jumps) else 0,
        "jumps": {str(int(k)): int(v) for k, v in sorted(Counter(jumps.tolist()).items())},
    }


def c3_pass(stats: list[dict]) -> bool:
    """PASS iff for EVERY f: n_diff <= C3_MAX_DIFF and every jump <= C3_MAX_JUMP."""
    return all(s["n_diff"] <= C3_MAX_DIFF and s["max_jump"] <= C3_MAX_JUMP for s in stats)


def c5_pass(mrr: float, high: float | None = C5_HIGH) -> bool:
    """MRR at f=0.0 within [C5_LOW, high]; `high=None` = lower bound only (§50)."""
    return bool(C5_LOW <= mrr and (high is None or mrr <= high))


def pair_spec(pair: str, split: str = "valid") -> dict:
    """What changes between the true fusion (§49) and the ensemble control (§50).

    `first` is the alpha/beta=1 side, `second` the alpha/beta=0 side. On the TEST
    the vision-graph pair keeps only the lower C5 bound, as pre-registered in §52
    (the upper bound was written for a single branch: fusion breaks ties between
    RPLAN duplicates and legitimately goes above it, §49.1).
    """
    if pair == "vision-graph":
        return {"pair": pair, "labels": ("vision", "graph"), "branches": ("vision", "graph"),
                "strategies": (VISION_STRATEGY, GRAPH_STRATEGY), "fused": FUSED_STRATEGY,
                "c5_high": C5_HIGH if split == "valid" else None,
                "select_json": "select_{split}.json",
                "section": "§49", "select_notes": VERDICT_NOTES}
    if pair == "graph-graph":
        return {"pair": pair, "labels": ("graph1", "graph2"), "branches": ("graph", "graph"),
                "strategies": (GRAPH_STRATEGY, GRAPH_STRATEGY), "fused": GRAPHGRAPH_STRATEGY,
                "c5_high": None, "select_json": "select_graphgraph_{split}.json",
                "section": "§50", "select_notes": GRAPHGRAPH_NOTES}
    if pair == "vision-vision":
        return {"pair": pair, "labels": ("vision1", "vision2"), "branches": ("vision", "vision"),
                "strategies": (VISION_STRATEGY, VISION_STRATEGY), "fused": VISIONVISION_STRATEGY,
                "c5_high": None, "select_json": "select_visionvision_{split}.json",
                "section": "§51", "select_notes": VISIONVISION_NOTES}
    raise ValueError(f"unknown pair {pair!r} (expected one of {PAIRS})")


def _pair_inputs(args, spec) -> dict:
    """CLI arguments of the first/second side of a pair (None if not given)."""
    if spec["pair"] == "vision-graph":
        first = ("vision_qvec", "vision_perquery", "vision_hist", "vision_full_hist")
        second = ("graph_qvec", "graph_perquery", "graph_hist", "graph_full_hist")
    elif spec["pair"] == "vision-vision":
        first = ("vision_qvec", "vision_perquery", "vision_hist", "vision_full_hist")
        second = ("vision2_qvec", "vision2_perquery", "vision2_hist", "vision2_full_hist")
    else:
        first = ("graph_qvec", "graph_perquery", "graph_hist", "graph_full_hist")
        second = ("graph2_qvec", "graph2_perquery", "graph2_hist", "graph2_full_hist")
    out = {}
    for side, names in (("first", first), ("second", second)):
        out[side] = {k: getattr(args, n, None) for k, n in zip(("qvec", "perquery", "hist", "full_hist"), names)}
    for side in ("first", "second"):
        for k in ("qvec", "perquery"):
            if not out[side][k]:
                raise ValueError(f"{spec['pair']}: missing the {k} prefix of the {side} run")
    return out


# ----------------------------------------------------------------------
# check.
# ----------------------------------------------------------------------

def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _auc_available(fractions) -> bool:
    return all(float(f) in {float(x) for x in fractions} for f in AUC_FRACTIONS)


def cmd_check(args) -> dict:
    """Runs C1-C5 and prints PASS/FAIL. Returns the report (also as JSON if --out-json)."""
    fractions = [float(f) for f in args.fractions]
    alphas = [float(a) for a in args.alphas]
    spec = pair_spec(getattr(args, "pair", "vision-graph"), args.split)
    if spec["pair"] != "vision-graph" and args.split != "valid":
        raise ValueError(f"{spec['pair']} control is valid only (status.md {spec['section']})")
    inp = _pair_inputs(args, spec)
    (l1, l2), (b1, b2), (s1, s2) = spec["labels"], spec["branches"], spec["strategies"]
    fused_strategy = spec["fused"]
    report = {"split": args.split, "fractions": fractions, "alphas": alphas, "pair": spec["pair"]}
    failed = []

    # C1 — same removed rooms (hard checks of late_fusion, re-run here).
    shared = load_shared_names(args.gallery_names)
    shared_sha1 = gallery_sha1(shared)
    c1 = {}
    vq = gq = None
    try:
        vq = load_branch_qvecs(inp["first"]["qvec"], args.split, fractions, s1)
        gq = load_branch_qvecs(inp["second"]["qvec"], args.split, fractions, s2)
        check_branch(vq, b1, args.split, shared_sha1, len(shared))
        check_branch(gq, b2, args.split, shared_sha1, len(shared))
        check_cross_branch(vq, gq, (l1, l2))
        for f in fractions:
            check_query_rows(vq[f], shared, f"{l1} f={f}")
            check_query_rows(gq[f], shared, f"{l2} f={f}")
            j = join_queries(vq[f], gq[f], f, (l1, l2))
            c1[str(f)] = {"n_paired": len(j.names), f"n_only_{l1}": j.n_only_vision,
                          f"n_only_{l2}": j.n_only_graph}
        c1_ok, c1_msg = True, ""
    except (ValueError, FileNotFoundError, KeyError) as exc:
        c1_ok, c1_msg = False, str(exc)
    report["C1"] = {"status": _status(c1_ok), "per_fraction": c1, "error": c1_msg}
    print(f"C1 stesse stanze tolte (qvec): {_status(c1_ok)} {c1_msg}")
    for f, v in c1.items():
        print(f"    f={f}: appaiate {v['n_paired']} · solo {l1} {v[f'n_only_{l1}']} "
              f"· solo {l2} {v[f'n_only_{l2}']}")
    if not c1_ok:
        failed.append("C1")

    # C2 — every branch query paired in the fused files, same gallery.
    c2_ok, c2 = True, {}
    for f in fractions:
        pv = load_perquery(fraction_path(inp["first"]["perquery"], f, args.split, s1))
        pg = load_perquery(fraction_path(inp["second"]["perquery"], f, args.split, s2))
        ref = {str(n) for n in pv.names}
        row = {f"n_{l1}": len(pv.names), f"n_{l2}": len(pg.names), "per_alpha": {}}
        ok_f = ({str(n) for n in pg.names} == ref
                and pv.meta["gallery"]["sha1"] == shared_sha1 == pg.meta["gallery"]["sha1"])
        for a in alphas:
            pf = load_perquery(fused_partial_path(args.fusion_dir, a, f, args.split, fused_strategy))
            names = {str(n) for n in pf.names}
            paired = len(names & ref)
            ok_a = (names == ref and pf.meta["gallery"]["sha1"] == shared_sha1
                    and file_pair(pf.meta) == spec["pair"])
            row["per_alpha"][f"{a:g}"] = {"n_paired": paired, "ok": ok_a}
            ok_f = ok_f and ok_a
        row["ok"] = ok_f
        c2[str(f)] = row
        c2_ok = c2_ok and ok_f
        print(f"C2 f={f}: {_status(ok_f)} · {l1} {row[f'n_{l1}']} · {l2} {row[f'n_{l2}']} · "
              f"appaiate (min sugli alpha) {min(v['n_paired'] for v in row['per_alpha'].values())}"
              f"/{len(ref)} · sha1 {shared_sha1[:12]}")
    report["C2"] = {"status": _status(c2_ok), "per_fraction": c2}
    if not c2_ok:
        failed.append("C2")

    # C3 — alpha=1 / alpha=0 reproduce the branches of the SAME job.
    c3 = {l1: {}, l2: {}}
    stats_all = []
    same_job_ok = True
    for branch, alpha, prefix, strategy, qvecs in (
        (l1, 1.0, inp["first"]["perquery"], s1, vq),
        (l2, 0.0, inp["second"]["perquery"], s2, gq),
    ):
        for f in fractions:
            fused = load_perquery(fused_partial_path(args.fusion_dir, alpha, f, args.split,
                                                     fused_strategy))
            own_path = fraction_path(prefix, f, args.split, strategy)
            own = load_perquery(own_path)
            rr_a, rr_b, _ = paired_self_rr(fused, own)
            s = c3_stats(rr_a, rr_b, int(fused.meta["max_k"]))
            # The branch per-query must be the file written by the job that wrote the
            # qvec: historical files with the same name live in other folders.
            s["same_job"] = bool(
                qvecs is not None and f in qvecs
                and Path(qvecs[f].meta.get("perquery_file", "")).resolve() == own_path.resolve())
            same_job_ok = same_job_ok and s["same_job"]
            c3[branch][str(f)] = s
            stats_all.append(s)
            print(f"C3 alpha={alpha:g} vs {branch} f={f}: self_rr diversi {s['n_diff']}/{s['n']} "
                  f"· salto max {s['max_jump']} · salti {s['jumps']} "
                  f"· stesso job del qvec {_status(s['same_job'])}")
        if _auc_available(fractions):
            cmp = compare_auc(load_auc(fusion_prefix(args.fusion_dir, alpha), args.split,
                                       strategy=fused_strategy),
                              load_auc(prefix, args.split, strategy=strategy))
            c3[branch]["auc_delta"] = cmp
            print(f"    AUC fusione alpha={alpha:g} - {branch}: {cmp['delta']:+.5f} "
                  f"[{cmp['ci_lo']:+.5f}, {cmp['ci_hi']:+.5f}]")
    c3_ok = c3_pass(stats_all) and same_job_ok
    report["C3"] = {"status": _status(c3_ok), **c3, "same_job": same_job_ok,
                    "thresholds": {"max_diff": C3_MAX_DIFF, "max_jump": C3_MAX_JUMP}}
    print(f"C3 soglia (ogni f: <= {C3_MAX_DIFF} diversi, salto <= {C3_MAX_JUMP}) "
          f"e per-query dello stesso job dei qvec: {_status(c3_ok)}")
    if not c3_ok:
        failed.append("C3")

    # C3 full (descriptive): fused alpha=1/0 full vs historical full files.
    full_desc = {}
    for branch, alpha, hist in ((l1, 1.0, inp["first"]["full_hist"]),
                                (l2, 0.0, inp["second"]["full_hist"])):
        if not hist:
            continue
        # Descriptive only: an incompatible historical file must not stop C5.
        try:
            fused = load_perquery(fused_full_path(args.fusion_dir, alpha, args.split))
            other = load_perquery(hist)
            full_desc[branch] = {ax: significance_compare(fused, other, ax, "ndcg", FULL_K)
                                 for ax in AXES}
        except (ValueError, KeyError, FileNotFoundError) as exc:
            full_desc[branch] = {"error": str(exc)}
            print(f"C3-full (descrittivo) {branch}: non calcolabile — {exc}")
            continue
        for ax, c in full_desc[branch].items():
            print(f"C3-full (descrittivo) alpha={alpha:g} - {branch} storico nDCG@{FULL_K} {ax}: "
                  f"{c['delta']:+.5f} [{c['ci_lo']:+.5f}, {c['ci_hi']:+.5f}] n={c['n_pairs']}")
    report["C3_full_descriptive"] = full_desc

    # C4 — same-job branch files vs historical ones (descriptive).
    c4 = {}
    for branch, prefix, hist, strategy in (
        (l1, inp["first"]["perquery"], inp["first"]["hist"], s1),
        (l2, inp["second"]["perquery"], inp["second"]["hist"], s2),
    ):
        if not hist or not _auc_available(fractions):
            continue
        # Descriptive only: an incompatible historical file must not stop C5.
        try:
            cmp = compare_auc(load_auc(prefix, args.split, strategy=strategy),
                              load_auc(hist, args.split, strategy=strategy))
        except (ValueError, KeyError, FileNotFoundError) as exc:
            c4[branch] = {"error": str(exc)}
            print(f"C4 (descrittivo) {branch}: non calcolabile — {exc}")
            continue
        c4[branch] = cmp
        print(f"C4 (descrittivo) {branch} nuovo - storico: AUC {cmp['delta']:+.5f} "
              f"[{cmp['ci_lo']:+.5f}, {cmp['ci_hi']:+.5f}] n={cmp['n_pairs']}")
    report["C4_descriptive"] = c4

    # C5 — MRR at f=0.0 for every alpha.
    c5 = {}
    if 0.0 in fractions:
        for a in alphas:
            pf = load_perquery(fused_partial_path(args.fusion_dir, a, 0.0, args.split,
                                                  fused_strategy))
            mrr = float(np.nanmean(pf.self_rr))
            c5[f"{a:g}"] = {"mrr": mrr, "ok": c5_pass(mrr, spec["c5_high"])}
        c5_ok = all(v["ok"] for v in c5.values())
        c5_msg = ""
    else:
        c5_ok, c5_msg = False, "f=0.0 non fra le frazioni: controllo non calcolabile"
    report["C5"] = {"status": _status(c5_ok), "per_alpha": c5, "range": [C5_LOW, spec["c5_high"]],
                    "error": c5_msg}
    rng_txt = f"in [{C5_LOW}, {C5_HIGH}]" if spec["c5_high"] is not None else f">= {C5_LOW} (solo limite basso, §50)"
    print(f"C5 MRR f=0.0 {rng_txt}: {_status(c5_ok)} {c5_msg}")
    for a, v in c5.items():
        print(f"    alpha={a}: MRR {v['mrr']:.4f} {_status(v['ok'])}")
    if not c5_ok:
        failed.append("C5")

    report["failed"] = failed
    print(f"\nESITO: {'PASS' if not failed else 'FAIL ' + ', '.join(failed)}")
    if args.out_json:
        _write_json(args.out_json, report)
    return report


# ----------------------------------------------------------------------
# select.
# ----------------------------------------------------------------------

def _write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[fusion_select] scritto {path}")


def _full_block(fusion_dir, split, alpha_star, labels=("vision", "graph")) -> dict:
    """(6) full plan: nDCG@10 per axis for {0, alpha*, 1}, paired deltas, oracle.

    `labels` = (alpha=1 side, alpha=0 side): key names `star_minus_<label>`."""
    datas = {a: load_perquery(fused_full_path(fusion_dir, a, split))
             for a in sorted({0.0, alpha_star, 1.0})}
    k1, k0 = f"star_minus_{labels[0]}", f"star_minus_{labels[1]}"
    out = {"k": FULL_K, "means": {}, k0: {}, k1: {},
           "oracle": {}, "note": FULL_NOTE}
    for a, d in datas.items():
        out["means"][f"{a:g}"] = {ax: float(np.nanmean(d.ndcg[d.axis_index(ax), d.k_index(FULL_K)]))
                                  for ax in AXES}
    star, g, v = datas[alpha_star], datas[0.0], datas[1.0]
    for ax in AXES:
        out[k0][ax] = significance_compare(star, g, ax, "ndcg", FULL_K)
        out[k1][ax] = significance_compare(star, v, ax, "ndcg", FULL_K)
        # Oracle per axis: per query max(nDCG alpha=0, nDCG alpha=1), paired by name.
        col = {str(n): i for i, n in enumerate(v.names)}
        row_g = g.ndcg[g.axis_index(ax), g.k_index(FULL_K)]
        row_v = v.ndcg[v.axis_index(ax), v.k_index(FULL_K)]
        row_s = star.ndcg[star.axis_index(ax), star.k_index(FULL_K)]
        col_s = {str(n): i for i, n in enumerate(star.names)}
        orc, st = [], []
        for i, n in enumerate(g.names):
            n = str(n)
            if n not in col or n not in col_s:
                continue
            a0, a1, s = float(row_g[i]), float(row_v[col[n]]), float(row_s[col_s[n]])
            if np.isnan(a0) or np.isnan(a1) or np.isnan(s):
                continue
            orc.append(max(a0, a1))
            st.append(s)
        out["oracle"][ax] = oracle_summary(np.asarray(st), np.asarray(orc))
    return out


def cmd_select(args) -> dict:
    """Applies the §49 rule (§50 for graph-graph) and writes the select json."""
    spec = pair_spec(getattr(args, "pair", "vision-graph"), args.split)
    if spec["pair"] != "vision-graph" and (args.split != "valid" or args.fixed_alpha_from):
        raise ValueError(f"{spec['pair']} control: valid only, no fixed alpha "
                         f"(status.md {spec['section']})")
    fixed = None
    if args.fixed_alpha_from:
        fixed = float(json.loads(Path(args.fixed_alpha_from).read_text())["alpha_star"])
        alphas = sorted({0.0, fixed, 1.0})
    else:
        alphas = sorted(float(a) for a in args.alphas)
    if 0.0 not in alphas or 1.0 not in alphas:
        raise ValueError("the grid must contain alpha=0 and alpha=1 (the two branches)")

    # (1) AUC per alpha, identical query names.
    datas = {a: load_auc(fusion_prefix(args.fusion_dir, a), args.split, strategy=spec["fused"])
             for a in alphas}
    ref_names = datas[alphas[0]].names
    for a, d in datas.items():
        if not np.array_equal(d.names, ref_names):
            raise ValueError(f"alpha={a:g}: query names differ from alpha={alphas[0]:g}")
        if file_pair(d.meta) != spec["pair"]:
            raise ValueError(f"alpha={a:g}: fused files of pair {file_pair(d.meta)!r}, "
                             f"expected {spec['pair']!r}")
    means = {a: d.mean for a, d in datas.items()}
    result = {"split": args.split, "fusion_dir": str(args.fusion_dir),
              "alphas": alphas, "auc_means": {f"{a:g}": m for a, m in means.items()},
              "n_queries": int(len(ref_names)), "notes": split_notes(args.split),
              "pair": spec["pair"], "strategy": spec["fused"]}

    # (2)-(3) best, ties, central alpha.
    if fixed is None:
        best = max(alphas, key=lambda a: means[a])
        comparisons = {a: compare_auc(datas[best], datas[a]) for a in alphas}
        ties = ties_from_comparisons(comparisons)
        alpha_star = central_alpha(ties, means)
        n = len(ties)
        if n % 2 == 0:
            lo, hi = ties[n // 2 - 1], ties[n // 2]
            if round(abs(lo - 0.5), 9) == round(abs(hi - 0.5), 9) and means[lo] == means[hi]:
                result["notes"].append("central alphas equidistant with equal AUC: "
                                       "smaller alpha taken (not pre-registered)")
        shape = tie_set_shape(alphas, ties, best)
        table = [{"alpha": a, "auc": means[a], "delta_best_minus_alpha": comparisons[a]["delta"],
                  "ci_lo": comparisons[a]["ci_lo"], "ci_hi": comparisons[a]["ci_hi"],
                  "tie": a in ties} for a in alphas]
        if shape["only_best"]:
            result["notes"].append("insieme dei pari = {best}: nessun altro α indistinguibile")
        if not shape["contiguous"]:
            result["notes"].append("insieme dei pari NON contiguo sulla griglia")
        result.update({"best": best,
                       "best_vs_alpha": {f"{a:g}": c for a, c in comparisons.items()},
                       "ties": ties, "ties_shape": shape, "ties_table": table,
                       "alpha_star": alpha_star, "alpha_star_rule": "central_tie"})
    else:
        alpha_star = fixed
        result.update({"alpha_star": alpha_star, "alpha_star_rule": "fixed",
                       "fixed_alpha_from": str(args.fixed_alpha_from)})

    # (4) reference branch and verdict.
    ref = reference_alpha(means)
    if means[0.0] == means[1.0]:
        result["notes"].append("alpha=0 and alpha=1 with equal mean AUC: graph taken as reference")
    cmp = compare_auc(datas[alpha_star], datas[ref])
    result.update({"reference_alpha": ref, "star_vs_reference": cmp,
                   "verdict": {**verdict(alpha_star, cmp),
                               "notes": list(spec["select_notes"])}})

    # (5) oracle, descriptive.
    orc = oracle_auc(datas[0.0].per_fraction, datas[1.0].per_fraction)
    result["oracle"] = oracle_summary(datas[alpha_star].auc, orc)

    # (6) full plan.
    result["full"] = _full_block(args.fusion_dir, args.split, alpha_star, spec["labels"])

    # Similarity spread of the two branches (descriptive sidecar of late_fusion).
    sp = spread_path(args.fusion_dir, args.split)
    if sp.exists():
        result["similarity_spread"] = {**json.loads(sp.read_text()), "note": ALPHA_NOMINAL_NOTE}
    else:
        result["similarity_spread"] = None
        result["notes"].append(f"sidecar dello spread mancante: {sp}")

    _print_select(result)
    out = args.out_json or f"results/fusion/{spec['select_json'].format(split=args.split)}"
    _write_json(out, result)
    return result


def _print_select(r: dict) -> None:
    print(f"[fusion_select] {r.get('pair', 'vision-graph')} · split={r['split']} · "
          f"{r['n_queries']} query · AUC ({r.get('strategy', FUSED_STRATEGY)}, "
          f"f={AUC_FRACTIONS})")
    for a, m in r["auc_means"].items():
        tie = ""
        if "best_vs_alpha" in r:
            c = r["best_vs_alpha"][a]
            tie = f"  best-alpha {c['delta']:+.4f} [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]" \
                  + ("  PARI" if c["ci_lo"] <= 0 <= c["ci_hi"] else "")
        print(f"  alpha={a:>4}: {m:.4f}{tie}")
    if "ties" in r:
        shape = r["ties_shape"]
        print(f"pari (per intero): {[f'{a:g}' for a in r['ties']]} · solo best: "
              f"{shape['only_best']} · contigui: {shape['contiguous']}")
    print(f"alpha* = {r['alpha_star']:g} ({r['alpha_star_rule']}) · riferimento alpha="
          f"{r['reference_alpha']:g}")
    c = r["star_vs_reference"]
    print(f"alpha* - riferimento: {c['delta']:+.4f} [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] "
          f"n={c['n_pairs']} -> {r['verdict']['text']}")
    for note in r["verdict"]["notes"]:
        print(f"  verdetto, nota: {note}")
    o = r["oracle"]
    print(f"oracolo (descrittivo): {o['oracle_mean']:.4f} · alpha* - oracolo "
          f"{o['delta_star_minus_oracle']:+.4f} · alpha* lo supera su {o['n_star_exceeds_oracle']}/{o['n']}")
    for a, m in r["full"]["means"].items():
        print(f"  full alpha={a}: " + " · ".join(f"{ax} {v:.4f}" for ax, v in m.items()))
    print(f"  full, nota: {r['full']['note']}")
    sp = r.get("similarity_spread")
    if sp:
        for branch in ("vision", "graph", "graph1", "graph2", "vision1", "vision2"):
            for key, v in (sp.get(branch) or {}).items():
                print(f"  spread {branch} [{key}]: std top-{v['k']} {v['mean_std_topk']:.4f} · "
                      f"gap 1-{v['k']} {v['mean_gap_rank1_rankk']:.4f}")
        print(f"  spread, nota: {sp['note']}")
    for note in r["notes"]:
        print(f"NOTA: {note}")


# ----------------------------------------------------------------------
# complementarity (§50).
# ----------------------------------------------------------------------

def complementarity_verdict(ci_lo: float, ci_hi: float) -> str:
    """Three-way rule of §50 on the CI of D = mean(G_F - G_C)."""
    if ci_lo > 0.0:
        return "complementarity"
    if ci_hi < 0.0:
        return "ensemble"
    return "indistinguishable"


def four_way_verdict(ci_lo: float, ci_hi: float, margin: float = NEGLIGIBLE_DELTA) -> str:
    """Four-way rule of §51 on the CI of D2 = mean(G_F - G_V).

    CI > 0 complementarity · CI < 0 refuted · CI contains 0 and lies entirely
    within [-margin, +margin] (bounds included) equivalent · otherwise inconclusive.
    """
    if ci_lo > 0.0:
        return "complementarity"
    if ci_hi < 0.0:
        return "refuted"
    if -margin <= ci_lo and ci_hi <= margin:
        return "equivalent"
    return "inconclusive"


def average_ranks(x) -> np.ndarray:
    """Ranks 1..n with ties sharing their average rank (as Spearman needs)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman rank correlation = Pearson on average ranks (NaN if a side is constant)."""
    ra, rb = average_ranks(a), average_ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def align_by_name(names_ref, names, values) -> np.ndarray:
    """`values` (aligned with `names`) reordered on `names_ref`.

    Raises:
        ValueError: the two name sets differ (with a count and an example).
    """
    ref = [str(n) for n in names_ref]
    col = {str(n): i for i, n in enumerate(names)}
    missing = [n for n in ref if n not in col]
    if missing or len(col) != len(ref):
        extra = [n for n in col if n not in set(ref)]
        raise ValueError(f"query sets differ: {len(missing)} missing (e.g. {missing[:2]}), "
                         f"{len(extra)} extra (e.g. {extra[:2]})")
    return np.asarray([values[col[n]] for n in ref], dtype=float)


def _mean_ci(x: np.ndarray) -> dict:
    lo, hi = bootstrap_ci(np.asarray(x, dtype=float))
    return {"mean": float(np.mean(x)) if len(x) else float("nan"), "ci_lo": lo, "ci_hi": hi}


def complementarity_stats(f_star, f_comp, c_star, c_comp, rule: str = "three_way") -> dict:
    """Per-query gains of the true fusion F and of the control C (same order).

    Args: AUC arrays [P], aligned by name: F at alpha*, F best component,
        C at beta*, C best component. `rule`: "three_way" (§50) | "four_way" (§51).
    """
    g_f = np.asarray(f_star, float) - np.asarray(f_comp, float)
    g_c = np.asarray(c_star, float) - np.asarray(c_comp, float)
    d = _mean_ci(g_f - g_c)
    mean_gf, mean_gc = float(np.mean(g_f)), float(np.mean(g_c))
    return {
        "n": int(len(g_f)),
        "D": d,
        "verdict": (complementarity_verdict(d["ci_lo"], d["ci_hi"]) if rule == "three_way"
                    else four_way_verdict(d["ci_lo"], d["ci_hi"])),
        "G_F": _mean_ci(g_f),
        "G_C": _mean_ci(g_c),
        "AUC_F_minus_AUC_C": _mean_ci(np.asarray(f_star, float) - np.asarray(c_star, float)),
        "share_G_C_over_G_F": mean_gc / mean_gf if mean_gf != 0 else float("nan"),
    }


def _fusion_endpoints(fusion_dir, split, star, strategy, pair) -> tuple[dict, float]:
    """AucData for {0, star, 1} of one fusion + its best component (higher mean)."""
    datas = {a: load_auc(fusion_prefix(fusion_dir, a), split, strategy=strategy)
             for a in sorted({0.0, float(star), 1.0})}
    for a, d in datas.items():
        if file_pair(d.meta) != pair:
            raise ValueError(f"{fusion_dir} alpha={a:g}: pair {file_pair(d.meta)!r}, expected {pair!r}")
    comp = reference_alpha({0.0: datas[0.0].mean, 1.0: datas[1.0].mean})
    return datas, comp


def cmd_complementarity(args) -> dict:
    """§50 decision: gain of vision + graph vs gain of graph + graph, paired by name."""
    true_sel = json.loads(Path(args.true_select).read_text())
    ctrl_sel = json.loads(Path(args.control_select).read_text())
    for what, sel, fdir in (("true", true_sel, args.true_dir), ("control", ctrl_sel, args.control_dir)):
        if sel.get("split") != args.split:
            raise ValueError(f"{what} select json: split {sel.get('split')!r}, expected {args.split!r}")
        # the select json must describe the fusion folder passed on the command line
        if Path(str(sel.get("fusion_dir", ""))).resolve() != Path(fdir).resolve():
            raise ValueError(f"{what} select json: fusion_dir {sel.get('fusion_dir')!r}, "
                             f"expected {str(fdir)!r}")
    a_star, b_star = float(true_sel["alpha_star"]), float(ctrl_sel["alpha_star"])
    control_pair = getattr(args, "control_pair", "graph-graph")
    F, comp_f = _fusion_endpoints(args.true_dir, args.split, a_star, FUSED_STRATEGY, "vision-graph")
    C, comp_c = _fusion_endpoints(args.control_dir, args.split, b_star,
                                  pair_spec(control_pair)["fused"], control_pair)

    # Same gallery everywhere; same query set in the two fusions (join by NAME).
    sha = {d.meta["gallery"]["sha1"] for d in list(F.values()) + list(C.values())}
    if len(sha) != 1:
        raise ValueError(f"gallery sha1 differs between the fusions: {sorted(s[:12] for s in sha)}")
    names = F[a_star].names
    f_star = align_by_name(names, F[a_star].names, F[a_star].auc)
    f_comp = align_by_name(names, F[comp_f].names, F[comp_f].auc)
    c_star = align_by_name(names, C[b_star].names, C[b_star].auc)
    c_comp = align_by_name(names, C[comp_c].names, C[comp_c].auc)
    if control_pair == "vision-vision":
        return _complementarity_visionvision(args, true_sel, ctrl_sel, F, comp_f, C, comp_c,
                                             names, f_star, f_comp, c_star, c_comp, sha.pop())
    stats = complementarity_stats(f_star, f_comp, c_star, c_comp)

    result = {
        "split": args.split, "rule": "status.md §50",
        "true": {"dir": str(args.true_dir), "select": str(args.true_select), "alpha_star": a_star,
                 "best_component": comp_f,
                 "means": {f"{a:g}": d.mean for a, d in F.items()}},
        "control": {"dir": str(args.control_dir), "select": str(args.control_select),
                    "beta_star": b_star, "best_component": comp_c,
                    "means": {f"{a:g}": d.mean for a, d in C.items()},
                    "full_descriptive": {"full": ctrl_sel.get("full"), "note": FULL_NOTE}},
        "gallery_sha1": sha.pop(),
        **stats,
        "verdict_text": COMPLEMENTARITY_VERDICTS[stats["verdict"]],
        "notes": list(COMPLEMENTARITY_NOTES),
    }
    _print_complementarity(result)
    _write_json(args.out_json or f"results/fusion/complementarity_{args.split}.json", result)
    return result


def _complementarity_visionvision(args, true_sel, ctrl_sel, F, comp_f, V, comp_v, names,
                                 f_star, f_comp, v_star, v_comp, sha1) -> dict:
    """§51 decision (four-way rule) and descriptives. See `cmd_complementarity`."""
    a_star, g_star = float(true_sel["alpha_star"]), float(ctrl_sel["alpha_star"])
    stats = complementarity_stats(f_star, f_comp, v_star, v_comp, rule="four_way")
    notes = list(VISIONVISION_COMPLEMENTARITY_NOTES)

    # Spearman of the per-query AUC of the two components of each pair.
    spear = {
        "vision/graph": spearman(align_by_name(names, F[1.0].names, F[1.0].auc),
                                 align_by_name(names, F[0.0].names, F[0.0].auc)),
        "pespatial/radio": spearman(align_by_name(names, V[1.0].names, V[1.0].auc),
                                    align_by_name(names, V[0.0].names, V[0.0].auc)),
        "graph/replica": None,
    }
    gg_dir = Path(args.graphgraph_dir)
    try:
        g1 = load_auc(fusion_prefix(gg_dir, 1.0), args.split, strategy=GRAPHGRAPH_STRATEGY)
        g0 = load_auc(fusion_prefix(gg_dir, 0.0), args.split, strategy=GRAPHGRAPH_STRATEGY)
        spear["graph/replica"] = spearman(align_by_name(names, g1.names, g1.auc),
                                          align_by_name(names, g0.names, g0.auc))
    except (FileNotFoundError, ValueError) as exc:
        notes.append(f"Spearman graph/replica non calcolabile ({gg_dir}): {exc}")

    # G_C of §50: READ from its complementarity json (not recomputed), same true fusion checked.
    g_c = None
    gg_json = Path(args.graphgraph_json)
    if gg_json.exists():
        gg = json.loads(gg_json.read_text())
        same_true = (float(gg.get("true", {}).get("alpha_star", float("nan"))) == a_star
                     and Path(str(gg.get("true", {}).get("dir", ""))).resolve()
                     == Path(args.true_dir).resolve())
        if not same_true:
            raise ValueError(f"{gg_json}: computed on another true fusion (dir/alpha*): "
                             "the gain scale would mix two fusions")
        g_c = {**gg["G_C"], "source": f"letto da {gg_json} (non ricalcolato)"}
    else:
        notes.append(f"G_C (§50) non disponibile: {gg_json} mancante")

    # Full plan: V(gamma*) - pespatial (gamma=1), from the control select json (circular).
    full = ctrl_sel.get("full") or {}
    means = full.get("means") or {}
    star_m, pes_m = means.get(f"{g_star:g}"), means.get("1")
    full_delta = ({ax: star_m[ax] - pes_m[ax] for ax in AXES if ax in star_m and ax in pes_m}
                  if star_m and pes_m else None)

    result = {
        "split": args.split, "rule": "status.md §51", "margin": NEGLIGIBLE_DELTA,
        "true": {"dir": str(args.true_dir), "select": str(args.true_select), "alpha_star": a_star,
                 "best_component": comp_f, "means": {f"{a:g}": d.mean for a, d in F.items()}},
        "control": {"pair": "vision-vision", "dir": str(args.control_dir),
                    "select": str(args.control_select), "gamma_star": g_star,
                    "best_component": comp_v, "means": {f"{a:g}": d.mean for a, d in V.items()}},
        "gallery_sha1": sha1,
        "n": stats["n"],
        "D2": stats["D"],
        "verdict": stats["verdict"],
        "verdict_text": VISIONVISION_VERDICTS[stats["verdict"]],
        "gain_scale": {"G_C": g_c, "G_V": stats["G_C"], "G_F": stats["G_F"]},
        "AUC_F_minus_AUC_V": stats["AUC_F_minus_AUC_C"],
        "share_G_V_over_G_F": stats["share_G_C_over_G_F"],
        "spearman_components": spear,
        "full_descriptive": {"V_star_minus_pespatial": full_delta,
                             "star_minus_vision1": full.get("star_minus_vision1"),
                             "note": FULL_NOTE},
        "notes": notes,
    }
    _print_complementarity_visionvision(result)
    _write_json(args.out_json or f"results/fusion/complementarity_visionvision_{args.split}.json",
                result)
    return result


def _print_complementarity_visionvision(r: dict) -> None:
    t, c = r["true"], r["control"]
    print(f"[complementarity §51] split={r['split']} · n={r['n']} query appaiate per nome · "
          f"sha1 {r['gallery_sha1'][:12]}")
    print(f"  F = vision+graph α*={t['alpha_star']:g} · componente migliore α={t['best_component']:g} "
          f"· medie {t['means']}")
    print(f"  V = pespatial+radio γ*={c['gamma_star']:g} · componente migliore γ={c['best_component']:g} "
          f"· medie {c['means']}")
    for key, v in r["gain_scale"].items():
        if v:
            print(f"  scala {key}: {v['mean']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]")
    for key in ("AUC_F_minus_AUC_V", "D2"):
        v = r[key]
        print(f"  {key}: {v['mean']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]")
    print(f"  quota G_V/G_F: {r['share_G_V_over_G_F']:.3f}")
    for pair, rho in r["spearman_components"].items():
        print(f"  Spearman componenti {pair}: " + ("n/d" if rho is None else f"{rho:.3f}"))
    fd = r["full_descriptive"]["V_star_minus_pespatial"]
    if fd:
        print("  pianta intera (descrittivo) V(γ*) − pespatial: "
              + " · ".join(f"{ax} {v:+.4f}" for ax, v in fd.items()))
    print(f"VERDETTO (§51, margine ±{r['margin']}): {r['verdict_text']}")
    for note in r["notes"] + [r["full_descriptive"]["note"]]:
        print(f"NOTA: {note}")


def _print_complementarity(r: dict) -> None:
    t, c = r["true"], r["control"]
    print(f"[complementarity] split={r['split']} · n={r['n']} query appaiate per nome · "
          f"sha1 {r['gallery_sha1'][:12]}")
    print(f"  F = vision+graph α*={t['alpha_star']:g} · componente migliore α={t['best_component']:g} "
          f"· medie {t['means']}")
    print(f"  C = graph+graph β*={c['beta_star']:g} · componente migliore β={c['best_component']:g} "
          f"· medie {c['means']}")
    for key in ("G_F", "G_C", "AUC_F_minus_AUC_C", "D"):
        v = r[key]
        print(f"  {key}: {v['mean']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]")
    print(f"  quota G_C/G_F: {r['share_G_C_over_G_F']:.3f}")
    means = ((c["full_descriptive"].get("full") or {}).get("means") or {})
    star, comp = means.get(f"{c['beta_star']:g}"), means.get(f"{c['best_component']:g}")
    if star and comp:
        deltas = " · ".join(f"{ax} {star[ax] - comp[ax]:+.4f}" for ax in AXES if ax in star and ax in comp)
        print(f"  pianta intera (descrittivo) C β*={c['beta_star']:g} − training migliore "
              f"β={c['best_component']:g}: {deltas}")
    print(f"VERDETTO (§50): {r['verdict_text']}")
    for note in r["notes"] + [r["control"]["full_descriptive"]["note"]]:
        print(f"NOTA: {note}")


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Late fusion: checks and alpha choice (status.md §49)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="validity controls C1-C5")
    c.add_argument("--vision-qvec", default=None, dest="vision_qvec")
    c.add_argument("--graph-qvec", default=None, dest="graph_qvec")
    c.add_argument("--vision-perquery", default=None, dest="vision_perquery",
                   help="prefix of the vision per-query files of the SAME job as the qvec")
    c.add_argument("--graph-perquery", default=None, dest="graph_perquery",
                   help="prefix of the graph per-query files of the SAME job as the qvec")
    c.add_argument("--graph2-qvec", default=None, dest="graph2_qvec",
                   help="graph-graph: qvec prefix of the replica (beta=0)")
    c.add_argument("--graph2-perquery", default=None, dest="graph2_perquery",
                   help="graph-graph: per-query prefix of the replica, SAME job as its qvec")
    c.add_argument("--vision2-qvec", default=None, dest="vision2_qvec",
                   help="vision-vision: qvec prefix of the second vision run (gamma=0)")
    c.add_argument("--vision2-perquery", default=None, dest="vision2_perquery",
                   help="vision-vision: per-query prefix of the second run, SAME job as its qvec")
    c.add_argument("--vision2-hist", default=None, dest="vision2_hist")
    c.add_argument("--vision2-full-hist", default=None, dest="vision2_full_hist")
    c.add_argument("--graph2-hist", default=None, dest="graph2_hist")
    c.add_argument("--graph2-full-hist", default=None, dest="graph2_full_hist")
    c.add_argument("--gallery-names", default="results/shared_gallery.json", dest="gallery_names")
    c.add_argument("--fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    c.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    c.add_argument("--vision-hist", default=None, dest="vision_hist",
                   help="C4: prefix of the historical vision per-query files (descriptive)")
    c.add_argument("--graph-hist", default=None, dest="graph_hist",
                   help="C4: prefix of the historical graph per-query files (descriptive)")
    c.add_argument("--vision-full-hist", default=None, dest="vision_full_hist",
                   help="C3 full: historical vision full .npz (descriptive)")
    c.add_argument("--graph-full-hist", default=None, dest="graph_full_hist",
                   help="C3 full: historical graph full .npz (descriptive)")

    s = sub.add_parser("select", help="alpha*, verdict, oracle, full-plan cost")
    s.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    s.add_argument("--fixed-alpha-from", default=None, dest="fixed_alpha_from",
                   help="select json of the valid: skip the choice, use its alpha* (the test)")

    for sp in (c, s):
        sp.add_argument("--pair", choices=list(PAIRS), default="vision-graph",
                        help="vision-graph (§49, default) | graph-graph (§50) | vision-vision (§51)")
        sp.add_argument("--split", required=True, choices=["valid", "test"])
        sp.add_argument("--fusion-dir", required=True, dest="fusion_dir")
        sp.add_argument("--out-json", default=None, dest="out_json",
                        help="select default: results/fusion/select_<split>.json "
                             "(graph-graph: select_graphgraph_<split>.json, "
                             "vision-vision: select_visionvision_<split>.json)")

    k = sub.add_parser("complementarity",
                       help="§50/§51: vision+graph gain vs the gain of a control fusion")
    k.add_argument("--split", choices=["valid"], default="valid")
    k.add_argument("--control-pair", choices=["graph-graph", "vision-vision"], default="graph-graph",
                   dest="control_pair",
                   help="graph-graph (§50, three-way rule) | vision-vision (§51, four-way rule)")
    k.add_argument("--true-select", default="results/fusion/select_valid.json", dest="true_select")
    k.add_argument("--true-dir", default="results/perquery/fusion_valid", dest="true_dir")
    k.add_argument("--control-select", default=None, dest="control_select",
                   help="default: results/fusion/select_{graphgraph|visionvision}_valid.json")
    k.add_argument("--control-dir", default=None, dest="control_dir",
                   help="default: results/perquery/fusion_{graphgraph|visionvision}_valid")
    k.add_argument("--graphgraph-json", default="results/fusion/complementarity_valid.json",
                   dest="graphgraph_json", help="vision-vision: G_C of §50 is READ from here")
    k.add_argument("--graphgraph-dir", default="results/perquery/fusion_graphgraph_valid",
                   dest="graphgraph_dir", help="vision-vision: Spearman of graph/replica")
    k.add_argument("--out-json", default=None, dest="out_json",
                   help="default: results/fusion/complementarity_<split>.json "
                        "(vision-vision: complementarity_visionvision_<split>.json)")
    args = p.parse_args(argv)
    if args.cmd == "complementarity":
        tag = {"graph-graph": "graphgraph", "vision-vision": "visionvision"}[args.control_pair]
        if args.control_select is None:
            args.control_select = f"results/fusion/select_{tag}_{args.split}.json"
        if args.control_dir is None:
            args.control_dir = f"results/perquery/fusion_{tag}_{args.split}"
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.cmd == "check":
        report = cmd_check(args)
        if report["failed"]:
            raise SystemExit(1)
    elif args.cmd == "select":
        cmd_select(args)
    else:
        cmd_complementarity(args)


if __name__ == "__main__":
    main()
