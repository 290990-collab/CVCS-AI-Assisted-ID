"""Query-by-query diagnosis of a failed C3 (fusion endpoint vs the branch's own per-query).

C3 checks that alpha=1 (vision) / alpha=0 (graph) of a fusion reproduce the branch job's self_rr. The
branch job scored on GPU, the fusion on CPU: float32 rounding can swap the true plan with a competitor
whose similarity is equal to ~1e-6 (near-tie). C3 is waived only if every differing query is such a
near-tie, checked by recomputing the CPU similarities of the branch.

For each differing query: s = gallery @ query (CPU, the branch's own vectors and transform);
gap = |s(true plan) - s(plan at the job's rank)| in the CPU ranking. Near-tie iff gap <= GAP_TOL and both
ranks are within the top MAX_K.
`--boundary`: a query crossing the rank-MAX_K border (one rank within, the other beyond) is measured against
the plan at rank MAX_K+1 of the CPU ranking (gap = |s(true plan) - s(plan at rank MAX_K+1)|);
near-tie iff that gap <= GAP_TOL.

Usage (CPU, OMP_NUM_THREADS=4 on the login node):
    python -m src.evaluation.near_ties --branch vision-head --split test \
        --fusion-dir results/final_pipeline/fusion_head/s42/fusion_test \
        --qvec <VISION_QVEC> --perquery <VISION_PQ> \
        --out-json <dir>/c3_diagnosis_test.json [--waiver-out <dir>/c3_waiver_test.json]
`--waiver-out` writes the waiver (accepted) only if every differing query is a near-tie.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

from src.evaluation import late_fusion as lf
from src.evaluation.perquery import load_perquery
from src.evaluation.robustness_auc import fraction_path

GAP_TOL = 1e-5        # valid: max gap 5.8e-6, typical gap 1st-10th candidate 0.06-0.08
MAX_K = 100
RR_ATOL = 1e-6
BRANCHES = {          # branch -> (endpoint alpha, file strategy)
    "vision-head": (1.0, lf.VISION_STRATEGY),
    "vision": (1.0, lf.VISION_STRATEGY),
    "graph": (0.0, lf.GRAPH_STRATEGY),
}
CHUNK = 4096


def _gallery(branch: str, ref) -> np.ndarray:
    shared = lf.load_shared_names("results/shared_gallery.json")
    gv = ref.meta["gallery_vectors"]
    if branch == "graph":
        return lf.load_graph_gallery(gv["path"], shared, gv["sha1"], "graph")
    raw = lf.load_vision_gallery(gv["path"], shared, gv["sha1"])
    if branch == "vision":
        return lf.whiten(raw, ref.whiten_mean, ref.whiten_matrix)
    from src.evaluation import fusion_head as fh
    return np.concatenate([lf.whiten(fh.apply_head(raw[a:a + CHUNK]), ref.whiten_mean, ref.whiten_matrix)
                           for a in range(0, len(raw), CHUNK)])


def _queries(branch: str, d) -> np.ndarray:
    if branch == "graph":
        return np.asarray(d.vectors, np.float32)
    if branch == "vision":
        return lf.whiten(d.vectors, d.whiten_mean, d.whiten_matrix)
    from src.evaluation import fusion_head as fh
    return lf.whiten(fh.apply_head(d.vectors), d.whiten_mean, d.whiten_matrix)


def _rank(rr: float) -> int | None:
    return int(round(1 / rr)) if rr > 0 else None


def diagnose(branch: str, split: str, fusion_dir, qvec, perquery, fractions,
             pair: str = "vision-graph", alpha: float | None = None, boundary: bool = False) -> dict:
    """Per fraction: differing queries and their CPU gap, plus the overall verdict.

    `alpha` = fusion endpoint equal to this branch (default 1 vision, 0 graph; graph-graph: 1 = graph1, 0 = graph2).
    """
    default_alpha, strategy = BRANCHES[branch]
    alpha = default_alpha if alpha is None else float(alpha)
    qv = lf.load_branch_qvecs(qvec, split, fractions, strategy)
    if branch == "vision-head":
        from src.evaluation import fusion_head as fh
        fh._check_vision_head_meta(qv)
    gallery = _gallery(branch, qv[float(fractions[0])])
    per_fraction, all_ok = {}, True
    for f in fractions:
        d = qv[float(f)]
        job = load_perquery(fraction_path(perquery, f, split, strategy))
        fus = load_perquery(lf.fused_partial_path(fusion_dir, alpha, f, split, lf.fused_strategy(pair)))
        jidx = {str(n): i for i, n in enumerate(job.names)}
        vidx = {str(n): i for i, n in enumerate(d.names)}
        q = _queries(branch, d)
        rows = []
        for k, n in enumerate(fus.names):
            n = str(n)
            rj, rc = float(job.self_rr[jidx[n]]), float(fus.self_rr[k])
            if abs(rj - rc) <= RR_ATOL:
                continue
            i = vidx[n]
            qi = int(d.qi[i])
            s = gallery @ q[i]
            Rj, Rc = _rank(rj), _rank(rc)
            row = {"name": n, "rank_job": Rj, "rank_fusion": Rc,
                   "rank_cpu": 1 + int((s > s[qi]).sum())}
            order = np.sort(s)[::-1]
            if (Rj is None or Rc is None) and boundary:
                row.update({"gap": float(abs(s[qi] - order[MAX_K])), "boundary": True,
                            "note": f"confine del rango {MAX_K}: gap con la pianta al rango {MAX_K + 1}"})
                row["near_tie"] = bool(row["gap"] <= GAP_TOL)
            elif Rj is None or Rc is None:
                row.update({"gap": None, "near_tie": False, "note": f"oltre il rango {MAX_K}"})
            else:
                row["gap"] = float(abs(s[qi] - order[Rj - 1]))
                row["near_tie"] = bool(row["gap"] <= GAP_TOL)
            rows.append(row)
        gaps = [r["gap"] for r in rows if r["gap"] is not None]
        ok = all(r["near_tie"] for r in rows)
        all_ok = all_ok and ok
        per_fraction[f"{float(f)}"] = {
            "n": int(len(fus.names)), "n_differing": len(rows), "all_near_ties": ok,
            "max_gap": max(gaps) if gaps else None,
            "max_rank_jump": max((abs(r["rank_job"] - r["rank_fusion"]) for r in rows
                                  if r["rank_job"] and r["rank_fusion"]), default=0),
            "n_exact_f32_tie": sum(1 for g in gaps if g == 0.0), "rows": rows}
        print(f"[near_ties] {branch} f={f}: diverse {len(rows)}/{len(fus.names)} · "
              f"gap max {per_fraction[f'{float(f)}']['max_gap']} · tutte pareggi quasi esatti: {ok}")
    gaps = [p["max_gap"] for p in per_fraction.values() if p["max_gap"] is not None]
    return {"branch": branch, "split": split, "pair": pair, "alpha_endpoint": alpha, "fusion_dir": str(fusion_dir),
            "qvec": str(qvec), "perquery": str(perquery), "gap_tol": GAP_TOL, "boundary": boundary,
            "n_differing_total": sum(p["n_differing"] for p in per_fraction.values()),
            "max_gap": max(gaps) if gaps else None, "all_near_ties": all_ok,
            "per_fraction": per_fraction,
            "method": "CPU scores of the branch's own vectors (head and whitening as the fusion); "
                      "gap = |score(true plan) - score(plan at the job's rank)|"
                      + (f"; across the rank-{MAX_K} border: score of the plan at rank {MAX_K + 1} "
                         "(status.md §62, extension of 5 Oct)" if boundary else ""),
            "written": datetime.now().isoformat(timespec="seconds")}


def waiver(diag: dict) -> dict:
    """C3 waiver json read by `graph_config_select fusion-check` (only if all differences are near-ties)."""
    if not diag["all_near_ties"]:
        raise ValueError("not every differing query is a near-tie: no waiver, diagnose first")
    return {"accepted": True, "date": date.today().isoformat(),
            "approved_by": "users, test pre-registration status.md §62 (near-tie waiver, query by query, "
                           "as §59.H.1)",
            "rule": f"C3 alpha={diag['alpha_endpoint']:g} vs {diag['branch']}: every differing self_rr is a "
                    f"near-tie (gap <= {GAP_TOL:g}) in the CPU recomputation",
            "reason": f"{diag['n_differing_total']} differing queries, all near-ties, max gap {diag['max_gap']}",
            "diagnosis": {"per_fraction": {f: {k: v for k, v in p.items() if k != "rows"}
                                           for f, p in diag["per_fraction"].items()},
                          "n_differing_total": diag["n_differing_total"], "max_gap": diag["max_gap"],
                          "method": diag["method"]}}


def _write_once(path: Path, obj) -> None:
    if path.exists():
        raise SystemExit(f"!! {path} esiste gia': niente sovrascritture")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True))
    print(f"[near_ties] scritto {path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="near_ties", description=__doc__.split("\n")[0])
    ap.add_argument("--branch", required=True, choices=list(BRANCHES))
    ap.add_argument("--split", required=True, choices=["valid", "test"])
    ap.add_argument("--fusion-dir", required=True)
    ap.add_argument("--qvec", required=True, help="prefix of the branch qvec files")
    ap.add_argument("--perquery", required=True, help="prefix of the branch per-query files")
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75])
    ap.add_argument("--pair", default="vision-graph", choices=["vision-graph", "graph-graph"])
    ap.add_argument("--alpha", type=float, default=None, help="endpoint (default: 1 vision, 0 graph)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--waiver-out", default=None)
    ap.add_argument("--boundary", action="store_true",
                    help=f"measure queries crossing the rank-{MAX_K} border (status.md §62, 5 Oct)")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    diag = diagnose(a.branch, a.split, a.fusion_dir, a.qvec, a.perquery, a.fractions, a.pair, a.alpha, a.boundary)
    _write_once(Path(a.out_json), diag)
    print(f"[near_ties] {diag['n_differing_total']} query diverse · tutte pareggi quasi esatti: "
          f"{diag['all_near_ties']} (gap max {diag['max_gap']}, soglia {GAP_TOL:g})")
    if a.waiver_out:
        if not diag["all_near_ties"]:
            raise SystemExit("!! non tutte le differenze sono pareggi quasi esatti: nessuna deroga")
        _write_once(Path(a.waiver_out), waiver(diag))


if __name__ == "__main__":
    main()
