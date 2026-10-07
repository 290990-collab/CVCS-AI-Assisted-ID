"""Checks, ranking and choice of the graph configuration (stage 1 / stage 2 / final), plus fusion and gate helpers.

Commands (CPU):

  status        how many of the expected trainings/evaluations are complete
  check-runs    checks C1-C8 on an explicit list of runs (smoke test), nothing written
  stage1        checks C1-C8 on the 34 x 4 runs, ranking, "netta" changes per encoder
                -> selection/stage1.json + selection/stage2.json (the `comb` to train)
  show-stage2   prints the comb of each encoder (or "nessuna")
  final         candidates = stage 1 + combs -> W (winner), ties, tie-break, E
                -> selection/final.json
  fusion-paths  shell assignments of the fusion inputs/outputs of a replica (03-05)
  fusion-check  rule on a `fusion_select check` json (C5 = MRR f=0.0 >= 0.90)
  crossenc      four-way verdict of the cross-encoder control
  test-gate / ctrl-gate / ctrltest-gate   guards of 02_eval_graph.sh (exit 1 = not allowed)

Rules:
- score of a configuration = mean over the 4 seeds of the AUC (self_rr, `random`,
  f in {0.25, 0.5, 0.75}, valid); replica S of every configuration has the same
  queries and removed rooms (damage seed S) -> paired by name;
- A clearly beats B iff the bootstrap CI 95% (B=10000, seed 0) of
  d(q) = mean_S [A_S(q) - B_S(q)] is entirely > 0 and A beats B in every replica;
  otherwise "pari";
- W = highest score; ties = configurations W does not clearly beat; tie-break:
  topology nDCG@10 on the full plan (mean of 4 seeds, same rule) -> fewest
  parameters -> highest score;
- stage 2: a change is "netta" iff it clearly beats `ref` of its encoder; per
  factor the netta with the highest score is kept; >= 2 kept -> `comb`;
- E = highest score among the configurations of an encoder different from W's.
Checks verify the pipeline, not the model: no band calibrated on a model, MRR at f=0.0 has only a lower bound (0.90).

Every json of selection/ is written once; rerunning a command that would rewrite it stops.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.graph import final_graph_configs as rc
from src.evaluation.perquery import load_perquery
from src.evaluation.query_vectors import load_qvec
from src.evaluation.relevance import AXES
from src.evaluation.robustness_auc import AUC_FRACTIONS, fraction_path, load_auc
from src.evaluation.significance import bootstrap_ci

GALLERY_SHA1_PREFIX = "0c24cfc05e18"   # shared gallery
N_QUERIES = 2000
MRR_F0_MIN = 0.90                      # C4 / fusion C5: lower bound only (7.5)
FULL_K = 10
TIEBREAK_AXIS = "topology"

VISION_TAG = "vision_pespatial_gem_whiten-train"   # frozen vision config, read only
VISIONVISION_DIR = "results/perquery/fusion_visionvision_valid"
VISIONVISION_SELECT = "results/fusion/select_visionvision_valid.json"
CONTROLS = ("replica", "crossenc", "visionvision")
# besides W, only these configs (seed 42 only) go to the test: encoder control and reduced curve
TEST_EXTRA = (("graph_sage", "comb"), ("gcn", "ref"), ("gat", "t05"), ("gat", "ref"), ("gat", "t01"))
TEST_EXTRA_SEED = 42


# --- loading one run ---

def yaml_name(key: str) -> str:
    """Registry key (gcn | sage | gat) -> YAML basename (gcn | graph_sage | gat)."""
    inv = {v: k for k, v in rc.ENCODERS.items()}
    if key not in inv:
        raise ValueError(f"unknown encoder key {key!r}")
    return inv[key]


@dataclass
class RunData:
    key: str
    cfg: str
    seed: int
    names: np.ndarray                  # [P] queries valid in all fractions
    auc: np.ndarray                    # [P]
    per_fraction: dict                 # f -> mean self_rr
    mrr_f0: float
    full_names: np.ndarray             # [Q]
    full: dict                         # axis -> [Q] nDCG@10 (NaN = singleton)
    best_epoch: int
    epochs_run: int
    n_params: int | None
    failures: list = field(default_factory=list)

    @property
    def score(self) -> float:
        return float(self.auc.mean()) if len(self.auc) else float("nan")


def run_files(key: str, cfg: str, seed: int, smoke: bool = False) -> dict:
    """Expected files of one (configuration, seed) on the valid."""
    p = rc.run_paths(yaml_name(key), cfg, seed, "valid", smoke)
    pq, qv, tag = Path(p["PQ_DIR"]), Path(p["QV_DIR"]), p["TAG"]
    return {
        "dest": Path(p["DEST"]),
        "summary": Path(p["DEST"]) / "training_summary.json",
        "recipe": Path(p["DEST"]) / "reset_recipe.json",
        "encoder": Path(p["DEST"]) / "encoder.pt",
        "prefix": pq / tag,
        "full": pq / f"{tag}_full_valid.npz",
        "partial": {f: fraction_path(pq / tag, f, "valid") for f in rc.FRACTIONS},
        "qvec": {f: fraction_path(qv / tag, f, "valid") for f in rc.FRACTIONS},
    }


def run_state(key: str, cfg: str, seed: int, smoke: bool = False) -> str:
    """missing | training (folder without checkpoint) | trained | evaluated."""
    f = run_files(key, cfg, seed, smoke)
    if not f["dest"].exists():
        return "missing"
    if not (f["encoder"].exists() and f["summary"].exists()):
        return "training"
    evaluated = [f["full"], *f["partial"].values(), *f["qvec"].values()]
    return "evaluated" if all(p.exists() for p in evaluated) else "trained"


def count_params(path: Path) -> int | None:
    """Parameters of a checkpoint (sum of numel of the state dict)."""
    try:
        import torch
    except ImportError:                 # pragma: no cover - env without torch
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    return int(sum(v.numel() for v in state.values()))


def load_run(key: str, cfg: str, seed: int, smoke: bool = False, params: bool = True) -> tuple[RunData, dict]:
    """RunData + the raw material of the cross-run checks (qvec removed rooms, names)."""
    f = run_files(key, cfg, seed, smoke)
    summary = json.loads(f["summary"].read_text())
    auc = load_auc(f["prefix"], "valid", strategy="random")
    f0 = load_perquery(f["partial"][0.0])
    full = load_perquery(f["full"])
    ki = full.k_index(FULL_K)
    run = RunData(
        key=key, cfg=cfg, seed=seed, names=auc.names, auc=auc.auc,
        per_fraction={float(x): float(np.mean(v)) for x, v in auc.per_fraction.items()},
        mrr_f0=float(np.nanmean(f0.self_rr)),
        full_names=np.asarray([str(n) for n in full.names]),
        full={ax: np.asarray(full.ndcg[full.axis_index(ax), ki], dtype=float) for ax in AXES},
        best_epoch=int(summary.get("best_epoch", -1)),
        epochs_run=int(summary.get("epochs_run", -1)),
        n_params=count_params(f["encoder"]) if params else None,
    )
    qvecs = {x: load_qvec(p) for x, p in f["qvec"].items()}
    files_meta = [load_perquery(p).meta for p in f["partial"].values()] + [full.meta]
    raw = {"summary": summary, "qvecs": qvecs, "files_meta": files_meta, "n_ref": auc.n_ref,
           "recipe": json.loads(f["recipe"].read_text()) if f["recipe"].exists() else None,
           "partial_names": {str(n) for n in f0.names}}
    return run, raw


# --- checks C1-C8 (pipeline, not model) ---

def removed_map(q) -> dict:
    return {str(n): tuple(q.removed(i)) for i, n in enumerate(q.names)}


def check_run(run: RunData, raw: dict, ref_names: set, ref_removed: dict | None,
              smoke: bool = False) -> list[str]:
    """Failures of one run (empty list = PASS).

    ref_names: the query set every file must have (C2).
    ref_removed: f -> {name: removed rooms} of the reference run of the SAME seed (C3),
        or None for the reference itself.
    """
    out = []
    s, qv = raw["summary"], raw["qvecs"]
    sha = [m.get("gallery", {}).get("sha1", "") for m in raw["files_meta"]]
    sha += [q.meta.get("gallery", {}).get("sha1", "") for q in qv.values()]
    if not all(str(x).startswith(GALLERY_SHA1_PREFIX) for x in sha):
        out.append(f"C1 gallery sha1 diversa da {GALLERY_SHA1_PREFIX}")
    if raw["n_ref"] != N_QUERIES or raw["partial_names"] != ref_names \
            or set(run.full_names) != ref_names:
        out.append(f"C2 query: {raw['n_ref']} (attese {N_QUERIES}) o insieme diverso dal riferimento")
    for f, q in qv.items():
        if int(q.meta.get("partial_seed", -1)) != int(run.seed):
            out.append(f"C3 f={f}: partial_seed {q.meta.get('partial_seed')} != seed {run.seed}")
        elif ref_removed is not None and removed_map(q) != ref_removed[f]:
            out.append(f"C3 f={f}: stanze tolte diverse dalla configurazione di riferimento (stesso seed)")
    if not run.mrr_f0 >= MRR_F0_MIN:
        out.append(f"C4 MRR f=0.0 {run.mrr_f0:.4f} < {MRR_F0_MIN}")
    if (s.get("probe_partial") or {}).get("overlap") != 0:
        out.append(f"C5 sovrapposizione sonda/valutazione {(s.get('probe_partial') or {}).get('overlap')}")
    if any(int(q.meta.get("n_degenerate", -1)) != 0 for q in qv.values()) or len(run.names) != raw["n_ref"]:
        out.append("C6 query svuotate o senza self_rr")
    if run.epochs_run != rc.epochs(smoke) or s.get("selection_probe") != "partial":
        out.append(f"C7 epochs_run {run.epochs_run} (attese {rc.epochs(smoke)}) o selezione non 'partial'")
    out += check_recipe(run, raw, smoke)
    return out


def check_recipe(run: RunData, raw: dict, smoke: bool = False) -> list[str]:
    """C8: the run was trained with the configuration of the table."""
    rec, s = raw["recipe"], raw["summary"]
    if rec is None:
        return ["C8 reset_recipe.json mancante"]
    exp = rc.merged_config(yaml_name(run.key), run.cfg, run.seed, smoke)
    bad = []
    if rec.get("overrides") != rc.config_overrides(yaml_name(run.key), run.cfg, smoke):
        bad.append("overrides")
    hp = s.get("hyperparams") or {}
    for k in ("temperature", "hidden_dim", "num_layers", "pooling", "batch_size", "lr"):
        if k in hp and hp[k] != exp[k]:
            bad.append(k)
    if s.get("pair_mode") != "asym_partial" or bool(s.get("lost_marker")) != bool(exp.get("lost_marker", False)):
        bad.append("pair_mode/lost_marker")
    if int(s.get("epochs_max", -1)) != rc.epochs(smoke) or s.get("variant") != exp["variant"]:
        bad.append("epochs_max/variant")
    return [f"C8 ricetta diversa dalla tabella: {', '.join(bad)}"] if bad else []


# --- pure rules (tested in tests/test_final_pipeline.py) ---

def _aligned(runs: dict, which: str, axis: str | None = None):
    """name -> value per seed, for `auc` or the full `axis` (NaN dropped)."""
    out = {}
    for s, r in runs.items():
        if which == "auc":
            names, vals = r.names, r.auc
        else:
            names, vals = r.full_names, r.full[axis]
        out[s] = {str(n): float(v) for n, v in zip(names, vals) if not np.isnan(v)}
    return out


def paired(a: dict, b: dict, which: str = "auc", axis: str | None = None) -> dict:
    """A - B over the 4 replicas, paired by seed and by query name.

    Args: a, b: seed -> RunData (same seeds). Queries = common to all seeds of both.
    Returns: delta (mean of d(q)), ci_lo/ci_hi (bootstrap), per_seed deltas, n.
    """
    seeds = sorted(a)
    if sorted(b) != seeds:
        raise ValueError(f"seeds differ: {seeds} vs {sorted(b)}")
    va, vb = _aligned(a, which, axis), _aligned(b, which, axis)
    names = set.intersection(*(set(va[s]) for s in seeds), *(set(vb[s]) for s in seeds))
    names = sorted(names)
    diff = np.asarray([[va[s][n] - vb[s][n] for n in names] for s in seeds], dtype=float)
    d = diff.mean(axis=0) if len(names) else np.zeros(0)
    lo, hi = bootstrap_ci(d)
    return {"delta": float(d.mean()) if len(d) else float("nan"), "ci_lo": lo, "ci_hi": hi,
            "per_seed": {str(s): float(diff[i].mean()) for i, s in enumerate(seeds)},
            "n": len(names)}


def clearly_better(cmp: dict) -> bool:
    """CI entirely above 0 and better in every replica."""
    return bool(cmp["ci_lo"] > 0.0 and all(v > 0.0 for v in cmp["per_seed"].values()))


def config_summary(runs: dict) -> dict:
    """Score and descriptives of one configuration (seed -> RunData)."""
    scores = [r.score for r in runs.values()]
    first = runs[min(runs)]
    return {
        "score": float(np.mean(scores)),
        "score_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "per_seed": {str(s): r.score for s, r in sorted(runs.items())},
        "per_fraction": {f"{f:g}": float(np.mean([r.per_fraction[f] for r in runs.values()]))
                         for f in AUC_FRACTIONS},
        "mrr_f0": float(np.mean([r.mrr_f0 for r in runs.values()])),
        "full_ndcg10": {ax: float(np.mean([np.nanmean(r.full[ax]) for r in runs.values()]))
                        for ax in AXES},
        "n_params": first.n_params,
        "best_epochs": {str(s): r.best_epoch for s, r in sorted(runs.items())},
        "budget_binding": sum(r.best_epoch > rc.BINDING_FRACTION * r.epochs_run for r in runs.values()),
    }


def ranking(summaries: dict) -> list:
    """(key, cfg) sorted by score desc; exact ties broken by name (deterministic)."""
    return sorted(summaries, key=lambda kc: (-summaries[kc]["score"], kc[0], kc[1]))


def netta_changes(key: str, cfgs, runs: dict, summaries: dict) -> dict:
    """Stage 2: changes that clearly beat `ref` of their encoder, best per factor.

    Returns: {"tested": {cfg: comparison}, "netta": [cfg], "kept": [cfg], "overrides": {...} | None}.
    """
    ref = runs[(key, "ref")]
    tested, netta = {}, []
    for cfg in cfgs:
        if cfg == "ref":
            continue
        cmp = paired(runs[(key, cfg)], ref)
        tested[cfg] = {**cmp, "clearly_better": clearly_better(cmp)}
        if tested[cfg]["clearly_better"]:
            netta.append(cfg)
    best = {}
    for cfg in netta:
        fac = rc.FACTOR[cfg]
        if fac not in best or summaries[(key, cfg)]["score"] > summaries[(key, best[fac])]["score"]:
            best[fac] = cfg
    kept = sorted(best.values(), key=lambda c: list(cfgs).index(c))
    overrides = None
    if len(kept) >= 2:
        overrides = {}
        for cfg in kept:
            overrides.update(rc.OVERRIDES[cfg])
    return {"tested": tested, "netta": netta, "kept": kept, "overrides": overrides}


def choose_winner(runs: dict, summaries: dict) -> dict:
    """W, its ties, the tie-break and E."""
    order = ranking(summaries)
    top = order[0]
    vs_top = {}
    ties = []
    for kc in order[1:]:
        cmp = paired(runs[top], runs[kc])
        cmp["clearly_better"] = clearly_better(cmp)
        vs_top[f"{kc[0]}/{kc[1]}"] = cmp
        if not cmp["clearly_better"]:
            ties.append(kc)

    tiebreak = None
    winner = top
    if ties:
        group = [top] + ties
        topo = {kc: summaries[kc]["full_ndcg10"][TIEBREAK_AXIS] for kc in group}
        t1 = sorted(group, key=lambda kc: (-topo[kc], kc[0], kc[1]))[0]
        still, topo_cmp = [t1], {}
        for kc in group:
            if kc == t1:
                continue
            cmp = paired(runs[t1], runs[kc], which="full", axis=TIEBREAK_AXIS)
            cmp["clearly_better"] = clearly_better(cmp)
            topo_cmp[f"{kc[0]}/{kc[1]}"] = cmp
            if not cmp["clearly_better"]:
                still.append(kc)
        params = {kc: summaries[kc]["n_params"] for kc in still}
        if any(v is None for v in params.values()):
            raise ValueError("tie-break: parameter count unavailable (torch missing?)")
        min_p = min(params.values())
        cheapest = [kc for kc in still if params[kc] == min_p]
        winner = sorted(cheapest, key=lambda kc: (-summaries[kc]["score"], kc[0], kc[1]))[0]
        tiebreak = {
            "group": [f"{k}/{c}" for k, c in group],
            "topology_means": {f"{k}/{c}": v for (k, c), v in topo.items()},
            "topology_best": f"{t1[0]}/{t1[1]}",
            "topology_vs_best": topo_cmp,
            "still_tied_after_topology": [f"{k}/{c}" for k, c in still],
            "n_params": {f"{k}/{c}": v for (k, c), v in params.items()},
            "fewest_params": [f"{k}/{c}" for k, c in cheapest],
        }
    others = [kc for kc in order if kc[0] != winner[0]]
    e = others[0] if others else None
    return {"top_by_score": top, "winner": winner, "ties": ties, "vs_top": vs_top,
            "tiebreak": tiebreak, "E": e}


def fusion_check_verdict(report: dict, waiver: dict | None = None) -> dict:
    """Rule on a `fusion_select check` report: C1-C3 PASS (C3 may be waived), C5 = MRR f=0.0 >= MRR_F0_MIN for every alpha."""
    status = {k: (report.get(k) or {}).get("status") for k in ("C1", "C2", "C3", "C5")}
    c3_waived = bool(waiver and waiver.get("accepted") is True and status["C3"] != "PASS")
    per_alpha = (report.get("C5") or {}).get("per_alpha") or {}
    c5 = {a: float(v["mrr"]) for a, v in per_alpha.items()}
    c5_ok = bool(c5) and all(v >= MRR_F0_MIN for v in c5.values())
    failed = [k for k in ("C1", "C2") if status[k] != "PASS"]
    if status["C3"] != "PASS" and not c3_waived:
        failed.append("C3")
    if not c5_ok:
        failed.append("C5")
    return {"failed": failed, "pass": not failed, "status_fusion_select": status,
            "c3_waived": c3_waived, "c5_mrr_f0": c5, "c5_rule": f">= {MRR_F0_MIN} (solo limite basso)"}


# --- loading the whole stage ---

def expected_stage1(smoke: bool = False, encoders=None) -> list:
    out = []
    for enc in (encoders or rc.ENCODERS):
        for cfg in rc.stage1_configs(enc):
            out.append((rc.encoder_key(enc), cfg))
    return out


def load_configs(pairs, seeds, smoke: bool = False) -> tuple[dict, dict]:
    """(key, cfg) -> seed -> RunData, plus failures per run. Reference for C2/C3 = first
    configuration of the list, per seed."""
    runs, failures = {}, {}
    ref_names, ref_removed = None, {}
    for kc in pairs:
        runs[kc] = {}
        for s in seeds:
            run, raw = load_run(kc[0], kc[1], s, smoke)
            if ref_names is None:
                ref_names = raw["partial_names"]
            ref_rm = ref_removed.get(s)
            fails = check_run(run, raw, ref_names, ref_rm, smoke)
            if ref_rm is None:
                ref_removed[s] = {f: removed_map(q) for f, q in raw["qvecs"].items()}
            runs[kc][s] = run
            if fails:
                failures[f"{kc[0]}/{kc[1]}/s{s}"] = fails
    return runs, failures


def _seeds(smoke: bool, args) -> tuple:
    return tuple(args.seeds) if getattr(args, "seeds", None) else rc.SEEDS


def _sel_dir(smoke: bool) -> Path:
    return rc.root(smoke) / "selection"


def _write_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise SystemExit(f"!! {path} esiste gia': niente sovrascritture (spostalo a mano se va rifatto)")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv, **payload}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(f"[graph_config_select] scritto {path}")


def _parse_excluded(items) -> set:
    out = set()
    for it in items or []:
        key, _, cfg = it.partition("/")
        out.add((key, cfg))
    return out


# --- commands ---

def cmd_status(args) -> None:
    seeds = _seeds(args.smoke, args)
    pairs = expected_stage1(args.smoke)
    st2 = rc.stage2_path(args.smoke)
    if st2.exists():
        for key, entry in json.loads(st2.read_text())["combs"].items():
            if entry:
                pairs.append((key, rc.COMB))
    counts = {"missing": 0, "training": 0, "trained": 0, "evaluated": 0}
    lines = []
    for key, cfg in pairs:
        states = [run_state(key, cfg, s, args.smoke) for s in seeds]
        for x in states:
            counts[x] += 1
        if any(x != "evaluated" for x in states):
            lines.append(f"  {key}/{cfg}: " + " ".join(f"s{s}={x}" for s, x in zip(seeds, states)))
    total = len(pairs) * len(seeds)
    print(f"[status] {counts['evaluated']}/{total} valutate · {counts['trained']} allenate non valutate · "
          f"{counts['training']} in corso o fallite (cartella senza checkpoint) · {counts['missing']} mai partite")
    for line in lines:
        print(line)


def cmd_check_runs(args) -> None:
    """Checks C1-C8 on an explicit list of runs (the smoke test), no ranking, nothing written."""
    seeds = _seeds(args.smoke, args)
    pairs = []
    for item in args.runs:
        key, _, cfg = item.partition("/")
        pairs.append((key, cfg))
    runs, failures = load_configs(pairs, seeds, args.smoke)
    for kc, by_seed in runs.items():
        for s, r in by_seed.items():
            tag = f"{kc[0]}/{kc[1]}/s{s}"
            print(f"{tag:<22} AUC {r.score:.4f} · MRR f=0 {r.mrr_f0:.4f} · epoche {r.epochs_run} "
                  f"(scelta {r.best_epoch}) · parametri {r.n_params} · "
                  f"{'FAIL ' + '; '.join(failures[tag]) if tag in failures else 'controlli OK'}")
    if failures:
        raise SystemExit(1)


def _print_ranking(summaries: dict) -> None:
    print(f"{'#':>3} {'config':<16} {'AUC':>7} {'sd':>7}  {'f.25':>6} {'f.5':>6} {'f.75':>6}  "
          f"{'C':>6} {'T':>6} {'G':>6}  {'param':>8} {'vinc':>4}")
    for i, kc in enumerate(ranking(summaries), 1):
        s = summaries[kc]
        pf, fu = s["per_fraction"], s["full_ndcg10"]
        print(f"{i:>3} {kc[0] + '/' + kc[1]:<16} {s['score']:>7.4f} {s['score_sd']:>7.4f}  "
              f"{pf['0.25']:>6.3f} {pf['0.5']:>6.3f} {pf['0.75']:>6.3f}  "
              f"{fu['composition']:>6.3f} {fu['topology']:>6.3f} {fu['geometry']:>6.3f}  "
              f"{str(s['n_params']):>8} {s['budget_binding']:>2}/{len(s['best_epochs'])}")


def cmd_stage1(args) -> None:
    seeds = _seeds(args.smoke, args)
    excluded = _parse_excluded(args.exclude)
    encoders = args.encoders or list(rc.ENCODERS)
    pairs = [kc for kc in expected_stage1(args.smoke, encoders) if kc not in excluded]
    incomplete = [f"{k}/{c}/s{s}" for k, c in pairs for s in seeds
                  if run_state(k, c, s, args.smoke) != "evaluated"]
    if incomplete:
        raise SystemExit(f"!! stage1: {len(incomplete)} run non complete (es. {incomplete[:5]}); "
                         "vedi `graph_config_select status`")
    runs, failures = load_configs(pairs, seeds, args.smoke)
    if failures:
        for k, v in failures.items():
            print(f"FAIL {k}: {'; '.join(v)}")
        raise SystemExit("!! stage1: controlli falliti -> diagnosi prima di ogni classifica "
                         "(una config si esclude solo con --exclude <key>/<cfg>, dichiarato)")
    summaries = {kc: config_summary(r) for kc, r in runs.items()}
    _print_ranking(summaries)
    combs, tests = {}, {}
    for enc in encoders:
        key = rc.encoder_key(enc)
        cfgs = [c for c in rc.stage1_configs(enc) if (key, c) not in excluded]
        if (key, "ref") not in runs:
            raise SystemExit(f"!! {key}/ref escluso: la fase 2 di {key} non e' definibile")
        res = netta_changes(key, cfgs, runs, summaries)
        tests[key] = res
        combs[key] = {"cfgs": res["kept"], "overrides": res["overrides"]} if res["overrides"] else None
        print(f"\n[{key}] modifiche nette vs ref: {res['netta'] or 'nessuna'} · tenute per fattore: "
              f"{res['kept'] or 'nessuna'} -> comb: {combs[key]['cfgs'] if combs[key] else 'NESSUNA'}")
        for cfg, cmp in res["tested"].items():
            print(f"    {cfg:<7} − ref {cmp['delta']:+.4f} [{cmp['ci_lo']:+.4f}, {cmp['ci_hi']:+.4f}] · "
                  f"per seed {' '.join(f'{v:+.4f}' for v in cmp['per_seed'].values())} · "
                  f"{'NETTA' if cmp['clearly_better'] else '-'}")
    common = {"seeds": list(seeds), "excluded": sorted(f"{k}/{c}" for k, c in excluded),
              "encoders": [rc.encoder_key(e) for e in encoders]}
    _write_once(_sel_dir(args.smoke) / "stage1.json",
                {**common, "summaries": {f"{k}/{c}": v for (k, c), v in summaries.items()},
                 "ranking": [f"{k}/{c}" for k, c in ranking(summaries)], "stage2_tests": tests})
    _write_once(rc.stage2_path(args.smoke), {**common, "combs": combs,
                                            "rule": "VERIFICA_GRAFI.md §7.4 (fase 2)"})


def cmd_show_stage2(args) -> None:
    combs = json.loads(rc.stage2_path(args.smoke).read_text())["combs"]
    for key in ("gcn", "sage", "gat"):
        if key in combs:
            e = combs[key]
            print(f"{yaml_name(key)}: " + (f"comb = {' + '.join(e['cfgs'])} {e['overrides']}" if e else "nessuna"))


def cmd_final(args) -> None:
    seeds = _seeds(args.smoke, args)
    st1 = json.loads((_sel_dir(args.smoke) / "stage1.json").read_text())
    st2 = json.loads(rc.stage2_path(args.smoke).read_text())
    excluded = _parse_excluded(st1["excluded"])
    pairs = [kc for kc in expected_stage1(args.smoke, [yaml_name(k) for k in st1["encoders"]])
             if kc not in excluded]
    pairs += [(k, rc.COMB) for k, e in st2["combs"].items() if e]
    incomplete = [f"{k}/{c}/s{s}" for k, c in pairs for s in seeds
                  if run_state(k, c, s, args.smoke) != "evaluated"]
    if incomplete:
        raise SystemExit(f"!! final: {len(incomplete)} run non complete (es. {incomplete[:5]})")
    runs, failures = load_configs(pairs, seeds, args.smoke)
    if failures:
        for k, v in failures.items():
            print(f"FAIL {k}: {'; '.join(v)}")
        raise SystemExit("!! final: controlli falliti -> diagnosi")
    summaries = {kc: config_summary(r) for kc, r in runs.items()}
    _print_ranking(summaries)
    ch = choose_winner(runs, summaries)
    w, e = ch["winner"], ch["E"]
    print(f"\nprimo per AUC: {ch['top_by_score'][0]}/{ch['top_by_score'][1]} · "
          f"pari: {[f'{k}/{c}' for k, c in ch['ties']] or 'nessuno'}")
    if ch["tiebreak"]:
        print(f"spareggio: {json.dumps(ch['tiebreak'], indent=1, default=str)}")
    print(f"VINCITORE W = {w[0]}/{w[1]} ({yaml_name(w[0])} {w[1]}) · "
          f"E (controllo encoder diversi) = {e[0] + '/' + e[1] if e else 'nessuno'}")
    out = {
        "seeds": list(seeds), "excluded": st1["excluded"],
        "W": {"key": w[0], "encoder": yaml_name(w[0]), "cfg": w[1], **summaries[w]},
        "E": ({"key": e[0], "encoder": yaml_name(e[0]), "cfg": e[1], **summaries[e]} if e else None),
        "top_by_score": f"{ch['top_by_score'][0]}/{ch['top_by_score'][1]}",
        "ties": [f"{k}/{c}" for k, c in ch["ties"]],
        "vs_top": ch["vs_top"], "tiebreak": ch["tiebreak"],
        "summaries": {f"{k}/{c}": v for (k, c), v in summaries.items()},
        "ranking": [f"{k}/{c}" for k, c in ranking(summaries)],
        "best_per_encoder": {k: next(f"{kk}/{cc}" for kk, cc in ranking(summaries) if kk == k)
                             for k in sorted({kc[0] for kc in summaries})},
        "rule": "VERIFICA_GRAFI.md §7.5",
    }
    _write_once(_sel_dir(args.smoke) / "final.json", out)


def _final(smoke: bool) -> dict:
    path = _sel_dir(smoke) / "final.json"
    if not path.exists():
        raise SystemExit(f"!! {path} mancante: prima `graph_config_select final`")
    return json.loads(path.read_text())


def vision_prefixes(seed: int, split: str) -> tuple[str, str]:
    """qvec and per-query prefixes of the frozen vision (read only)."""
    if int(seed) == 42:
        return (f"results/queryvec/{split}/{VISION_TAG}",
                f"results/perquery/fusion_branches_{split}/{VISION_TAG}")
    return (f"results/queryvec/seeds/s{seed}/{split}/{VISION_TAG}",
            f"results/perquery/seeds/s{seed}/fusion_branches_{split}/{VISION_TAG}")


def fusion_paths(seed: int, split: str, smoke: bool = False) -> dict:
    """Everything 03/04/05 need for one replica (and the controls)."""
    fin = _final(smoke)
    w, e = fin["W"], fin["E"]
    r = rc.root(smoke)
    wp = rc.run_paths(w["encoder"], w["cfg"], seed, split, smoke)
    vq, vp = vision_prefixes(seed, split)
    fdir = r / "fusion" / f"s{seed}"
    out = {
        "W_ENC": w["encoder"], "W_CFG": w["cfg"], "W_KEY": w["key"],
        "GRAPH_QVEC": f"{wp['QV_DIR']}/{wp['TAG']}", "GRAPH_PQ": f"{wp['PQ_DIR']}/{wp['TAG']}",
        "VISION_QVEC": vq, "VISION_PQ": vp,
        "FUSION_DIR": str(fdir / f"fusion_{split}"),
        "CHECK_JSON": str(fdir / f"check_{split}.json"),
        "CHECK_RESET_JSON": str(fdir / f"check_{split}_reset.json"),
        "WAIVER_JSON": str(fdir / f"c3_waiver_{split}.json"),
        "SELECT_JSON": str(fdir / f"select_{split}.json"),
        "SELECT_VALID_JSON": str(fdir / "select_valid.json"),
        "CTRL_ROOT": str(r / "controls"),
        "MAIN_FUSION_DIR": str(r / "fusion" / "s42" / "fusion_valid"),
        "MAIN_SELECT_VALID": str(r / "fusion" / "s42" / "select_valid.json"),
        "VV_DIR": VISIONVISION_DIR, "VV_SELECT": VISIONVISION_SELECT,
    }
    w42 = rc.run_paths(w["encoder"], w["cfg"], 42, "valid", smoke)
    rep = rc.run_paths(w["encoder"], w["cfg"], rc.CTRL_REPLICA, "ctrl", smoke)
    out.update({"W42_QVEC": f"{w42['QV_DIR']}/{w42['TAG']}", "W42_PQ": f"{w42['PQ_DIR']}/{w42['TAG']}",
                "REP_QVEC": f"{rep['QV_DIR']}/{rep['TAG']}", "REP_PQ": f"{rep['PQ_DIR']}/{rep['TAG']}"})
    if e:
        ep = rc.run_paths(e["encoder"], e["cfg"], 42, "valid", smoke)
        out.update({"E_ENC": e["encoder"], "E_CFG": e["cfg"],
                    "E_QVEC": f"{ep['QV_DIR']}/{ep['TAG']}", "E_PQ": f"{ep['PQ_DIR']}/{ep['TAG']}"})
    return out


def cmd_fusion_paths(args) -> None:
    for k, v in fusion_paths(args.seed, args.split, args.smoke).items():
        print(f"{k}={shlex.quote(str(v))}")


def cmd_fusion_check(args) -> None:
    report = json.loads(Path(args.check_json).read_text())
    waiver = None
    if args.waiver and Path(args.waiver).exists():
        waiver = json.loads(Path(args.waiver).read_text())
    v = fusion_check_verdict(report, waiver)
    v["check_json"] = str(args.check_json)
    print(f"[fusion-check RESET] C1 {v['status_fusion_select']['C1']} · C2 {v['status_fusion_select']['C2']} · "
          f"C3 {v['status_fusion_select']['C3']}{' (deroga accettata)' if v['c3_waived'] else ''} · "
          f"C5 MRR f=0.0 {v['c5_rule']}: " + " ".join(f"α={a}:{m:.4f}" for a, m in v["c5_mrr_f0"].items()))
    print(f"ESITO RESET: {'PASS' if v['pass'] else 'FAIL ' + ', '.join(v['failed'])}")
    if not v["pass"]:
        raise SystemExit(1)
    _write_once(Path(args.out_json), v)


def cmd_crossenc(args) -> None:
    """Cross-encoder control: D3 = mean(G_F - G_E), four-way rule."""
    from src.evaluation import fusion_select as fs
    true_sel = json.loads(Path(args.true_select).read_text())
    ctrl_sel = json.loads(Path(args.control_select).read_text())
    for what, sel, fdir in (("true", true_sel, args.true_dir), ("control", ctrl_sel, args.control_dir)):
        if sel.get("split") != "valid" or Path(str(sel.get("fusion_dir", ""))).resolve() != Path(fdir).resolve():
            raise SystemExit(f"!! {what} select json non descrive {fdir} sul valid")
    a_star, b_star = float(true_sel["alpha_star"]), float(ctrl_sel["alpha_star"])
    F, comp_f = fs._fusion_endpoints(args.true_dir, "valid", a_star, fs.FUSED_STRATEGY, "vision-graph")
    C, comp_c = fs._fusion_endpoints(args.control_dir, "valid", b_star,
                                     fs.pair_spec("graph-graph")["fused"], "graph-graph")
    sha = {d.meta["gallery"]["sha1"] for d in list(F.values()) + list(C.values())}
    if len(sha) != 1:
        raise SystemExit("!! gallery diverse fra le due fusioni")
    names = F[a_star].names
    st = fs.complementarity_stats(
        fs.align_by_name(names, F[a_star].names, F[a_star].auc),
        fs.align_by_name(names, F[comp_f].names, F[comp_f].auc),
        fs.align_by_name(names, C[b_star].names, C[b_star].auc),
        fs.align_by_name(names, C[comp_c].names, C[comp_c].auc), rule="four_way")
    texts = {"complementarity": "complementarità: vision + graph guadagna più di due encoder graph diversi",
             "refuted": "complementarità smentita: due encoder graph diversi guadagnano di più",
             "equivalent": "equivalenti (entro ±0.04): la diversità dei modelli basta",
             "inconclusive": "non conclusivo"}
    out = {"rule": "VERIFICA_GRAFI.md §7.6.5 (regola a quattro esiti di §51, margine ±0.04)",
           "true": {"dir": str(args.true_dir), "alpha_star": a_star, "best_component": comp_f,
                    "means": {f"{a:g}": d.mean for a, d in F.items()}},
           "control": {"pair": "graph-graph (W + E)", "dir": str(args.control_dir), "beta_star": b_star,
                       "best_component": comp_c, "means": {f"{a:g}": d.mean for a, d in C.items()}},
           "gallery_sha1": sha.pop(), "n": st["n"], "D3": st["D"], "verdict": st["verdict"],
           "verdict_text": texts[st["verdict"]], "G_F": st["G_F"], "G_E": st["G_C"],
           "AUC_F_minus_AUC_E": st["AUC_F_minus_AUC_C"], "share_G_E_over_G_F": st["share_G_C_over_G_F"]}
    for k in ("G_F", "G_E", "D3"):
        v = out[k]
        print(f"  {k}: {v['mean']:+.4f} [{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]")
    print(f"VERDETTO (§7.6.5): {out['verdict_text']}")
    _write_once(Path(args.out_json), out)


def gate(kind: str, encoder: str, cfg: str, seed: int, smoke: bool = False) -> list[str]:
    """Reasons why a test/ctrl/ctrltest evaluation is NOT allowed (empty = allowed)."""
    path = _sel_dir(smoke) / "final.json"
    if not path.exists():
        return [f"{path} mancante (vincitore non ancora scelto)"]
    w = json.loads(path.read_text())["W"]
    out = []
    extra = kind == "test" and (encoder, cfg) in TEST_EXTRA
    if (w["encoder"], w["cfg"]) != (encoder, cfg) and not extra:
        out.append(f"{encoder}/{cfg} non e' il vincitore ({w['encoder']}/{w['cfg']})")
    if extra and (w["encoder"], w["cfg"]) != (encoder, cfg) and int(seed) != TEST_EXTRA_SEED:
        out.append(f"{encoder}/{cfg} va sul test solo con il seed {TEST_EXTRA_SEED} (status.md §62)")
    r = rc.root(smoke)
    if kind == "ctrl":
        if int(seed) != rc.CTRL_REPLICA:
            out.append(f"ctrl solo per la replica {rc.CTRL_REPLICA}")
        if not (r / "fusion" / f"s{rc.CTRL_REPLICA}" / "select_valid.json").exists():
            out.append(f"prima la fusione valid della replica {rc.CTRL_REPLICA} (riscrive embeddings.npy)")
        # the head fusion reads the same sha1-pinned embeddings.npy
        if not (r / "fusion_head" / f"s{rc.CTRL_REPLICA}" / "head_fusion_valid.json").exists():
            out.append(f"prima la fusione con la head della replica {rc.CTRL_REPLICA} (riscrive embeddings.npy)")
        return out
    if kind == "ctrltest":
        # W replica on the test with damage seed 42, after the replica's test fusions are checked
        # (the eval rewrites the sha1-pinned embeddings.npy they read)
        if int(seed) != rc.CTRL_REPLICA:
            out.append(f"ctrltest solo per la replica {rc.CTRL_REPLICA}")
        for what, d in (("fusione principale", "fusion"), ("fusione con la head", "fusion_head")):
            if not (r / d / f"s{rc.CTRL_REPLICA}" / "check_test_reset.json").exists():
                out.append(f"prima il check sul test della {what} della replica {rc.CTRL_REPLICA} "
                           "(riscrive embeddings.npy)")
        if not (r / "TEST_PREREGISTERED").exists():
            out.append(f"manca {r / 'TEST_PREREGISTERED'}")
        return out
    if int(seed) not in rc.SEEDS:
        out.append(f"seed {seed} non pre-registrato")
    if not (r / "fusion" / f"s{seed}" / "select_valid.json").exists():
        out.append(f"manca la fusione valid della replica s{seed}")
    for ctrl in CONTROLS:
        if not (r / "controls" / ctrl / "complementarity_valid.json").exists():
            out.append(f"manca il controllo {ctrl} sul valid")
    if not (r / "TEST_PREREGISTERED").exists():
        out.append(f"manca {r / 'TEST_PREREGISTERED'} (si crea a mano dopo l'approvazione di status.md §59.T)")
    return out


def cmd_gate(args) -> None:
    reasons = gate(args.cmd.split("-")[0], args.encoder, args.cfg, args.seed, args.smoke)
    if reasons:
        for x in reasons:
            print(f"!! {args.cmd}: {x}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[{args.cmd}] ok: {args.encoder}/{args.cfg} s{args.seed}")


# --- CLI ---

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RESET GRAPHS: checks and choice of the graph (§7)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("status", "stage1", "show-stage2", "final"):
        s = sub.add_parser(name)
        s.add_argument("--smoke", action="store_true")
        s.add_argument("--seeds", nargs="+", type=int, default=None,
                       help="only with --smoke (default: the 4 pre-registered seeds)")
        if name == "stage1":
            s.add_argument("--exclude", nargs="*", default=[], help="<key>/<cfg> excluded (declared)")
            s.add_argument("--encoders", nargs="+", choices=list(rc.ENCODERS), default=None,
                           help="only with --smoke")
    cr = sub.add_parser("check-runs", help="checks C1-C8 on given runs (smoke test)")
    cr.add_argument("--runs", nargs="+", required=True, help="<key>/<cfg>, e.g. gcn/ref sage/ref")
    cr.add_argument("--seeds", nargs="+", type=int, default=None)
    cr.add_argument("--smoke", action="store_true")
    fp = sub.add_parser("fusion-paths")
    fp.add_argument("--seed", type=int, required=True)
    fp.add_argument("--split", choices=["valid", "test"], required=True)
    fp.add_argument("--smoke", action="store_true")
    fc = sub.add_parser("fusion-check")
    fc.add_argument("--check-json", required=True)
    fc.add_argument("--waiver", default=None)
    fc.add_argument("--out-json", required=True)
    ce = sub.add_parser("crossenc")
    for a in ("--true-select", "--true-dir", "--control-select", "--control-dir", "--out-json"):
        ce.add_argument(a, required=True)
    for name in ("test-gate", "ctrl-gate", "ctrltest-gate"):
        g = sub.add_parser(name)
        g.add_argument("--encoder", required=True)
        g.add_argument("--cfg", required=True)
        g.add_argument("--seed", type=int, required=True)
        g.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if getattr(args, "seeds", None) and not args.smoke:
        p.error("--seeds only with --smoke: the selection uses the 4 pre-registered seeds")
    if getattr(args, "encoders", None) and not args.smoke:
        p.error("--encoders only with --smoke: the selection uses all three encoders")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    {"status": cmd_status, "check-runs": cmd_check_runs, "stage1": cmd_stage1, "show-stage2": cmd_show_stage2,
     "final": cmd_final, "fusion-paths": cmd_fusion_paths, "fusion-check": cmd_fusion_check,
     "crossenc": cmd_crossenc, "test-gate": cmd_gate, "ctrl-gate": cmd_gate,
     "ctrltest-gate": cmd_gate}[args.cmd](args)


if __name__ == "__main__":
    main()
