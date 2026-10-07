"""Export the final retrieval results to CSV for notebook analysis.

Values are read once with the evaluation loaders (`robustness_auc`, `perquery`) and written as flat tables.
No new numbers: per-query files already on disk are re-read; every aggregate mean is checked against the
selection json (`fusion_select`) and `ValueError` is raised if they differ by more than `TOL`.
`_sources.txt` next to the CSVs lists files read, command and main means.

Usage:
    python -m src.evaluation.export_csv_historical                    # -> results/csv_historical/
    python -m src.evaluation.export_csv_historical --out-dir <dir>

Systems exported (test, protocol B, final configs):
    vision   pespatial/gem/whiten-train   (= fusion with alpha=1)
    graph    gat/asymrob                  (= fusion with alpha=0)
    fusion   weighted_concat_sqrt, alpha*=0.6 fixed on the valid
    hist     training-free baseline (room-type histogram)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from src.evaluation.perquery import load_perquery
from src.evaluation.robustness_auc import (
    AUC_FRACTIONS,
    ROBUST_STRATEGIES,
    load_auc,
    load_robust_auc,
)
from src.evaluation.significance import bootstrap_ci, wilcoxon_p

# --- protocol constants ---

AXES = ("composition", "topology", "geometry")
K_REPORT = 10                    # nDCG@10
TOL = 1e-6                       # CSV vs published json
# per-query files are written unrounded (see `export_perquery_selfrr`)
DEFAULT_OUT = Path("results/csv_historical")

RESULTS = Path("results")
PERQUERY = RESULTS / "perquery"
FUSION = RESULTS / "fusion"

# test systems come from the fusion runs: alpha=0 and alpha=1 are the two branches (0/2000 differences),
# so queries and exclusions are paired by construction
TEST_SYSTEMS = {
    "vision": dict(prefix=PERQUERY / "fusion_test" / "fusion_a1",
                   strategy="nowalls-random",
                   full=PERQUERY / "fusion_test" / "fusion_a1_full_test.npz"),
    "graph": dict(prefix=PERQUERY / "fusion_test" / "fusion_a0",
                  strategy="nowalls-random",
                  full=PERQUERY / "fusion_test" / "fusion_a0_full_test.npz"),
    "fusion": dict(prefix=PERQUERY / "fusion_test" / "fusion_a0.6",
                   strategy="nowalls-random",
                   full=PERQUERY / "fusion_test" / "fusion_a0.6_full_test.npz"),
    "hist_baseline": dict(prefix=PERQUERY / "graph_test_B" / "graph_hist-baseline_base",
                          strategy="random",
                          full=PERQUERY / "graph_test_B" / "graph_hist-baseline_base_full_test.npz"),
}

# same on the valid
VALID_SYSTEMS = {
    "vision": dict(prefix=PERQUERY / "fusion_valid" / "fusion_a1",
                   strategy="nowalls-random",
                   full=PERQUERY / "fusion_valid" / "fusion_a1_full_valid.npz"),
    "graph": dict(prefix=PERQUERY / "fusion_valid" / "fusion_a0",
                  strategy="nowalls-random",
                  full=PERQUERY / "fusion_valid" / "fusion_a0_full_valid.npz"),
    "fusion": dict(prefix=PERQUERY / "fusion_valid" / "fusion_a0.6",
                   strategy="nowalls-random",
                   full=PERQUERY / "fusion_valid" / "fusion_a0.6_full_valid.npz"),
}

# vision under the three damage kinds: only the vision folder has crop and patch (fusion runs `nowalls` only)
VISION_DAMAGE_PREFIX = PERQUERY / "vision_test_B" / "vision_pespatial_gem_whiten-train"

# `random` null on the whole plan (random ranking), used as denominator; only the graph file is on the
# shared gallery (the vision one is on an old 67,453-row gallery, unpaired, so excluded)
NULL_FLOOR = {"test": RESULTS / "random_floor" / "random_graph_test_seed0.npz"}

# complementarity pairs and their publishing json
ALPHA_PAIRS = {
    "vision-graph": FUSION / "select_valid.json",
    "vision-vision": FUSION / "select_visionvision_valid.json",
    "graph-graph": FUSION / "select_graphgraph_valid.json",
}

# independent graph run, kept as reproducibility check (same command plus `--query-vectors-out`)
GRAPH_REPLICATE = PERQUERY / "graph_test_B" / "graph_gat_asymrob"


# --- utilities ---

def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    """Write `rows` as CSV and return the path (no rows: header only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def check_against_json(label: str, computed: float, published: float) -> None:
    """Raise if a recomputed value differs from the published json."""
    if not np.isclose(computed, published, atol=TOL, rtol=0.0):
        raise ValueError(
            f"{label}: ricalcolato {computed!r} != pubblicato {published!r} "
            f"(tolleranza {TOL}). I per-query e il json di selezione non sono più allineati."
        )


def full_means(path: Path, k: int = K_REPORT) -> dict[str, float]:
    """Mean nDCG@k per axis of a `full` run (NaN queries ignored)."""
    data = load_perquery(path)
    axes, ks = data.meta["axes"], data.meta["k_values"]
    ki = ks.index(k)
    return {axis: float(np.nanmean(data.ndcg[axes.index(axis), ki])) for axis in AXES}


def paired_delta(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, float]:
    """Paired mean delta a - b with bootstrap 95% CI and Wilcoxon p."""
    diff = values_a - values_b
    lo, hi = bootstrap_ci(diff)
    p, note = wilcoxon_p(values_a, values_b)
    return dict(delta=float(diff.mean()), ci_lo=lo, ci_hi=hi, p=p,
                p_note=note, n_pairs=int(len(diff)))


def aligned_auc(systems: dict, split: str) -> tuple[list[str], dict]:
    """Robustness AUC of each system aligned on common names.

    Returns (names, {system: {"auc": [P], "f<fraction>": [P], "data": AucData}}), names = queries in all systems, first system's order.
    """
    loaded = {name: load_auc(cfg["prefix"], split=split, strategy=cfg["strategy"])
              for name, cfg in systems.items()}
    common = set.intersection(*(set(map(str, d.names)) for d in loaded.values()))
    first = next(iter(loaded.values()))
    names = [str(n) for n in first.names if n in common]

    aligned = {}
    for system, data in loaded.items():
        index = {str(n): i for i, n in enumerate(data.names)}
        take = np.asarray([index[n] for n in names])
        aligned[system] = {"auc": data.auc[take], "data": data}
        for frac in AUC_FRACTIONS:
            aligned[system][f"f{frac}"] = data.per_fraction[frac][take]
    return names, aligned


# --- individual CSVs ---

def export_summary(names, aligned, systems, split, out_dir, published=None) -> Path:
    """One row per system: robustness AUC, self_rr per fraction, nDCG@10 per axis.

    With a `random` null for the split, whole-plan scores are also normalised as (score - null) / (1 - null)
    and the null gets its own row.
    """
    floor_path = NULL_FLOOR.get(split)
    floor = full_means(floor_path) if floor_path else None

    rows = []
    for system in systems:
        auc = aligned[system]["auc"]
        row = dict(split=split, system=system, n_queries=len(names),
                   robustness_auc=round(float(auc.mean()), 6))
        for frac in AUC_FRACTIONS:
            row[f"self_rr_f{frac}"] = round(float(aligned[system][f"f{frac}"].mean()), 6)
        for axis, value in full_means(systems[system]["full"]).items():
            row[f"ndcg10_{axis}"] = round(value, 6)
            if floor:
                row[f"ndcg10_{axis}_vs_null"] = round((value - floor[axis]) / (1 - floor[axis]), 6)
        rows.append(row)

    if floor:
        rows.append(dict(split=split, system="null_random", n_queries=len(names),
                         robustness_auc="",
                         **{f"self_rr_f{f}": "" for f in AUC_FRACTIONS},
                         **{f"ndcg10_{a}": round(floor[a], 6) for a in AXES},
                         **{f"ndcg10_{a}_vs_null": 0.0 for a in AXES}))

    # means must match the selection json
    if published:
        for system, alpha in (("graph", "0"), ("fusion", "0.6"), ("vision", "1")):
            if system in aligned and alpha in published["auc_means"]:
                check_against_json(f"{split}/{system} robustness_auc",
                                   float(aligned[system]["auc"].mean()),
                                   published["auc_means"][alpha])

    fields = (["split", "system", "n_queries", "robustness_auc"]
              + [f"self_rr_f{f}" for f in AUC_FRACTIONS]
              + [f"ndcg10_{a}" for a in AXES]
              + ([f"ndcg10_{a}_vs_null" for a in AXES] if floor else []))
    return write_csv(out_dir / f"{split}_summary.csv", fields, rows)


def curve_fraction_zero(systems, names, split) -> dict:
    """self_rr at f=0.0 (intact query) aligned on `names`: the data ceiling (< 1 due to exact RPLAN duplicates).

    Kept out of `aligned_auc`: f=0.0 is not in the AUC and would change the paired query set.
    """
    zero = {}
    for system, cfg in systems.items():
        data = load_auc(cfg["prefix"], split=split, fractions=(0.0,),
                        strategy=cfg["strategy"])
        index = {str(n): i for i, n in enumerate(data.names)}
        zero[system] = np.asarray([data.per_fraction[0.0][index[n]] for n in names
                                   if n in index])
    return zero


def export_damage_curve(names, aligned, systems, split, out_dir) -> Path:
    """Long format: one row per (system, fraction) with mean and bootstrap CI; `in_auc` is 0 for f=0.0 (plotted, not averaged)."""
    zero = curve_fraction_zero(systems, names, split)

    rows = []
    for system, values in aligned.items():
        series_by_frac = {0.0: zero[system]}
        series_by_frac.update({f: values[f"f{f}"] for f in AUC_FRACTIONS})
        for frac, series in series_by_frac.items():
            lo, hi = bootstrap_ci(series)
            rows.append(dict(split=split, system=system, fraction=frac,
                             in_auc=int(frac in AUC_FRACTIONS),
                             mean_self_rr=round(float(series.mean()), 6),
                             ci_lo=round(lo, 6), ci_hi=round(hi, 6),
                             n_queries=len(series)))
    fields = ["split", "system", "fraction", "in_auc", "mean_self_rr",
              "ci_lo", "ci_hi", "n_queries"]
    return write_csv(out_dir / f"{split}_damage_curve.csv", fields, rows)


def export_perquery_selfrr(names, aligned, split, out_dir) -> Path:
    """2000 rows x systems: self_rr per fraction + AUC, the file for analyses.

    Written at full precision: values are means of reciprocal ranks and hundreds of queries tie exactly;
    any rounding turns a tie into a win (counts like "fusion beats oracle" shift by ~180 of 2000).
    """
    rows = []
    for i, name in enumerate(names):
        row = {"query": name}
        for system, values in aligned.items():
            for frac in AUC_FRACTIONS:
                row[f"{system}_f{frac}"] = float(values[f"f{frac}"][i])
            row[f"{system}_auc"] = float(values["auc"][i])
        rows.append(row)
    fields = ["query"] + [f"{s}_{c}" for s in aligned
                          for c in [f"f{f}" for f in AUC_FRACTIONS] + ["auc"]]
    return write_csv(out_dir / f"{split}_perquery_selfrr.csv", fields, rows)


def export_perquery_full(systems, split, out_dir) -> Path:
    """Whole-plan per-query nDCG@10 per axis + class size; `num_relevant` is ground truth, checked equal across systems."""
    full_files = {s: cfg["full"] for s, cfg in systems.items()}
    if split in NULL_FLOOR:
        full_files["null_random"] = NULL_FLOOR[split]

    per_system, reference = {}, None
    for system, path in full_files.items():
        data = load_perquery(path)
        axes, ki = data.meta["axes"], data.meta["k_values"].index(K_REPORT)
        per_system[system] = dict(
            names=[str(n) for n in data.names],
            ndcg={a: data.ndcg[axes.index(a), ki] for a in AXES},
            num_relevant={a: data.num_relevant[axes.index(a)] for a in AXES},
        )
        if reference is None:
            reference = system
        elif per_system[system]["names"] != per_system[reference]["names"]:
            raise ValueError(f"{split}/full: query non appaiate fra {reference} e {system}")

    names = per_system[reference]["names"]
    for system in per_system:
        for axis in AXES:
            if not np.array_equal(per_system[system]["num_relevant"][axis],
                                  per_system[reference]["num_relevant"][axis]):
                raise ValueError(
                    f"{split}/full: `num_relevant` su {axis} differisce fra "
                    f"{reference} e {system}: non è più ground truth condivisa")

    rows = []
    for i, name in enumerate(names):
        row = {"query": name}
        for axis in AXES:
            row[f"num_relevant_{axis}"] = int(per_system[reference]["num_relevant"][axis][i])
        for system in per_system:
            for axis in AXES:
                value = float(per_system[system]["ndcg"][axis][i])
                row[f"{system}_ndcg10_{axis}"] = "" if np.isnan(value) else value
        rows.append(row)

    fields = (["query"] + [f"num_relevant_{a}" for a in AXES]
              + [f"{s}_ndcg10_{a}" for s in per_system for a in AXES])
    return write_csv(out_dir / f"{split}_perquery_full_ndcg10.csv", fields, rows)


def export_full_metrics_all_k(systems, split, out_dir) -> Path:
    """Long format, whole plan: mean of each metric per axis and k.

    `recall` and `map` are NaN on singleton queries and on the continuous `geometry` axis; means skip them
    and `n_valid` counts the queries used.
    """
    full_files = {s: cfg["full"] for s, cfg in systems.items()}
    if split in NULL_FLOOR:
        full_files["null_random"] = NULL_FLOOR[split]

    rows = []
    for system, path in full_files.items():
        data = load_perquery(path)
        axes, ks = data.meta["axes"], data.meta["k_values"]
        for metric_name, values in (("ndcg", data.ndcg), ("recall", data.recall),
                                    ("map", data.map)):
            for axis in AXES:
                for ki, k in enumerate(ks):
                    series = values[axes.index(axis), ki]
                    valid = int((~np.isnan(series)).sum())
                    rows.append(dict(
                        split=split, system=system, axis=axis, metric=metric_name, k=k,
                        mean=("" if valid == 0 else round(float(np.nanmean(series)), 6)),
                        n_valid=valid, n_queries=len(series)))
    fields = ["split", "system", "axis", "metric", "k", "mean", "n_valid", "n_queries"]
    return write_csv(out_dir / f"{split}_full_metrics_all_k.csv", fields, rows)


def export_vision_damage_kinds(split, out_dir) -> tuple[Path, float]:
    """Vision three-damage metric: AUC per damage kind and fraction, plus R."""
    robust = load_robust_auc(VISION_DAMAGE_PREFIX, split=split)
    rows = []
    for strategy in ROBUST_STRATEGIES:
        component = robust.components[strategy]
        row = dict(split=split, system="vision", strategy=strategy,
                   auc=round(component.mean, 6), n_queries=len(component.auc))
        for frac in AUC_FRACTIONS:
            row[f"self_rr_f{frac}"] = round(float(component.per_fraction[frac].mean()), 6)
        rows.append(row)
    rows.append(dict(split=split, system="vision", strategy="mean(R)",
                     auc=round(robust.mean, 6), n_queries=len(robust.auc),
                     **{f"self_rr_f{f}": "" for f in AUC_FRACTIONS}))
    fields = (["split", "system", "strategy", "auc"]
              + [f"self_rr_f{f}" for f in AUC_FRACTIONS] + ["n_queries"])
    return write_csv(out_dir / f"{split}_vision_damage_kinds.csv", fields, rows), robust.mean


def export_alpha_sweep(out_dir) -> Path:
    """Robustness AUC vs alpha on the valid, for the three pairs."""
    rows = []
    for pair, path in ALPHA_PAIRS.items():
        published = json.loads(path.read_text())
        for alpha, auc in sorted(published["auc_means"].items(), key=lambda kv: float(kv[0])):
            rows.append(dict(split="valid", pair=pair, alpha=float(alpha),
                             robustness_auc=round(float(auc), 6),
                             is_alpha_star=int(float(alpha) == published["alpha_star"]),
                             reference_alpha=published["reference_alpha"],
                             source=str(path)))
    fields = ["split", "pair", "alpha", "robustness_auc", "is_alpha_star",
              "reference_alpha", "source"]
    return write_csv(out_dir / "valid_alpha_sweep.csv", fields, rows)


def export_paired_deltas(names, aligned, split, out_dir, published) -> Path:
    """Paired comparisons: delta, bootstrap 95% CI, Wilcoxon p; families: robustness AUC and whole plan per axis (descriptive, circular)."""
    rows = []
    comparisons = [("fusion", "graph"), ("fusion", "vision"), ("graph", "vision")]
    for a, b in comparisons:
        if a not in aligned or b not in aligned:
            continue
        stats = paired_delta(aligned[a]["auc"], aligned[b]["auc"])
        rows.append(dict(split=split, family="robustness_auc", comparison=f"{a}-{b}",
                         axis="", metric="auc_self_recovery",
                         mean_a=round(float(aligned[a]["auc"].mean()), 6),
                         mean_b=round(float(aligned[b]["auc"].mean()), 6),
                         **{k: (round(v, 6) if isinstance(v, float) else v)
                            for k, v in stats.items()},
                         note="metro di selezione"))

    # whole plan: deltas copied from `fusion_select`, not recomputed
    for key, label in (("star_minus_graph", "fusion-graph"),
                       ("star_minus_vision", "fusion-vision")):
        for axis in AXES:
            entry = published["full"][key][axis]
            rows.append(dict(split=split, family="full_plan", comparison=label,
                             axis=axis, metric=f"ndcg@{entry['k']}",
                             mean_a=round(entry["mean_a"], 6),
                             mean_b=round(entry["mean_b"], 6),
                             delta=round(entry["delta"], 6),
                             ci_lo=round(entry["ci_lo"], 6),
                             ci_hi=round(entry["ci_hi"], 6),
                             p=entry["p"], p_note="", n_pairs=entry["n_pairs"],
                             note="descrittivo e circolare (label dall'input del graph)"))

    fields = ["split", "family", "comparison", "axis", "metric", "mean_a", "mean_b",
              "delta", "ci_lo", "ci_hi", "p", "p_note", "n_pairs", "note"]
    return write_csv(out_dir / f"{split}_paired_deltas.csv", fields, rows)


def export_oracle(split, out_dir, published) -> Path:
    """Per-query oracle (max of the two branches): ceiling of the ideal choice."""
    rows = [dict(split=split, scope="robustness_auc", axis="",
                 star_mean=round(published["oracle"]["star_mean"], 6),
                 oracle_mean=round(published["oracle"]["oracle_mean"], 6),
                 delta_star_minus_oracle=round(published["oracle"]["delta_star_minus_oracle"], 6),
                 n_star_exceeds_oracle=published["oracle"]["n_star_exceeds_oracle"],
                 n=published["oracle"]["n"])]
    for axis in AXES:
        entry = published["full"]["oracle"][axis]
        rows.append(dict(split=split, scope="full_plan", axis=axis,
                         star_mean=round(entry["star_mean"], 6),
                         oracle_mean=round(entry["oracle_mean"], 6),
                         delta_star_minus_oracle=round(entry["delta_star_minus_oracle"], 6),
                         n_star_exceeds_oracle=entry["n_star_exceeds_oracle"],
                         n=entry["n"]))
    fields = ["split", "scope", "axis", "star_mean", "oracle_mean",
              "delta_star_minus_oracle", "n_star_exceeds_oracle", "n"]
    return write_csv(out_dir / f"{split}_oracle.csv", fields, rows)


def export_graph_replicate(names, aligned, out_dir) -> tuple[Path, float]:
    """Reproducibility check: second run of the same graph config."""
    replicate = load_auc(GRAPH_REPLICATE, split="test", strategy="random")
    index = {str(n): i for i, n in enumerate(replicate.names)}
    take = np.asarray([index[n] for n in names])
    diff = replicate.auc[take] - aligned["graph"]["auc"]
    rows = [dict(split="test", system="graph_gat_asymrob",
                 run_a="results/perquery/fusion_test (alpha=0)",
                 run_b=str(GRAPH_REPLICATE),
                 mean_a=round(float(aligned["graph"]["auc"].mean()), 6),
                 mean_b=round(float(replicate.auc[take].mean()), 6),
                 mean_diff=round(float(diff.mean()), 6),
                 n_queries_differing=int((diff != 0).sum()),
                 max_abs_diff=round(float(np.abs(diff).max()), 6),
                 n=len(diff))]
    fields = ["split", "system", "run_a", "run_b", "mean_a", "mean_b", "mean_diff",
              "n_queries_differing", "max_abs_diff", "n"]
    return write_csv(out_dir / "test_graph_reproducibility.csv", fields, rows), float(diff.mean())


# --- entrypoint ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                        help=f"cartella di destinazione (default: {DEFAULT_OUT})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    select_test = json.loads((FUSION / "select_test.json").read_text())
    select_valid = json.loads((FUSION / "select_valid.json").read_text())

    written, notes = [], []

    # --- test ---
    names, aligned = aligned_auc(TEST_SYSTEMS, split="test")
    notes.append(f"test: {len(names)} query appaiate su {len(TEST_SYSTEMS)} sistemi")
    written.append(export_summary(names, aligned, TEST_SYSTEMS, "test", out_dir, select_test))
    written.append(export_damage_curve(names, aligned, TEST_SYSTEMS, "test", out_dir))
    written.append(export_perquery_selfrr(names, aligned, "test", out_dir))
    written.append(export_perquery_full(TEST_SYSTEMS, "test", out_dir))
    written.append(export_full_metrics_all_k(TEST_SYSTEMS, "test", out_dir))
    path, vision_r = export_vision_damage_kinds("test", out_dir)
    written.append(path)
    notes.append(f"vision R (3 danni, test) = {vision_r:.6f}")
    written.append(export_paired_deltas(names, aligned, "test", out_dir, select_test))
    written.append(export_oracle("test", out_dir, select_test))
    path, replicate_diff = export_graph_replicate(names, aligned, out_dir)
    written.append(path)
    notes.append(f"replica graph: differenza media AUC {replicate_diff:+.6f}")

    # --- valid ---
    valid_names, valid_aligned = aligned_auc(VALID_SYSTEMS, split="valid")
    written.append(export_summary(valid_names, valid_aligned, VALID_SYSTEMS, "valid",
                                  out_dir, select_valid))
    written.append(export_damage_curve(valid_names, valid_aligned, VALID_SYSTEMS,
                                       "valid", out_dir))
    written.append(export_perquery_selfrr(valid_names, valid_aligned, "valid", out_dir))
    written.append(export_alpha_sweep(out_dir))

    # --- provenance ---
    sources = [
        "# results/csv_historical — export dei risultati finali (nessun numero nuovo)",
        f"# generato: {datetime.now().isoformat(timespec='seconds')}",
        f"# comando: python -m src.evaluation.export_csv_historical --out-dir {out_dir}",
        "",
        "## come leggere i valori",
        "- `self_rr`: reciprocal rank della pianta ORIGINALE quando la query è la sua",
        "  versione danneggiata (self-recovery). 1.0 = ritrovata al primo posto, 0 = fuori",
        "  dai primi 100. `robustness_auc` = media di self_rr sulle frazioni 0.25/0.5/0.75",
        "  (f=0.0 escluso: è il tetto dei dati, non misura robustezza).",
        "- `fraction` = quota di stanze tolte alla query (vision: svuotate senza muri).",
        "- `num_relevant` = quante piante della gallery sono nella stessa classe di",
        "  equivalenza esatta della query (self escluso). 0 = query singleton;",
        "  -1 = asse continuo (geometry), nessuna classe.",
        "- nDCG usa rilevanza GRADUATA, quindi è definito anche sulle singleton;",
        "  `recall` e `map` no: sono vuoti lì e su tutto l'asse geometry.",
        "- PRECISIONE: le tabelle riassuntive sono arrotondate a 6 cifre (si leggono),",
        "  i file per-query NO. Centinaia di query pareggiano esattamente fra due",
        "  sistemi: su valori arrotondati un `>` conta quei pareggi come vittorie",
        "  (~180 righe su 2000 nel confronto con l'oracolo). Conteggi e confronti",
        "  si fanno sui file per-query, non sulle tabelle.",
        "- I delta `full_plan` sono DESCRITTIVI: le label degli assi derivano da",
        "  rType/rEdge, cioè dall'input del ramo graph (circolarità dichiarata).",
        "",
        "## verifiche superate",
        f"- ogni media di robustezza coincide con select_*.json entro {TOL}",
        "- `num_relevant` identico fra i sistemi su ogni asse (ground truth condivisa)",
        "- query appaiate per nome fra tutti i sistemi",
        "",
        "## note",
        *(f"- {n}" for n in notes),
        "",
        "## letti",
        *(f"- {path}" for path in dict.fromkeys(
            [FUSION / "select_test.json", FUSION / "select_valid.json", *ALPHA_PAIRS.values()])),
        *(f"- {cfg['prefix']}_partial-{cfg['strategy']}-f*_test.npz, {cfg['full']}"
          for cfg in TEST_SYSTEMS.values()),
        *(f"- {cfg['prefix']}_partial-{cfg['strategy']}-f*_valid.npz, {cfg['full']}"
          for cfg in VALID_SYSTEMS.values()),
        f"- {VISION_DAMAGE_PREFIX}_partial-{{{','.join(ROBUST_STRATEGIES)}}}-f*_test.npz",
        f"- {GRAPH_REPLICATE}_partial-random-f*_test.npz",
        *(f"- {path}  (null `random`, pianta intera)" for path in NULL_FLOOR.values()),
        "",
        "## scritti",
        *(f"- {path}" for path in written),
    ]
    (out_dir / "_sources.txt").write_text("\n".join(sources) + "\n")

    print(f"[export_csv_historical] {len(written)} CSV in {out_dir}")
    for path in written:
        print(f"  {path}")
    print(f"  {out_dir / '_sources.txt'}")


if __name__ == "__main__":
    sys.exit(main())
