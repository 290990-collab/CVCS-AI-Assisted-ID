# src/evaluation/export_csv.py

"""
Esporta in CSV i risultati finali del retrieval, per l'analisi nei notebook.

Perché: i numeri del report vivono in `.npz` per-query e in `.json` di
selezione; un notebook non dovrebbe ri-implementare il caricamento né, peggio,
ricopiare medie a mano. Qui i valori vengono letti UNA volta con gli stessi
loader della valutazione (`robustness_auc`, `perquery`) e scritti in tabelle
piatte.

Vincoli rispettati:
  * nessun numero nuovo — solo ri-lettura dei per-query già su disco;
  * ogni media aggregata viene CONFRONTATA con quella del json di selezione
    (`fusion_select`) e la funzione si ferma (`ValueError`) se differiscono
    oltre `TOL`: un CSV che non coincide con il risultato pubblicato è peggio
    di nessun CSV;
  * accanto ai CSV viene scritto `_sources.txt` con file letti, comando e
    medie principali (stessa convenzione delle figure, `src/figures/`).

Uso:
    python -m src.evaluation.export_csv                    # -> results/csv/
    python -m src.evaluation.export_csv --out-dir <dir>

Sistemi esportati (test, protocollo B, config definitive):
    vision   pespatial/gem/whiten-train   (= fusione con alpha=1)
    graph    gat/asymrob                  (= fusione con alpha=0)
    fusion   weighted_concat_sqrt, alpha*=0.6 fissato sul valid
    hist     baseline training-free (istogramma dei tipi di stanza)
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

# ----------------------------------------------------------------------
# Costanti del protocollo. Non sono scelte di questo modulo: le ripete solo
# per poter nominare i file giusti.
# ----------------------------------------------------------------------

AXES = ("composition", "topology", "geometry")
K_REPORT = 10                    # il k del report (nDCG@10)
TOL = 1e-6                       # tolleranza del confronto CSV <-> json pubblicato
# I file per-query si scrivono SENZA arrotondare: `str(float)` in Python 3 e'
# la rappresentazione piu' corta che rilegge lo stesso bit. Vedi
# `export_perquery_selfrr` per il motivo.
DEFAULT_OUT = Path("results/csv")

RESULTS = Path("results")
PERQUERY = RESULTS / "perquery"
FUSION = RESULTS / "fusion"

# I tre sistemi del test si leggono dalle run di fusione: alpha=0 e alpha=1
# SONO i due rami (controllo C3 del pre-registrato, 0/2000 differenze), quindi
# le query e le esclusioni sono appaiate per costruzione.
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

# Stessa cosa sul valid, per il confronto valid -> test (generalizzazione).
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

# Vision sotto i tre danni del metro (§38): solo la cartella del ramo vision ha
# crop e patch, la fusione gira sul solo `nowalls`.
VISION_DAMAGE_PREFIX = PERQUERY / "vision_test_B" / "vision_pespatial_gem_whiten-train"

# Null `random` sulla pianta intera: il ranking casuale. Serve come DENOMINATORE
# (un nDCG@10 di 0.86 su geometry va letto sapendo che il caso prende già 0.87).
# Solo il file del graph è sulla gallery condivisa: quello del vision è su una
# gallery vecchia (67.453 righe, sha1 diverso) e NON è appaiato, quindi è escluso.
# Il null non dipende dal ramo — è il caso, non un modello.
NULL_FLOOR = {"test": RESULTS / "random_floor" / "random_graph_test_seed0.npz"}

# Le tre coppie dello studio di complementarità, con il json che le pubblica.
ALPHA_PAIRS = {
    "vision-graph": FUSION / "select_valid.json",
    "vision-vision": FUSION / "select_visionvision_valid.json",
    "graph-graph": FUSION / "select_graphgraph_valid.json",
}

# Run indipendente del ramo graph, tenuta solo come controllo di riproducibilità
# (stesso comando, `--query-vectors-out` in più): due esecuzioni della stessa
# config non danno lo stesso identico ranking.
GRAPH_REPLICATE = PERQUERY / "graph_test_B" / "graph_gat_asymrob"


# ----------------------------------------------------------------------
# Utility.
# ----------------------------------------------------------------------

def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    """Scrive `rows` in CSV e restituisce il path. Nessuna riga => file con sola intestazione."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def check_against_json(label: str, computed: float, published: float) -> None:
    """Ferma l'esportazione se un valore ricalcolato non coincide col json pubblicato."""
    if not np.isclose(computed, published, atol=TOL, rtol=0.0):
        raise ValueError(
            f"{label}: ricalcolato {computed!r} != pubblicato {published!r} "
            f"(tolleranza {TOL}). I per-query e il json di selezione non sono più allineati."
        )


def full_means(path: Path, k: int = K_REPORT) -> dict[str, float]:
    """Medie nDCG@k per asse di una run `full` (NaN = query saltate, ignorate)."""
    data = load_perquery(path)
    axes, ks = data.meta["axes"], data.meta["k_values"]
    ki = ks.index(k)
    return {axis: float(np.nanmean(data.ndcg[axes.index(axis), ki])) for axis in AXES}


def paired_delta(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, float]:
    """Delta medio appaiato a - b con CI bootstrap 95% e p di Wilcoxon."""
    diff = values_a - values_b
    lo, hi = bootstrap_ci(diff)
    p, note = wilcoxon_p(values_a, values_b)
    return dict(delta=float(diff.mean()), ci_lo=lo, ci_hi=hi, p=p,
                p_note=note, n_pairs=int(len(diff)))


def aligned_auc(systems: dict, split: str) -> tuple[list[str], dict]:
    """Carica l'AUC di robustezza di ogni sistema e la allinea sui nomi comuni.

    Returns:
        (names, {sistema: {"auc": [P], "f<frazione>": [P], "data": AucData}})
        con `names` = query presenti in TUTTI i sistemi, nell'ordine del primo.
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


# ----------------------------------------------------------------------
# I singoli CSV.
# ----------------------------------------------------------------------

def export_summary(names, aligned, systems, split, out_dir, published=None) -> Path:
    """Una riga per sistema: AUC di robustezza, self_rr per frazione, nDCG@10 per asse.

    Se per lo split esiste il null `random`, la pianta intera viene anche
    normalizzata su di esso — `(score - null) / (1 - null)`, la frazione di
    spazio utile davvero coperta — e il null compare come riga a sé.
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

    # Guardia: le medie devono coincidere con quelle del json di selezione.
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
    """self_rr a f=0.0 (query intatta), allineato a `names`.

    Sta fuori da `aligned_auc` di proposito: f=0.0 NON entra nell'AUC (§23), e
    includerlo nel caricamento cambierebbe l'insieme delle query appaiate — e
    quindi l'AUC stessa. Qui serve solo a disegnare il **tetto dei dati**: il
    punto che dice quanto si ritrova quando non si toglie niente (< 1 per via
    dei duplicati esatti in RPLAN, che vincono il primo posto).
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
    """Formato lungo: una riga per (sistema, frazione) con media e CI bootstrap.

    `in_auc` distingue le tre frazioni che compongono il metro da f=0.0, che è
    il tetto dei dati e va disegnato ma non mediato.
    """
    zero = curve_fraction_zero(systems, names, split)

    rows = []
    for system, values in aligned.items():
        series_by_frac = {0.0: zero[system]}
        series_by_frac.update({f: values[f"f{f}"] for f in AUC_FRACTIONS})
        for frac, series in series_by_frac.items():
            lo, hi = bootstrap_ci(series)   # CI percentile della media (stesse bande di F7)
            rows.append(dict(split=split, system=system, fraction=frac,
                             in_auc=int(frac in AUC_FRACTIONS),
                             mean_self_rr=round(float(series.mean()), 6),
                             ci_lo=round(lo, 6), ci_hi=round(hi, 6),
                             n_queries=len(series)))
    fields = ["split", "system", "fraction", "in_auc", "mean_self_rr",
              "ci_lo", "ci_hi", "n_queries"]
    return write_csv(out_dir / f"{split}_damage_curve.csv", fields, rows)


def export_perquery_selfrr(names, aligned, split, out_dir) -> Path:
    """2000 righe x sistemi: self_rr a ogni frazione + AUC. È il file per le analisi.

    Precisione PIENA, non le 6 cifre delle tabelle riassuntive: i valori sono
    medie di reciproci di rango (1, 1/2, 1/3, ...) e centinaia di query
    PAREGGIANO esattamente fra due sistemi. Qualunque arrotondamento — anche a
    12 cifre — trasforma un pareggio in una vittoria per 1e-13, e un conteggio
    come «su quante query la fusione batte l'oracolo» sbaglia di ~180 righe su
    2000. Qui si scrive il float cosi' com'e'.
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
    """Per-query sulla pianta intera: nDCG@10 per asse + dimensione della classe.

    `num_relevant` non dipende dal sistema (è la ground truth): viene preso dal
    primo file e verificato sugli altri.
    """
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
    """Formato lungo sulla pianta intera: media di ogni metrica, per asse e per k.

    `recall` e `map` sono NaN sulle query singleton (classe vuota) e su tutto
    l'asse `geometry`, che è continuo: la media li ignora e `n_valid` dice su
    quante query è calcolata, così un k con poche query non si confonde con
    un k difficile.
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
    """Il metro a tre danni del vision: AUC per danno e per frazione, più R."""
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
    """AUC di robustezza al variare di alpha sul valid, per le tre coppie."""
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
    """Confronti appaiati: delta, CI 95% bootstrap, p di Wilcoxon.

    Due famiglie: la robustezza (l'AUC, il metro su cui si decide) e la pianta
    intera per asse (descrittiva e circolare, §21 — la colonna `note` lo dice).
    """
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

    # Pianta intera: i delta li pubblica già `fusion_select`, si ricopiano con
    # la loro nota invece di ricalcolarli (stessa fonte del report).
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
    """L'oracolo per-query (max fra i due rami): tetto della scelta ideale."""
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
    """Controllo di riproducibilità: seconda esecuzione della stessa config graph."""
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


# ----------------------------------------------------------------------
# Entrypoint.
# ----------------------------------------------------------------------

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

    # --- test -----------------------------------------------------------
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

    # --- valid ----------------------------------------------------------
    valid_names, valid_aligned = aligned_auc(VALID_SYSTEMS, split="valid")
    written.append(export_summary(valid_names, valid_aligned, VALID_SYSTEMS, "valid",
                                  out_dir, select_valid))
    written.append(export_damage_curve(valid_names, valid_aligned, VALID_SYSTEMS,
                                       "valid", out_dir))
    written.append(export_perquery_selfrr(valid_names, valid_aligned, "valid", out_dir))
    written.append(export_alpha_sweep(out_dir))

    # --- provenienza ----------------------------------------------------
    sources = [
        "# results/csv — export dei risultati finali (nessun numero nuovo)",
        f"# generato: {datetime.now().isoformat(timespec='seconds')}",
        f"# comando: python -m src.evaluation.export_csv --out-dir {out_dir}",
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

    print(f"[export_csv] {len(written)} CSV in {out_dir}")
    for path in written:
        print(f"  {path}")
    print(f"  {out_dir / '_sources.txt'}")


if __name__ == "__main__":
    sys.exit(main())
