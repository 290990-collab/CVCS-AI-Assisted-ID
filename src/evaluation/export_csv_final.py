"""Export the final-pipeline results (valid and test) to CSV for notebooks, tables and figures.

Like `export_csv_historical` (which covers the historical `gat/asymrob` results), for the final pipeline:

    graph   W = gat/comb (selected among 37 configs x 4 seeds)
    vision  frozen pespatial/gem/whiten; vision with head (head + train whitening)
    fusion  main: W + frozen vision (alpha 0.4); with head (alpha 0.3 seed 42, 0.4 others)
    controls  W + second copy of W, W + sage/comb
    curve   fusion gain vs graph strength
    LayoutGKN re-trained like W

Constraints (same as `export_csv_historical`):
  * no new numbers: per-query files and json already written are re-read;
  * every mean recomputed from per-query is checked against the publishing json (tolerance `TOL`);
    the export stops on mismatch;
  * writes only to a new folder (default `results/csv_final/`), refuses to start if it exists;
  * `_sources.txt` next to the CSVs lists files read, command and checks.

Usage (CPU, ~1 minute):
    python -m src.evaluation.export_csv_final
    python -m src.evaluation.export_csv_final --out-dir <new folder>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from src.evaluation.export_csv_historical import (
    AXES,
    NULL_FLOOR,
    TOL,
    aligned_auc,
    check_against_json,
    export_damage_curve,
    export_full_metrics_all_k,
    export_perquery_full,
    export_perquery_selfrr,
    full_means,
    paired_delta,
    write_csv,
)
from src.evaluation.robustness_auc import AUC_FRACTIONS

DEFAULT_OUT = Path("results/csv_final")
R = Path("results/final_pipeline")
LGKN_PQ = Path("results/perquery/competitor_layoutgkn")
SEEDS = (42, 100042, 200042, 300042)
DECIDING_SEED = 42
# weights fixed on the valid
ALPHA_MAIN = 0.4
ALPHA_HEAD = {42: 0.3, 100042: 0.4, 200042: 0.4, 300042: 0.4}


def _a(alpha: float) -> str:
    return f"{alpha:g}"


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def systems(split: str) -> dict:
    """Seed-42 systems on a split: fusion runs (alpha=0 / alpha=1 are the branches), LayoutGKN, `hist` baseline on test."""
    fm = R / "fusion" / f"s{DECIDING_SEED}" / f"fusion_{split}"
    fh = R / "fusion_head" / f"s{DECIDING_SEED}" / f"fusion_{split}"
    ah = ALPHA_HEAD[DECIDING_SEED]

    def run(d: Path, alpha: float, strategy: str = "nowalls-random") -> dict:
        return dict(prefix=d / f"fusion_a{_a(alpha)}", strategy=strategy,
                    full=d / f"fusion_a{_a(alpha)}_full_{split}.npz")

    lg = LGKN_PQ / ("asym" if split == "valid" else "asym_test") / "graph_lgkn_asym_s42"
    out = {
        "graph_W": run(fm, 0.0),
        "vision_frozen": run(fm, 1.0),
        "vision_head": run(fh, 1.0),
        "fusion_main": run(fm, ALPHA_MAIN),
        "fusion_head": run(fh, ah),
        "layoutgkn": dict(prefix=lg, strategy="random", full=Path(f"{lg}_full_{split}.npz")),
    }
    if split == "test":
        b = Path("results/perquery/graph_test_B/graph_hist-baseline_base")
        out["hist_baseline"] = dict(prefix=b, strategy="random", full=Path(f"{b}_full_test.npz"))
    return out


def check_published(split: str, aligned: dict, sources: list) -> list[str]:
    """Each recomputed mean AUC must equal the selection / LayoutGKN json value within TOL."""
    s = DECIDING_SEED
    main = _json(R / "fusion" / f"s{s}" / f"select_{split}.json")
    head = _json(R / "fusion_head" / f"s{s}" / f"select_{split}.json")
    lg_path = Path("results/competitors/layoutgkn") / ("full/full.json" if split == "valid" else "test/test.json")
    lg = _json(lg_path)
    sources += [R / "fusion" / f"s{s}" / f"select_{split}.json",
                R / "fusion_head" / f"s{s}" / f"select_{split}.json", lg_path]
    pairs = [("graph_W", main["auc_means"]["0"]), ("vision_frozen", main["auc_means"]["1"]),
             ("fusion_main", main["auc_means"][_a(ALPHA_MAIN)]), ("vision_head", head["auc_means"]["1"]),
             ("fusion_head", head["auc_means"][_a(ALPHA_HEAD[s])]),
             ("layoutgkn", lg["auc"]["lgkn_asym"])]
    done = []
    for system, published in pairs:
        check_against_json(f"{split}/{system}", float(aligned[system]["auc"].mean()), published)
        done.append(f"{split}/{system}: AUC = json entro {TOL}")
    check_against_json(f"{split}/W in LayoutGKN json", float(aligned["graph_W"]["auc"].mean()),
                       lg["auc"]["w_gat_comb_s42"])
    return done


def export_summary_reset(names, aligned, syst, split, out_dir) -> Path:
    """One row per system: AUC, self_rr per fraction, whole-plan nDCG@10 (also normalised on the `random` null if present)."""
    floor = full_means(NULL_FLOOR[split]) if split in NULL_FLOOR else None
    rows = []
    for system, cfg in syst.items():
        row = dict(split=split, seed=DECIDING_SEED, system=system, n_queries=len(names),
                   robustness_auc=round(float(aligned[system]["auc"].mean()), 6))
        for f in AUC_FRACTIONS:
            row[f"self_rr_f{f}"] = round(float(aligned[system][f"f{f}"].mean()), 6)
        for axis, v in full_means(cfg["full"]).items():
            row[f"ndcg10_{axis}"] = round(v, 6)
            if floor:
                row[f"ndcg10_{axis}_vs_null"] = round((v - floor[axis]) / (1 - floor[axis]), 6)
        rows.append(row)
    fields = (["split", "seed", "system", "n_queries", "robustness_auc"]
              + [f"self_rr_f{f}" for f in AUC_FRACTIONS] + [f"ndcg10_{a}" for a in AXES]
              + ([f"ndcg10_{a}_vs_null" for a in AXES] if floor else []))
    return write_csv(out_dir / f"{split}_summary.csv", fields, rows)


def export_paired(names, aligned, split, out_dir) -> Path:
    """Seed-42 paired comparisons (per-query AUC), bootstrap CI and Wilcoxon p."""
    comps = [("fusion_main", "graph_W", "la fusione principale aiuta?"),
             ("fusion_head", "graph_W", "la fusione con head aiuta?"),
             ("fusion_head", "fusion_main", "Δ_H: il guadagno cresce con la head?"),
             ("vision_head", "vision_frozen", "head contro vision congelato"),
             ("layoutgkn", "graph_W", "LayoutGKN contro W")]
    if "hist_baseline" in aligned:
        comps.append(("graph_W", "hist_baseline", "W contro baseline hist"))
    rows = []
    for a, b, q in comps:
        d = paired_delta(aligned[a]["auc"], aligned[b]["auc"])
        rows.append(dict(split=split, seed=DECIDING_SEED, a=a, b=b, question=q,
                         mean_a=round(float(aligned[a]["auc"].mean()), 6),
                         mean_b=round(float(aligned[b]["auc"].mean()), 6),
                         delta=round(d["delta"], 6), ci_lo=round(d["ci_lo"], 6), ci_hi=round(d["ci_hi"], 6),
                         p_wilcoxon=d["p"], n_pairs=d["n_pairs"]))
    fields = ["split", "seed", "a", "b", "question", "mean_a", "mean_b", "delta", "ci_lo", "ci_hi",
              "p_wilcoxon", "n_pairs"]
    return write_csv(out_dir / f"{split}_paired_deltas.csv", fields, rows)


def export_seeds(split, out_dir, sources) -> Path:
    """One row per replica: branches, fusions, gains with CI (from selection/compare json)."""
    rows = []
    for s in SEEDS:
        main_p = R / "fusion" / f"s{s}" / f"select_{split}.json"
        head_p = R / "fusion_head" / f"s{s}" / f"select_{split}.json"
        cmp_p = R / "fusion_head" / f"s{s}" / f"compare_{split}.json"
        main, head, cmp_ = _json(main_p), _json(head_p), _json(cmp_p)
        sources += [main_p, head_p, cmp_p]
        am, ah = main["alpha_star"], head["alpha_star"]
        if split == "test" and (am != ALPHA_MAIN or ah != ALPHA_HEAD[s]):
            raise ValueError(f"s{s}: pesi del test {am}/{ah} diversi da quelli pre-registrati")
        g, dh = cmp_["gain_head"], cmp_["measure2_delta_H"]
        sv = main["star_vs_reference"]
        row = dict(split=split, seed=s, alpha_main=am, alpha_head=ah,
                   auc_graph_W=main["auc_means"]["0"], auc_vision_frozen=main["auc_means"]["1"],
                   auc_vision_head=head["auc_means"]["1"],
                   auc_fusion_main=main["auc_means"][_a(am)], auc_fusion_head=head["auc_means"][_a(ah)],
                   gain_main=sv["delta"], gain_main_ci_lo=sv["ci_lo"], gain_main_ci_hi=sv["ci_hi"],
                   gain_head=g["mean"], gain_head_ci_lo=g["ci_lo"], gain_head_ci_hi=g["ci_hi"],
                   delta_H=dh["mean"], delta_H_ci_lo=dh["ci_lo"], delta_H_ci_hi=dh["ci_hi"],
                   oracle_main=main["oracle"]["oracle_mean"], outcome_head=cmp_["outcome"])
        if split == "test":
            v = _json(R / "fusion" / f"s{s}" / "select_valid.json")
            fv = v["auc_means"][_a(v["alpha_star"])]
            row.update(auc_fusion_main_valid=fv, abs_test_minus_valid=abs(row["auc_fusion_main"] - fv),
                       P6_within_002=int(abs(row["auc_fusion_main"] - fv) <= 0.02))
        rows.append({k: (round(x, 6) if isinstance(x, float) else x) for k, x in row.items()})
    fields = list(rows[0].keys())
    return write_csv(out_dir / f"{split}_seeds.csv", fields, rows)


def export_alpha_sweep(out_dir, sources) -> Path:
    """Valid: AUC vs weight for main and head fusions (4 replicas) and the two graph + graph controls (seed 42)."""
    entries = [(f"fusion_main/s{s}", R / "fusion" / f"s{s}" / "select_valid.json") for s in SEEDS]
    entries += [(f"fusion_head/s{s}", R / "fusion_head" / f"s{s}" / "select_valid.json") for s in SEEDS]
    entries += [("control_replica/s42", R / "controls" / "replica" / "select_valid.json"),
                ("control_crossenc/s42", R / "controls" / "crossenc" / "select_valid.json")]
    rows = []
    for name, path in entries:
        sel = _json(path)
        sources.append(path)
        for alpha, auc in sorted(sel["auc_means"].items(), key=lambda kv: float(kv[0])):
            rows.append(dict(split="valid", fusion=name, pair=sel["pair"], alpha=float(alpha),
                             robustness_auc=round(float(auc), 6),
                             is_alpha_star=int(float(alpha) == sel["alpha_star"]),
                             reference_alpha=sel["reference_alpha"]))
    fields = ["split", "fusion", "pair", "alpha", "robustness_auc", "is_alpha_star", "reference_alpha"]
    return write_csv(out_dir / "valid_alpha_sweep.csv", fields, rows)


def _ci(d: dict, key: str = "mean") -> tuple:
    return round(d[key], 6), round(d["ci_lo"], 6), round(d["ci_hi"], 6)


def export_controls(out_dir, sources) -> list[Path]:
    """Gain g = fusion - best branch of the pair, for the true fusion and the controls, and differences D (valid and test)."""
    fields = ["split", "row", "what", "value", "ci_lo", "ci_hi", "weight", "source"]
    rows = []
    # replica: G_C, D; crossenc: G_E, D3 (same quantity g(F) - g(control))
    for ctrl, gk, dk in (("replica", "G_C", "D"), ("crossenc", "G_E", "D3")):
        p = R / "controls" / ctrl / "complementarity_valid.json"
        c = _json(p)
        sources.append(p)
        rows.append(dict(split="valid", row=f"g_fusion_main (vs {ctrl})", what="guadagno fusione principale",
                         **dict(zip(("value", "ci_lo", "ci_hi"), _ci(c["G_F"]))),
                         weight=c["true"]["alpha_star"], source=str(p)))
        rows.append(dict(split="valid", row=f"g_control_{ctrl}", what=f"guadagno controllo {ctrl}",
                         **dict(zip(("value", "ci_lo", "ci_hi"), _ci(c[gk]))),
                         weight=c["control"].get("beta_star"), source=str(p)))
        rows.append(dict(split="valid", row=f"D_{ctrl}_frozen", what="g(fusione principale) − g(controllo)",
                         **dict(zip(("value", "ci_lo", "ci_hi"), _ci(c[dk]))), weight="", source=str(p)))
    p = R / "controls_test" / "confirm_C_test.json"
    c = _json(p)
    sources.append(p)
    for k, w in (("F_H", c["fusions"]["F_H"]["weight"]), ("C_rep", c["fusions"]["C_rep"]["weight"]),
                 ("C_enc", c["fusions"]["C_enc"]["weight"])):
        rows.append(dict(split="test", row=f"g_{k}", what=f"guadagno {k}",
                         **dict(zip(("value", "ci_lo", "ci_hi"), _ci(c["g"][k]))), weight=w, source=str(p)))
    for k in ("D_rep", "D_enc"):
        rows.append(dict(split="test", row=k, what="g(F_H) − g(controllo), confermativo",
                         **dict(zip(("value", "ci_lo", "ci_hi"), _ci(c[k]))), weight="", source=str(p)))
    df = c["descriptive_frozen"]
    rows.append(dict(split="test", row="g_F_frozen", what="guadagno fusione principale (descrittivo)",
                     **dict(zip(("value", "ci_lo", "ci_hi"), _ci(df["g_F_frozen"]))),
                     weight=c["fusions"]["F_frozen"]["weight"], source=str(p)))
    for k in ("D_rep", "D_enc"):
        rows.append(dict(split="test", row=f"{k}_frozen", what="g(fusione principale) − g(controllo), descrittivo",
                         **dict(zip(("value", "ci_lo", "ci_hi"), _ci(df[k]))), weight="", source=str(p)))
    out = [write_csv(out_dir / "controls_gain.csv", fields, rows)]
    out.append(write_csv(out_dir / "test_controls_outcome.csv", ["outcome", "source"],
                         [dict(outcome=c["outcome"], source=str(p))]))
    return out


def export_selection(out_dir, sources) -> Path:
    """Valid: the 37 graph configs (3 encoders), mean over 4 seeds, and distance from W."""
    p = R / "selection" / "final.json"
    fin = _json(p)
    sources.append(p)
    rows = []
    for rank, key in enumerate(fin["ranking"], 1):
        s = fin["summaries"][key]
        vs = fin["vs_top"].get(key)
        enc, cfg = key.split("/")
        row = dict(rank=rank, encoder=enc, cfg=cfg, auc_mean_4seeds=s["score"], auc_sd=s["score_sd"],
                   mrr_f0=s["mrr_f0"], n_params=s["n_params"], budget_binding_seeds=s["budget_binding"],
                   is_W=int(key == fin["top_by_score"]),
                   W_minus_this=(vs["delta"] if vs else ""), ci_lo=(vs["ci_lo"] if vs else ""),
                   ci_hi=(vs["ci_hi"] if vs else ""))
        row.update({f"auc_s{k}": v for k, v in sorted(s["per_seed"].items(), key=lambda kv: int(kv[0]))})
        row.update({f"self_rr_f{k}": v for k, v in s["per_fraction"].items()})
        row.update({f"ndcg10_{a}": s["full_ndcg10"][a] for a in AXES})
        rows.append({k: (round(x, 6) if isinstance(x, float) else x) for k, x in row.items()})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    return write_csv(out_dir / "valid_graph_selection.csv", fields, rows)


def export_curve(out_dir, sources) -> list[Path]:
    """Gain vs graph strength: valid (37 configs x 2 visions) and test (5 configs), plus the test predictions E1-E4."""
    out = []
    for split, p in (("valid", R / "curve" / "summary_valid.json"), ("test", R / "curve_test" / "summary_test.json")):
        d = _json(p)
        sources.append(p)
        rows = []
        for r in d["rows"]:
            w = r.get("alpha_star", r.get("weight"))
            g = r["gain"]
            rows.append(dict(split=split, vision=r["vision"], encoder=r["encoder"], cfg=r["cfg"],
                             graph_auc=round(r["graph_auc"], 6), vision_auc=round(r["vision_auc"], 6),
                             weight=w, fusion_auc=round(r["auc_means"][_a(w)], 6),
                             best_component=r.get("best_component", r.get("reference_alpha")),
                             gain=round(g.get("mean", g.get("delta")), 6),
                             ci_lo=round(g["ci_lo"], 6), ci_hi=round(g["ci_hi"], 6)))
        fields = list(rows[0].keys())
        out.append(write_csv(out_dir / f"{split}_curve.csv", fields, rows))
        if split == "test":
            pr = []
            for fam, items in d["predictions"].items():
                if fam in ("shape", "descriptive_head_t05"):
                    continue
                for it in items:
                    pr.append(dict(family=fam, label=it["label"], predicted_sign=it["predicted_sign"],
                                   mean=round(it["mean"], 6), ci_lo=round(it["ci_lo"], 6),
                                   ci_hi=round(it["ci_hi"], 6), outcome=it["esito"]))
            t5 = d["predictions"]["descriptive_head_t05"]
            pr.append(dict(family="descrittivo", label="g_H − g_F per gat/t05", predicted_sign="",
                           mean=round(t5["mean"], 6), ci_lo=round(t5["ci_lo"], 6), ci_hi=round(t5["ci_hi"], 6),
                           outcome=""))
            pr.append(dict(family="forma", label=d["predictions"]["shape"], predicted_sign="", mean="",
                           ci_lo="", ci_hi="", outcome=""))
            out.append(write_csv(out_dir / "test_curve_predictions.csv", list(pr[0].keys()), pr))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSV di RESET GRAPHS (valid e test)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out_dir
    if out.exists():
        sys.exit(f"!! {out} esiste già: niente sovrascritture (scegli --out-dir nuova)")
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        sys.exit(f"!! {tmp} esiste (export precedente interrotto): controllarlo a mano")
    sources, checks, written, notes = [], [], [], []
    for split in ("valid", "test"):
        syst = systems(split)
        names, aligned = aligned_auc(syst, split)
        if len(names) != 2000:
            raise ValueError(f"{split}: {len(names)} query appaiate fra tutti i sistemi, attese 2000")
        sources += [str(c["prefix"]) + f"_*{split}.npz" for c in syst.values()]
        checks += check_published(split, aligned, sources)
        written += [export_summary_reset(names, aligned, syst, split, tmp),
                    export_damage_curve(names, aligned, syst, split, tmp),
                    export_perquery_selfrr(names, aligned, split, tmp),
                    export_perquery_full(syst, split, tmp),
                    export_full_metrics_all_k(syst, split, tmp),
                    export_paired(names, aligned, split, tmp),
                    export_seeds(split, tmp, sources)]
        notes.append(f"{split}: {len(names)} query appaiate su {len(syst)} sistemi (seed {DECIDING_SEED})")
    written.append(export_alpha_sweep(tmp, sources))
    written += export_controls(tmp, sources)
    written.append(export_selection(tmp, sources))
    written += export_curve(tmp, sources)
    tmp.rename(out)
    written = [out / Path(w).name for w in written]
    lines = ["# results/csv_final — export dei risultati di RESET GRAPHS (nessun numero nuovo)",
             f"# generato: {datetime.now().isoformat(timespec='seconds')}",
             "# comando: python -m src.evaluation.export_csv_final" + (f" --out-dir {out}" if out != DEFAULT_OUT else ""),
             "", "## come leggere",
             "- metro e convenzioni come in results/csv_historical/_sources.txt (self_rr, AUC su f 0.25/0.5/0.75, nDCG graduato,",
             "  per-query a precisione piena, delta full_plan descrittivi per circolarità)",
             "- sistemi del seed 42: graph_W = gat/comb · vision_frozen = pespatial/gem/whiten · vision_head = con head",
             f"  + whitening del train · fusion_main alpha {ALPHA_MAIN} · fusion_head alpha {ALPHA_HEAD[DECIDING_SEED]} ·",
             "  layoutgkn = LayoutGKN riallenato come W · hist_baseline (solo test) = baseline senza training",
             "- graph_W e vision_* sono gli estremi alpha=0 / alpha=1 delle fusioni (C3: riproducono i rami)",
             "- test letto una volta (status.md §63); valid = selezione (§59-§61); curva del valid ESPLORATIVA",
             "", "## verifiche superate", *[f"- {c}" for c in checks],
             "- 2000/2000 query appaiate fra tutti i sistemi, su valid e test",
             "- num_relevant identico fra i sistemi (ground truth condivisa)",
             "", "## note", *[f"- {n}" for n in notes],
             "", "## letti", *[f"- {s}" for s in dict.fromkeys(map(str, sources))],
             "", "## scritti", *[f"- {w}" for w in written], ""]
    (out / "_sources.txt").write_text("\n".join(lines), encoding="utf-8")
    for w in written:
        print(f"scritto: {w}")
    print(f"scritto: {out / '_sources.txt'}")


if __name__ == "__main__":
    main()
