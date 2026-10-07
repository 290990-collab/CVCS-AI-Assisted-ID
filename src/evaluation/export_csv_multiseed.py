"""Export the second-round results (4 replicas, head v2; valid and test) to CSV.

Complements `results/csv_final/`. Reads only the json written by the analysis:

    results/final_pipeline/round2/analysis_valid.json   A-E on the valid
    results/final_pipeline/round2/head_v2_valid.json    H on the valid
    results/final_pipeline/round2/geometry_valid.json   I on the valid
    results/final_pipeline/round2/analysis_test.json    A-E, H, I and predictions on the test

Constraints (as `export_csv_final`):
  * no new numbers: json are copied;
  * checks: per-replica fusion AUC equals the fusion's `select_<split>.json` (tolerance TOL) and the
    recomputed geometry (1,1,1) equals the stored value; otherwise stop;
  * writes only to a new folder (default `results/csv_multiseed/`), refuses if it exists;
  * `_sources.txt` next to the CSVs lists files read, command and checks.

Usage (CPU, seconds):
    python -m src.evaluation.export_csv_multiseed
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_OUT = Path("results/csv_multiseed")
R2 = Path("results/final_pipeline/round2")
FILES = {"analysis_valid": R2 / "analysis_valid.json", "head_v2_valid": R2 / "head_v2_valid.json",
         "geometry_valid": R2 / "geometry_valid.json", "analysis_test": R2 / "analysis_test.json"}
SEEDS = ("42", "100042", "200042", "300042")
CONTROLS = ("replica", "crossenc")
TOL = 1e-9


def _load() -> dict:
    j = {k: json.loads(p.read_text()) for k, p in FILES.items()}
    return {"valid": {"A": j["analysis_valid"]["A_frozen_vs_ensemble"], "B": j["analysis_valid"]["B_head_vs_ensemble"],
                      "C": j["analysis_valid"]["C_mid_strength"], "D": j["analysis_valid"]["D_curve_4_seeds"],
                      "E": j["analysis_valid"]["E_W_vs_sage"], "H": j["head_v2_valid"], "I": j["geometry_valid"]},
            "test": {"A": j["analysis_test"]["A_frozen_vs_ensemble"], "B": j["analysis_test"]["B_head_vs_ensemble"],
                     "C": j["analysis_test"]["C_mid_strength"], "D": j["analysis_test"]["D_curve_4_seeds"],
                     "E": j["analysis_test"]["E_W_vs_sage"], "H": j["analysis_test"]["H_head_v2"],
                     "I": j["analysis_test"]["I_geometry"], "P": j["analysis_test"]["predictions"]}}


def write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    with path.open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else (repr(v) if isinstance(v, float) else v) for v in r])
    return path


def _ci(d: dict, key: str = "mean") -> list:
    return [d[key], d["ci_lo"], d["ci_hi"]]


# --- checks (nothing is exported if one fails) ---

def check_fusions(data: dict) -> list[str]:
    """Every per-replica fusion AUC = auc_means[alpha*] of the select json of its folder."""
    done = []
    for split, d in data.items():
        rows = []
        for item in ("A", "B"):
            for s, r in d[item]["per_replica"].items():
                rows.append((f"{item} fusione s{s}", r["fusion"]))
                rows += [(f"{item} controllo {k} s{s}", r[k]["control"]) for k in CONTROLS]
        for s, r in d["H"]["measure2_fusion_vs_head"]["per_replica"].items():
            rows += [(f"H fusione v2 s{s}", r["fusion_v2"]), (f"H fusione head s{s}", r["fusion_head"])]
        for label, f in rows:
            sel = json.loads((Path(f["dir"]).parent / f"select_{split}.json").read_text())
            pub = float(sel["auc_means"][f"{float(f['weight']):g}"])
            if abs(pub - f["auc"]) > TOL or abs(float(sel["alpha_star"]) - f["weight"]) > TOL:
                raise SystemExit(f"!! {split} {label}: AUC {f['auc']} / peso {f['weight']} contro select "
                                 f"{pub} / {sel['alpha_star']}")
        done.append(f"{split}: {len(rows)} fusioni per replica = select_{split}.json (AUC e peso, entro {TOL:g})")
        for lab, c in d["I"]["check_baseline"].items():
            if abs(c["recomputed_111"] - c["stored"]) > 1e-6:
                raise SystemExit(f"!! {split} geometria {lab}: ricalcolo (1,1,1) != valore salvato")
        done.append(f"{split}: geometria (1,1,1) ricalcolata = salvata nei per-query, {len(d['I']['check_baseline'])} sistemi")
    return done


# --- tables ---

VISIONS = (("frozen", "A"), ("head", "B"))


def ensemble_rows(data: dict) -> list[list]:
    rows = []
    for split, d in data.items():
        items = [(v, d[i]) for v, i in VISIONS] + [("head_v2", d["H"]["measure3_vs_ensemble"])]
        for vision, x in items:
            for k in CONTROLS:
                g = x[f"G_{k}"]
                outcome = x.get(f"outcome_{k}") or x.get("outcome")
                rows.append([split, vision, k, "media 4 repliche", *_ci(x[f"D_{k}_pooled"]), None, None, None, None,
                             None, x[f"n_replicas_ci_pos_{k}"], g["sd"], g["verdict"], outcome])
                for s in SEEDS:
                    r = x["per_replica"][s]
                    rows.append([split, vision, k, f"s{s}", *_ci(r[k]["D"]), r["g_F"]["mean"], r[k]["g_C"]["mean"],
                                 r["fusion"]["auc"], r["fusion"]["weight"], r[k]["control"]["weight"],
                                 None, None, None, None])
    return rows


def mid_rows(data: dict) -> list[list]:
    rows = []
    for split, d in data.items():
        for key, r in d["C"].items():
            graph, vision = key.split("+")
            rows.append([split, graph.replace("_", "/"), vision, int(bool(r["decisive"])), *_ci(r["g_F"]),
                         *_ci(r["replica"]["D"]), *_ci(r["crossenc"]["D"]), r["outcome"]])
    return rows


def curve_rows(data: dict) -> list[list]:
    rows = []
    for split, d in data.items():
        for r in d["D"]["rows"]:
            rows.append([split, r["vision"], r["encoder"], r["cfg"], r["graph_auc_mean"], *_ci(r["gain_pooled"]),
                         *[r["gain_per_seed"][s] for s in SEEDS], *[r["alpha_star"][s] for s in SEEDS]])
    return rows


def curve_pred_rows(data: dict) -> list[list]:
    rows = []
    for split, d in data.items():
        D = d["D"]
        for key in ("E1_falling_frozen", "E2_rising_frozen", "E3_head_vs_frozen", "E4"):
            for p in D["predictions_pooled"][key]:
                g = D["G_E1_E2"].get(p["label"], {})
                rows.append([split, key, p["label"], *_ci(p), p["esito"], D["n_seeds_confirmed"][p["label"]],
                             g.get("verdict")])
    return rows


def w_sage_rows(data: dict) -> list[list]:
    rows = []
    for split, d in data.items():
        E = d["E"]
        rows.append([split, "media 4 repliche", None, None, *_ci(E["delta_pooled"]), E["outcome"], E["G"]["sd"],
                     E["G"]["verdict"]])
        for s in SEEDS:
            r = E["per_replica"][s]
            rows.append([split, f"s{s}", r["auc_W"], r["auc_E"], *_ci(r["delta"]), None, None, None])
    return rows


DAMAGES = ("nowalls-random", "crop", "patch", "media_tre_danni")


def damage_rows(data: dict) -> tuple[list[list], list[list]]:
    auc, deltas = [], []
    for split, d in data.items():
        m1 = d["H"]["measure1_robustness"]
        for system, a in m1["auc"].items():
            auc += [[split, system, dmg, a[dmg]] for dmg in DAMAGES]
        for other in ("vision_congelato", "vision_head"):
            for dmg in DAMAGES:
                c = m1[f"v2_minus_{other}"][dmg]
                deltas.append([split, f"vision_head_v2 - {other}", dmg, c["delta"], c["ci_lo"], c["ci_hi"],
                               c["n_pairs"]])
    return auc, deltas


def v2_fusion_rows(data: dict) -> list[list]:
    rows = []
    for split, d in data.items():
        m2 = d["H"]["measure2_fusion_vs_head"]
        rows.append([split, "media 4 repliche", None, None, None, None, *_ci(m2["auc_v2_minus_head_pooled"]),
                     m2["G"]["sd"], m2["G"]["verdict"]])
        for s in SEEDS:
            r = m2["per_replica"][s]
            rows.append([split, f"s{s}", r["fusion_v2"]["auc"], r["fusion_v2"]["weight"], r["fusion_head"]["auc"],
                         r["fusion_head"]["weight"], *_ci(r["auc_v2_minus_head"]), None, None])
    return rows


def full_rows(data: dict) -> list[list]:
    return [[split, system, v["composition"], v["topology"], v["geometry"]]
            for split, d in data.items() for system, v in d["H"]["full_plan_ndcg10"].items()]


def geometry_rows(data: dict) -> tuple[list[list], list[list]]:
    means, pairs = [], []
    for split, d in data.items():
        I = d["I"]
        for w, by in I["ndcg_geometry_means"].items():
            means += [[split, w, system, v] for system, v in by.items()]
        for pair, p in I["pairs"].items():
            for r in p["rows"]:
                pairs.append([split, pair, ",".join(f"{x:g}" for x in r["weights"]), r["delta"], r["ci_lo"],
                              r["ci_hi"], p["verdict"]])
    return means, pairs


def export(out: Path) -> list[str]:
    data = _load()
    checks = check_fusions(data)
    out.mkdir(parents=True)
    written = [
        write_csv(out / "ensemble_D.csv",
                  ["split", "vision", "control", "scope", "D", "ci_lo", "ci_hi", "g_fusion", "g_control",
                   "fusion_auc", "fusion_weight", "control_weight", "n_replicas_ci_pos", "sd_replicas",
                   "noise_verdict", "outcome"], ensemble_rows(data)),
        write_csv(out / "mid_strength.csv",
                  ["split", "graph", "vision", "decisive", "g_fusion", "g_ci_lo", "g_ci_hi", "D_replica",
                   "D_replica_ci_lo", "D_replica_ci_hi", "D_crossenc", "D_crossenc_ci_lo", "D_crossenc_ci_hi",
                   "outcome"], mid_rows(data)),
        write_csv(out / "curve_4seeds.csv",
                  ["split", "vision", "encoder", "cfg", "graph_auc_mean", "gain", "ci_lo", "ci_hi",
                   *[f"gain_s{s}" for s in SEEDS], *[f"weight_s{s}" for s in SEEDS]], curve_rows(data)),
        write_csv(out / "curve_predictions.csv",
                  ["split", "group", "prediction", "mean", "ci_lo", "ci_hi", "esito", "n_seeds_confirmed",
                   "noise_verdict"], curve_pred_rows(data)),
        write_csv(out / "w_vs_sage.csv",
                  ["split", "scope", "auc_W", "auc_sage_comb", "delta", "ci_lo", "ci_hi", "outcome",
                   "sd_replicas", "noise_verdict"], w_sage_rows(data)),
    ]
    auc, deltas = damage_rows(data)
    written += [write_csv(out / "vision_damage_auc.csv", ["split", "system", "damage", "auc"], auc),
                write_csv(out / "vision_damage_deltas.csv",
                          ["split", "comparison", "damage", "delta", "ci_lo", "ci_hi", "n_pairs"], deltas),
                write_csv(out / "head_v2_fusion.csv",
                          ["split", "scope", "fusion_v2_auc", "fusion_v2_weight", "fusion_head_auc",
                           "fusion_head_weight", "delta", "ci_lo", "ci_hi", "sd_replicas", "noise_verdict"],
                          v2_fusion_rows(data)),
                write_csv(out / "full_plan_ndcg10.csv", ["split", "system", "composition", "topology", "geometry"],
                          full_rows(data))]
    means, pairs = geometry_rows(data)
    written += [write_csv(out / "geometry_weightings.csv", ["split", "weights", "system", "ndcg10_geometry"], means),
                write_csv(out / "geometry_pairs.csv",
                          ["split", "pair", "weights", "delta", "ci_lo", "ci_hi", "verdict"], pairs),
                write_csv(out / "test_predictions.csv", ["id", "label", "esito"],
                          [[p["id"], p["label"], p["esito"]] for p in data["test"]["P"]])]
    lines = ["# results/csv_multiseed — export del secondo giro di RESET GRAPHS (nessun numero nuovo)",
             f"# generato: {datetime.now().isoformat(timespec='seconds')}",
             "# comando: python -m src.evaluation.export_csv_multiseed", "",
             "## come leggere",
             "- AUC = media del MRR di self-recovery su f 0.25/0.5/0.75; g = AUC(fusione) - AUC(ramo migliore della",
             "  coppia); D = g(fusione vision + W) - g(controllo d'insieme: W + seconda copia di W, W + sage/comb)",
             "- 'media 4 repliche' = media per query sulle 4 repliche, poi bootstrap (B = 10000, CI 95%)",
             "- vision: frozen = pespatial/gem/whiten congelato · head = head attuale (head_nowalls_conv) · head_v2 =",
             "  head residua solo sulla query, tre danni insieme (status.md §64.2)",
             "- valid = selezione e lettura provvisoria (§64.1, §64.3); test = lettura unica pre-registrata (§65-§66)",
             "- geometria: nDCG@10 a pianta intera, 6 pesature di §22, seed 42 (descrittivo, circolarità)", "",
             "## verifiche superate", *[f"- {c}" for c in checks], "",
             "## file letti", *[f"- {p}" for p in FILES.values()], "",
             "## file scritti", *[f"- {p.name}" for p in written], ""]
    (out / "_sources.txt").write_text("\n".join(lines))
    return [str(p) for p in written]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="export_csv_multiseed")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.out_dir.exists():
        raise SystemExit(f"!! {a.out_dir} esiste gia': niente sovrascritture (usa --out-dir con una cartella nuova)")
    for p in export(a.out_dir):
        print(f"scritto {p}")
    print(f"scritto {a.out_dir / '_sources.txt'}")


if __name__ == "__main__":
    main()
