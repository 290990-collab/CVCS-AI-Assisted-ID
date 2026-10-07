"""
Figures of the final pipeline with graph W = gat/comb, from the CSVs of `results/csv_final/`.
The CSVs are written by `src.evaluation.export_csv_final`, which checks every mean against the published json;
the historical fusion json gives one reference point of the curve.

    damage_test   TEST, seed 42: self-recovery vs fraction removed. Left single systems
                  (W, LayoutGKN, vision frozen / with head, hist baseline), right the fusions.
    gain_seeds    gain of the two fusions over W, 4 replicas, valid (hollow) and test (filled).
    controls      gain of the fusion (vision frozen / with head) against the two ensemble controls
                  (W + second copy of W, W + sage/comb), valid and test.
    curve         fusion gain vs graph strength: valid (37 graphs, exploratory) and test (5 graphs).
    alpha         VALID: AUC vs fusion weight alpha, main fusion and with head, 4 replicas.

Usage (CPU, seconds):

    python -m src.figures.final_results all --lang en --out-dir figures/final/english
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.figures.style import OKABE_ITO, TEXT_WIDTH_IN, apply_style, new_figure, save_figure

CSV_DIR = Path("results/csv_final")
HISTORICAL_SELECT = Path("results/fusion/select_valid.json")    # gat/asymrob + vision frozen
NOISE = 0.04                                                    # noise threshold between graph trainings

STYLE = {
    "graph_W": {"color": OKABE_ITO["blue"], "linestyle": "--", "marker": "s"},
    "layoutgkn": {"color": OKABE_ITO["purple"], "linestyle": (0, (3, 1, 1, 1)), "marker": "D"},
    "vision_frozen": {"color": OKABE_ITO["vermillion"], "linestyle": "-.", "marker": "^"},
    "vision_head": {"color": OKABE_ITO["orange"], "linestyle": "-.", "marker": "v"},
    "hist_baseline": {"color": OKABE_ITO["grey"], "linestyle": ":", "marker": "x"},
    "fusion_main": {"color": OKABE_ITO["sky"], "linestyle": "-", "marker": "o"},
    "fusion_head": {"color": OKABE_ITO["green"], "linestyle": "-", "marker": "o"},
}
VISION_COLOR = {"frozen": OKABE_ITO["vermillion"], "head": OKABE_ITO["green"]}

LABELS = {
    "it": {
        "graph_W": "graph W (gat/comb)", "layoutgkn": "LayoutGKN", "vision_frozen": "vision congelato",
        "vision_head": "vision con head", "hist_baseline": "baseline hist",
        "fusion_main": "fusione (vision congelato)", "fusion_head": "fusione (vision con head)",
        "x_frac": "frazione di stanze tolte dalla query", "y_mrr": "MRR di auto-ritrovamento",
        "singles": "(a) sistemi singoli", "fusions": "(b) fusioni (scala ingrandita)",
        "replica": "replica (seed)", "gain": "guadagno su W (AUC)", "valid": "valid", "test": "test",
        "main": "vision congelato", "head": "vision con head", "noise": "soglia di rumore 0.04",
        "ctrl_frozen": "W + vision\ncongelato", "ctrl_rep": "W + 2ª\ncopia di W",
        "ctrl_enc": "W +\nsage/comb", "ctrl_head": "W + vision\ncon head",
        "g": "guadagno g (AUC)",
        "x_graph": "forza del graph da solo (AUC)", "y_gain": "guadagno della fusione (AUC)",
        "curve_valid": "valid, 37 graph (esplorativa)", "curve_test": "test, 5 graph (pre-registrata)",
        "historical": "graph storico\n(gat/asymrob)", "vision_line": "forza del vision",
        "alpha": "peso del vision α", "auc": "AUC di robustezza", "alpha_star": "α scelto",
        "offscale": "per α ≥ 0.7 l'AUC scende\nverso il vision da solo\n(fuori scala)",
    },
    "en": {
        "graph_W": "graph W (gat/comb)", "layoutgkn": "LayoutGKN", "vision_frozen": "frozen vision",
        "vision_head": "vision with head", "hist_baseline": "hist baseline",
        "fusion_main": "fusion (frozen vision)", "fusion_head": "fusion (vision with head)",
        "x_frac": "fraction of rooms removed from the query", "y_mrr": "self-recovery MRR",
        "singles": "(a) single systems", "fusions": "(b) fusions (zoomed)",
        "replica": "replica (seed)", "gain": "gain over W (AUC)", "valid": "valid", "test": "test",
        "main": "frozen vision", "head": "vision with head", "noise": "noise threshold 0.04",
        "ctrl_frozen": "W + frozen\nvision", "ctrl_rep": "W + 2nd\ncopy of W",
        "ctrl_enc": "W +\nsage/comb", "ctrl_head": "W + vision\nwith head",
        "g": "gain g (AUC)",
        "x_graph": "strength of the graph alone (AUC)", "y_gain": "fusion gain (AUC)",
        "curve_valid": "valid, 37 graphs (exploratory)", "curve_test": "test, 5 graphs (pre-registered)",
        "historical": "historical graph\n(gat/asymrob)", "vision_line": "vision strength",
        "alpha": "vision weight α", "auc": "robustness AUC", "alpha_star": "chosen α",
        "offscale": "for α ≥ 0.7 the AUC falls\ntowards the vision alone\n(off scale)",
    },
}


def read_csv(name: str) -> list[dict]:
    with (CSV_DIR / name).open(newline="") as h:
        return list(csv.DictReader(h))


def num(x) -> float:
    return float(x) if x not in ("", None) else float("nan")


# --- damage curve on the test ---

def fig_damage_test(lang: str):
    t = LABELS[lang]
    rows = read_csv("test_damage_curve.csv")
    summary = {r["system"]: num(r["robustness_auc"]) for r in read_csv("test_summary.csv")}
    curve = {}
    for r in rows:
        curve.setdefault(r["system"], []).append((num(r["fraction"]), num(r["mean_self_rr"]),
                                                  num(r["ci_lo"]), num(r["ci_hi"])))
    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.5))
    panels = ((axes[0], ("hist_baseline", "vision_frozen", "vision_head", "layoutgkn", "graph_W"), t["singles"],
               (-0.02, 1.02)),
              (axes[1], ("graph_W", "fusion_main", "fusion_head"), t["fusions"], (0.76, 1.0)))
    for ax, keys, title, ylim in panels:
        for k in keys:
            pts = sorted(curve[k])
            x = np.array([p[0] for p in pts])
            ax.fill_between(x, [p[2] for p in pts], [p[3] for p in pts], color=STYLE[k]["color"],
                            alpha=0.18, linewidth=0)
            ax.plot(x, [p[1] for p in pts], label=f"{t[k]} — AUC {summary[k]:.3f}",
                    markerfacecolor="white", markeredgewidth=1.2, **STYLE[k])
        ax.set_title(title, loc="left")
        ax.set_xlabel(t["x_frac"])
        ax.set_xticks([0.0, 0.25, 0.5, 0.75])
        ax.set_xlim(-0.03, 0.78)
        ax.set_ylim(*ylim)
    axes[1].legend(loc="lower left")
    h, lab = axes[0].get_legend_handles_labels()
    axes[0].set_ylabel(t["y_mrr"])
    fig.tight_layout(w_pad=1.5, rect=(0, 0.13, 1, 1))
    fig.legend(h, lab, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=3, columnspacing=1.2)
    notes = [f"{k}: AUC {v:.4f}" for k, v in summary.items()]
    return fig, [CSV_DIR / "test_damage_curve.csv", CSV_DIR / "test_summary.csv"], notes


# --- gains over W, 4 replicas ---

def fig_gain_seeds(lang: str):
    t = LABELS[lang]
    fig, ax = new_figure("column", height_in=2.4)
    notes = []
    data = {s: read_csv(f"{s}_seeds.csv") for s in ("valid", "test")}
    seeds = [r["seed"] for r in data["test"]]
    x0 = np.arange(len(seeds))
    for j, (kind, col, key) in enumerate((("main", "gain_main", "fusion_main"), ("head", "gain_head", "fusion_head"))):
        for i, split in enumerate(("valid", "test")):
            rows = {r["seed"]: r for r in data[split]}
            m = np.array([num(rows[s][col]) for s in seeds])
            lo = np.array([num(rows[s][f"{col}_ci_lo"]) for s in seeds])
            hi = np.array([num(rows[s][f"{col}_ci_hi"]) for s in seeds])
            x = x0 + (j - 0.5) * 0.36 + (i - 0.5) * 0.14
            c = STYLE[key]["color"]
            ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o", color=c, markersize=4,
                        markerfacecolor=("white" if split == "valid" else c), markeredgewidth=1.1,
                        elinewidth=0.9, label=f"{t[kind]} · {t[split]}")
            notes.append(f"{kind} {split}: " + "  ".join(f"s{s} {v:+.4f}" for s, v in zip(seeds, m)))
    ax.axhline(0, color="0.5", linewidth=0.6)
    ax.set_xticks(x0)
    ax.set_xticklabels(seeds)
    ax.set_xlabel(t["replica"])
    ax.set_ylabel(t["gain"])
    ax.set_ylim(0, None)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.22), columnspacing=1.0)
    return fig, [CSV_DIR / "valid_seeds.csv", CSV_DIR / "test_seeds.csv"], notes


# --- the thesis question: fusion vs ensemble controls ---

def fig_controls(lang: str):
    t = LABELS[lang]
    rows = {(r["split"], r["row"]): r for r in read_csv("controls_gain.csv")}
    vseed = {r["seed"]: r for r in read_csv("valid_seeds.csv")}["42"]
    cats = [("ctrl_frozen", ("valid", "g_fusion_main (vs replica)"), ("test", "g_F_frozen"), "fusion_main"),
            ("ctrl_rep", ("valid", "g_control_replica"), ("test", "g_C_rep"), "graph_W"),
            ("ctrl_enc", ("valid", "g_control_crossenc"), ("test", "g_C_enc"), "graph_W"),
            ("ctrl_head", None, ("test", "g_F_H"), "fusion_head")]
    fig, ax = new_figure("column", height_in=2.4)
    notes = []
    for i, (lab, vkey, tkey, style) in enumerate(cats):
        for j, split in enumerate(("valid", "test")):
            if split == "valid" and vkey is None:
                m, lo, hi = (num(vseed["gain_head"]), num(vseed["gain_head_ci_lo"]), num(vseed["gain_head_ci_hi"]))
            else:
                r = rows[vkey if split == "valid" else tkey]
                m, lo, hi = num(r["value"]), num(r["ci_lo"]), num(r["ci_hi"])
            c = STYLE[style]["color"]
            ax.bar(i + (j - 0.5) * 0.38, m, width=0.34, color=c if split == "test" else "white",
                   edgecolor=c, linewidth=1.1, hatch=None if split == "test" else "////",
                   )
            ax.errorbar(i + (j - 0.5) * 0.38, m, yerr=[[m - lo], [hi - m]], color="0.2", elinewidth=0.8,
                        capsize=1.6)
            notes.append(f"{lab} {split}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    for k in ("D_rep", "D_enc"):
        r = rows[("test", k)]
        notes.append(f"test {k}: {num(r['value']):+.4f} [{num(r['ci_lo']):+.4f}, {num(r['ci_hi']):+.4f}]")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([t[c[0]] for c in cats], fontsize=6.3)
    ax.set_ylabel(t["g"])
    ax.set_ylim(0, None)
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="white", edgecolor="0.3", hatch="////"), Patch(facecolor="0.5", edgecolor="0.3")],
              [t["valid"], t["test"]], loc="upper left")
    ax.grid(axis="x", visible=False)
    return fig, [CSV_DIR / "controls_gain.csv", CSV_DIR / "valid_seeds.csv"], notes


# --- gain vs strength of the graph ---

def fig_curve(lang: str):
    t = LABELS[lang]
    valid, test = read_csv("valid_curve.csv"), read_csv("test_curve.csv")
    hist = json.loads(HISTORICAL_SELECT.read_text())
    h_graph = float(hist["auc_means"]["0"])
    h_gain = float(hist["auc_means"][f"{hist['alpha_star']:g}"]) - max(h_graph, float(hist["auc_means"]["1"]))
    fig, ax = new_figure("column", height_in=3.0)
    notes = []
    for vis in ("frozen", "head"):
        c = VISION_COLOR[vis]
        v = [r for r in valid if r["vision"] == vis]
        ax.scatter([num(r["graph_auc"]) for r in v], [num(r["gain"]) for r in v], s=9, facecolors="none",
                   edgecolors=c, linewidths=0.7, alpha=0.75)
        te = sorted((r for r in test if r["vision"] == vis), key=lambda r: num(r["graph_auc"]))
        x = np.array([num(r["graph_auc"]) for r in te])
        m = np.array([num(r["gain"]) for r in te])
        lo = np.array([num(r["ci_lo"]) for r in te])
        hi = np.array([num(r["ci_hi"]) for r in te])
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="-o", color=c, markersize=4, linewidth=1.1,
                    elinewidth=0.8, label=t["main" if vis == "frozen" else "head"])
        va = num(v[0]["vision_auc"])
        ax.axvline(va, color=c, linewidth=0.7, linestyle=":")
        notes.append(f"{vis}: vision valid {va:.4f} · test " +
                     "  ".join(f"{r['encoder']}/{r['cfg']} {num(r['graph_auc']):.3f}→{num(r['gain']):+.4f}" for r in te))
    ax.scatter([h_graph], [h_gain], marker="*", s=60, color="0.15", zorder=5)
    ax.annotate(t["historical"], xy=(h_graph, h_gain), xytext=(0.03, 0.215),
                fontsize=6, color="0.25", arrowprops={"arrowstyle": "-", "color": "0.4", "linewidth": 0.6})
    w = [r for r in test if r["vision"] == "frozen" and r["cfg"] == "comb"][0]
    ax.annotate("W", xy=(num(w["graph_auc"]), num(w["gain"])), xytext=(num(w["graph_auc"]) - 0.06, 0.07),
                fontsize=6.5, arrowprops={"arrowstyle": "-", "color": "0.4", "linewidth": 0.6})
    notes.append(f"storico (valid, gat/asymrob): graph {h_graph:.4f}, guadagno {h_gain:+.4f} (da {HISTORICAL_SELECT})")
    ax.axhline(0, color="0.5", linewidth=0.6)
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="0.4",
                       markersize=3.5),
                Line2D([], [], marker="o", linestyle="-", color="0.4", markersize=3.5),
                Line2D([], [], linestyle=":", color="0.4")]
    labels += [t["curve_valid"], t["curve_test"], t["vision_line"]]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6,
              columnspacing=1.0)
    ax.set_xlabel(t["x_graph"])
    ax.set_ylabel(t["y_gain"])
    ax.set_xlim(0, 1)
    return fig, [CSV_DIR / "valid_curve.csv", CSV_DIR / "test_curve.csv", HISTORICAL_SELECT], notes


# --- AUC vs alpha on the valid ---

def fig_alpha(lang: str):
    t = LABELS[lang]
    rows = read_csv("valid_alpha_sweep.csv")
    fig, ax = new_figure("column", height_in=2.4)
    notes = []
    for kind, key in (("main", "fusion_main"), ("head", "fusion_head")):
        for seed in ("42", "100042", "200042", "300042"):
            rs = sorted((r for r in rows if r["fusion"] == f"{key}/s{seed}"), key=lambda r: num(r["alpha"]))
            a = np.array([num(r["alpha"]) for r in rs])
            y = np.array([num(r["robustness_auc"]) for r in rs])
            keep = a <= 0.6
            bold = seed == "42"
            ax.plot(a[keep], y[keep], color=STYLE[key]["color"], linewidth=1.3 if bold else 0.6,
                    alpha=1 if bold else 0.5, marker="o" if bold else None, markersize=2.8,
                    label=t[kind] if bold else None)
            star = [r for r in rs if r["is_alpha_star"] == "1"][0]
            ax.scatter([num(star["alpha"])], [num(star["robustness_auc"])], s=26, marker="*",
                       color=STYLE[key]["color"], zorder=5, edgecolors="0.1", linewidths=0.4)
            notes.append(f"{key}/s{seed}: α* {num(star['alpha']):g} AUC {num(star['robustness_auc']):.4f} · "
                         f"α=0 {y[a == 0][0]:.4f} · α=1 {y[a == 1][0]:.4f}")
    ax.text(0.02, 0.98, t["offscale"], transform=ax.transAxes, ha="left", va="top", fontsize=6, color="0.4",
            linespacing=1.3)
    ax.set_xlabel(t["alpha"])
    ax.set_ylabel(t["auc"])
    ax.set_xlim(-0.02, 0.62)
    ax.legend(loc="lower center")
    return fig, [CSV_DIR / "valid_alpha_sweep.csv"], notes


FIGURES = {"damage_test": ("f_final_damage_test", fig_damage_test),
           "gain_seeds": ("f_final_gain_seeds", fig_gain_seeds),
           "controls": ("f_final_controls", fig_controls),
           "curve": ("f_final_curve", fig_curve),
           "alpha": ("f_final_alpha_valid", fig_alpha)}


def main() -> None:
    p = argparse.ArgumentParser(description="Figures of RESET GRAPHS")
    p.add_argument("figure", choices=(*FIGURES, "all"))
    p.add_argument("--lang", choices=("it", "en"), default="en")
    p.add_argument("--out-dir", default="figures/final/english")
    a = p.parse_args()
    for key in (FIGURES if a.figure == "all" else [a.figure]):
        name, fn = FIGURES[key]
        fig, sources, notes = fn(a.lang)
        for line in notes:
            print(line)
        save_figure(fig, name, sources, notes=notes, out_dir=Path(a.out_dir))


if __name__ == "__main__":
    main()
