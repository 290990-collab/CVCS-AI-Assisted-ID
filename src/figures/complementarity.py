"""
Figures of the complementarity gain: how much of a fusion gain is vision information, and where it appears.

    decomposition  each fusion gain g split into the ensemble part (the larger gain of the two controls,
                   W + second copy of W and W + sage/comb) and the complementarity gain CG = g(fusion) - g(control),
                   with 95% CI. (a) strong graph W, test, mean of 4 replicas (results/csv_multiseed/ensemble_D.csv);
                   (b) graphs as strong as the vision, test, seed 42 (results/csv_multiseed/mid_strength.csv).
    where          CG per query group (test bars, valid markers; seed 42): (a) damage level, (b) W alone perfect or not,
                   (c) plans with the same room graph in the gallery, (d) rooms of the plan.
                   From results/csv_complementarity/complementarity_by_query.csv (post-hoc, descriptive).

Writes only new files (refuses if the figure already exists in --out-dir).

Usage (CPU, seconds):
    python -m src.figures.complementarity all --out-dir figures/final/english
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.figures.style import COLUMN_WIDTH_IN, OKABE_ITO, TEXT_WIDTH_IN, apply_style, save_figure

MULTI = Path("results/csv_multiseed")
BYQ = Path("results/csv_complementarity/complementarity_by_query.csv")
CURVE = MULTI / "curve_4seeds.csv"
VISIONS = ("frozen", "head", "head_v2")
COLOR = {"frozen": OKABE_ITO["vermillion"], "head": OKABE_ITO["green"], "head_v2": OKABE_ITO["blue"]}
NAME = {"frozen": "frozen vision", "head": "vision + head", "head_v2": "vision + head v2"}
SHORT = {"frozen": "frozen", "head": "head", "head_v2": "head v2"}
ENSEMBLE_C = "0.78"
CONTROL_NAME = {"replica": "graph + its 2nd copy", "crossenc": "graph + SAGE with the same recipe"}


def read(path: Path) -> list[dict]:
    with path.open(newline="") as h:
        return list(csv.DictReader(h))


def num(x) -> float:
    return float(x) if x not in ("", None) else float("nan")


def decomposition_rows() -> tuple[list[dict], list[dict]]:
    """[(label, vision, g, g_ctrl, ctrl, cg, lo, hi)] for W (4 replicas) and for the mid-strength graphs."""
    ens = [r for r in read(MULTI / "ensemble_D.csv") if r["split"] == "test"]
    strong = []
    for v in VISIONS:
        per_ctrl = {}
        for c in ("replica", "crossenc"):
            seeds = [r for r in ens if r["vision"] == v and r["control"] == c and r["scope"].startswith("s")]
            mean = next(r for r in ens if r["vision"] == v and r["control"] == c and r["scope"] == "media 4 repliche")
            if len(seeds) != 4:
                raise ValueError(f"{v}/{c}: {len(seeds)} replicas, expected 4")
            per_ctrl[c] = {"g": np.mean([num(r["g_fusion"]) for r in seeds]),
                           "gc": np.mean([num(r["g_control"]) for r in seeds]),
                           "D": num(mean["D"]), "lo": num(mean["ci_lo"]), "hi": num(mean["ci_hi"])}
        c = max(per_ctrl, key=lambda k: per_ctrl[k]["gc"])          # conservative control
        p = per_ctrl[c]
        if abs((p["g"] - p["gc"]) - p["D"]) > 1e-6:
            raise ValueError(f"{v}/{c}: mean g − mean g_control ≠ D of the replica mean")
        strong.append({"label": f"W + {SHORT[v]}", "vision": v, "g": p["g"], "gc": p["gc"], "ctrl": c,
                       "cg": p["D"], "lo": p["lo"], "hi": p["hi"]})
    auc = {(r["vision"], r["cfg"]): num(r["graph_auc_mean"]) for r in read(CURVE)
           if r["split"] == "test" and r["encoder"] == "gat"}
    mid = []
    for r in read(MULTI / "mid_strength.csv"):
        if r["split"] != "test":
            continue
        g = num(r["g_fusion"])
        c = min(("replica", "crossenc"), key=lambda k: num(r[f"D_{k}"]))  # larger control gain = smaller D
        cfg = r["graph"].split("/")[1]
        mid.append({"label": f"{r['graph']} ({auc[(r['vision'], cfg)]:.2f}) + {SHORT[r['vision']]}",
                    "vision": r["vision"], "g": g, "gc": g - num(r[f"D_{c}"]), "ctrl": c,
                    "cg": num(r[f"D_{c}"]), "lo": num(r[f"D_{c}_ci_lo"]), "hi": num(r[f"D_{c}_ci_hi"])})
    return strong, mid


def fig_decomposition():
    strong, mid = decomposition_rows()
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 3.0),
                             gridspec_kw={"height_ratios": [len(strong), len(mid)]})
    titles = ("(a) strong graph W (AUC 0.90), test, mean of 4 replicas",
              "(b) graphs as strong as the vision (AUC), test, seed 42")
    notes = []
    for ax, rows, title in zip(axes, (strong, mid), titles):
        for i, r in enumerate(rows[::-1]):
            col = COLOR[r["vision"]]
            ens = min(r["g"], r["gc"])
            ax.barh(i, ens, height=0.6, color=ENSEMBLE_C, edgecolor="white", linewidth=0.6, zorder=2)
            if r["cg"] >= 0:
                ax.barh(i, r["cg"], left=r["gc"], height=0.6, color=col, edgecolor="white", linewidth=0.6, zorder=2)
            else:                                   # fusion below control: dashed outline
                ax.barh(i, -r["cg"], left=r["g"], height=0.6, facecolor="none", edgecolor=col, linewidth=0.8,
                        linestyle=(0, (2, 1.5)), zorder=2)
            ax.errorbar(r["gc"] + r["cg"], i, xerr=[[r["cg"] - r["lo"]], [r["hi"] - r["cg"]]], color="0.2",
                        elinewidth=0.8, capsize=1.6, zorder=3)
            notes.append(f"{r['label']}: g {r['g']:+.4f} · ensemble ({CONTROL_NAME[r['ctrl']]}) {r['gc']:+.4f}"
                         f" · CG {r['cg']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["label"] for r in rows[::-1]], fontsize=6.4)
        ax.set_title(title, fontsize=6.6, loc="left")
        ax.axvline(0, color="0.45", linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=0)
    axes[-1].set_xlabel("gain of the fusion over its best branch (robustness AUC)")
    fig.legend([Patch(facecolor=ENSEMBLE_C), Patch(facecolor="0.35"),
                Patch(facecolor="none", edgecolor="0.35", linestyle=(0, (2, 1.5)))],
               ["ensemble part (2nd graph model)", "complementarity gain", "fusion below the control"],
               loc="lower center", ncol=3, fontsize=5.6, bbox_to_anchor=(0.5, -0.01), handlelength=1.4,
               columnspacing=0.8)
    fig.tight_layout(h_pad=0.6, rect=(0, 0.06, 1, 1))
    return fig, [MULTI / "ensemble_D.csv", MULTI / "mid_strength.csv", CURVE], notes


PANELS = (("level", "(a) rooms removed", ("f=0.25", "f=0.5", "f=0.75"), ("25%", "50%", "75%")),
          ("graph_ok", "(b) W alone", ("W perfect", "W not perfect"), ("perfect", "not perfect")),
          ("twins", "(c) plans with the same room graph", ("0", "1-9", "10-99", "100+"), ("0", "1-9", "10-99", "100+")),
          ("rooms", "(d) rooms in the plan", ("≤5", "6-7", "≥8"), ("≤5", "6-7", "≥8")))


def fig_where():
    rows = read(BYQ)

    def pick(split, v, group, b):
        for r in rows:
            if r["split"] != split or r["vision"] != v:
                continue
            if group == "level" and r["level"] == b and r["group"] == "all":
                return r
            if group != "level" and r["level"] == "AUC" and r["group"] == group and r["bin"] == b:
                return r
        raise KeyError((split, v, group, b))

    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH_IN, 2.1), sharey=True,
                             gridspec_kw={"width_ratios": [3, 2, 4, 3]})
    notes, w = [], 0.26
    for ax, (group, title, bins, ticks) in zip(axes, PANELS):
        for i, b in enumerate(bins):
            for j, v in enumerate(VISIONS):
                x = i + (j - 1) * w
                te, va = pick("test", v, group, b), pick("valid", v, group, b)
                m, lo, hi = num(te["CG"]), num(te["CG_lo"]), num(te["CG_hi"])
                ax.bar(x, m, width=w * 0.9, color=COLOR[v], edgecolor=COLOR[v], linewidth=0.5, zorder=2)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="0.2", elinewidth=0.7, capsize=1.2, zorder=3)
                ax.scatter([x], [num(va["CG"])], marker="D", s=7, facecolors="white", edgecolors="0.15",
                           linewidths=0.5, zorder=4)
                notes.append(f"{group} {b} {v}: test CG {m:+.4f} [{lo:+.4f}, {hi:+.4f}] n={te['n']} "
                             f"(vs {te['control_cg']}) · valid {num(va['CG']):+.4f} n={va['n']}")
        ax.axhline(0, color="0.45", linewidth=0.6, zorder=1)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(ticks, fontsize=6.2)
        ax.set_title(title, fontsize=6.6)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("complementarity gain CG")
    handles = [Patch(facecolor=COLOR[v], edgecolor=COLOR[v]) for v in VISIONS]
    handles.append(Line2D([], [], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor="0.15",
                          markersize=3.2))
    fig.legend(handles, [f"W + {NAME[v]} (test)" for v in VISIONS] + ["valid"], loc="lower center", ncol=4,
               fontsize=6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=0.5, rect=(0, 0.09, 1, 1))
    return fig, [BYQ], notes


FIGURES = {"decomposition": ("f_final_complementarity", fig_decomposition),
           "where": ("f_final_complementarity_where", fig_where)}


def main() -> None:
    p = argparse.ArgumentParser(description="Figures of the complementarity gain")
    p.add_argument("figure", choices=(*FIGURES, "all"))
    p.add_argument("--out-dir", default="figures/final/english")
    a = p.parse_args()
    out = Path(a.out_dir)
    keys = list(FIGURES) if a.figure == "all" else [a.figure]
    for key in keys:
        if (out / f"{FIGURES[key][0]}.pdf").exists():
            raise SystemExit(f"!! {out / FIGURES[key][0]}.pdf esiste gia': niente sovrascritture")
    for key in keys:
        name, fn = FIGURES[key]
        fig, sources, notes = fn()
        save_figure(fig, name, sources, notes=notes, out_dir=out)


if __name__ == "__main__":
    main()
