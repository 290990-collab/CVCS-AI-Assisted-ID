"""
Figures of the second round of graph experiments, from `results/csv_multiseed/`.
The CSVs are written by `src.evaluation.export_csv_multiseed`, which checks every fusion against its select json.

    ensemble   D = gain of the fusion vision + W minus the gain of an ensemble control (W + second copy of W,
               W + sage/comb), for the frozen vision, the current head and head v2; valid (hatched) and test
               (filled), 95% CI; grey band = equivalence margin +-0.005.
    damage     robustness of the three visions alone, per damage and mean of the three: test bars, valid markers.

Writes only new files (refuses if the figure already exists in --out-dir).

Usage (CPU, seconds):
    python -m src.figures.multiseed_results all --lang en --out-dir figures/multiseed/english
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.figures.style import COLUMN_WIDTH_IN, OKABE_ITO, apply_style, save_figure

CSV_DIR = Path("results/csv_multiseed")
DELTA = 0.005                                  # equivalence margin
VISIONS = ("frozen", "head", "head_v2")
COLOR = {"frozen": OKABE_ITO["vermillion"], "head": OKABE_ITO["green"], "head_v2": OKABE_ITO["blue"]}
SYSTEM = {"frozen": "vision_congelato", "head": "vision_head", "head_v2": "vision_head_v2"}
DAMAGES = ("nowalls-random", "crop", "patch", "media_tre_danni")

LABELS = {
    "it": {"frozen": "vision congelato", "head": "head attuale", "head_v2": "head v2",
           "frozen_tick": "congelato", "head_tick": "head\nattuale", "head_v2_tick": "head\nv2",
           "replica": "(a) contro W +\n2ª copia di W", "crossenc": "(b) contro W +\nsage/comb",
           "D": "D = g(fusione) − g(controllo)", "valid": "valid", "test": "test", "margin": "margine ±0.005",
           "nowalls-random": "stanze\ntolte", "crop": "ritaglio", "patch": "copertura",
           "media_tre_danni": "media\ndei tre", "auc": "AUC di robustezza (vision da sola)"},
    "en": {"frozen": "frozen vision", "head": "current head", "head_v2": "head v2",
           "frozen_tick": "frozen\nvision", "head_tick": "current\nhead", "head_v2_tick": "head\nv2",
           "replica": "(a) vs W +\n2nd copy of W", "crossenc": "(b) vs W +\nsage/comb",
           "D": "D = g(fusion) − g(control)", "valid": "valid", "test": "test", "margin": "margin ±0.005",
           "nowalls-random": "rooms\nremoved", "crop": "crop", "patch": "patch",
           "media_tre_danni": "mean of\nthe three", "auc": "robustness AUC (vision alone)"},
}


def read_csv(name: str) -> list[dict]:
    with (CSV_DIR / name).open(newline="") as h:
        return list(csv.DictReader(h))


def num(x) -> float:
    return float(x) if x not in ("", None) else float("nan")


def fig_ensemble(lang: str):
    t = LABELS[lang]
    rows = {(r["split"], r["vision"], r["control"]): r for r in read_csv("ensemble_D.csv")
            if r["scope"] == "media 4 repliche"}
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN, 2.7), sharey=True)
    notes = []
    for ax, ctrl in zip(axes, ("replica", "crossenc")):
        ax.axhspan(-DELTA, DELTA, color="0.88", linewidth=0, zorder=0)
        ax.axhline(0, color="0.45", linewidth=0.6, zorder=1)
        for i, v in enumerate(VISIONS):
            for j, split in enumerate(("valid", "test")):
                r = rows[(split, v, ctrl)]
                m, lo, hi = num(r["D"]), num(r["ci_lo"]), num(r["ci_hi"])
                x = i + (j - 0.5) * 0.36
                ax.bar(x, m, width=0.32, color=COLOR[v] if split == "test" else "white", edgecolor=COLOR[v],
                       linewidth=1.0, hatch=None if split == "test" else "////", zorder=2)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="0.2", elinewidth=0.8, capsize=1.6, zorder=3)
                notes.append(f"{ctrl} {v} {split}: D {m:+.4f} [{lo:+.4f}, {hi:+.4f}] · "
                             f"{r['noise_verdict']} · {r['outcome']}")
        ax.set_xticks(range(len(VISIONS)))
        ax.set_xticklabels([t[f"{v}_tick"] for v in VISIONS], fontsize=6.3)
        ax.set_title(t[ctrl], fontsize=7)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(t["D"])
    fig.legend([Patch(facecolor="white", edgecolor="0.3", hatch="////"), Patch(facecolor="0.5", edgecolor="0.3"),
                Patch(facecolor="0.88", edgecolor="none")], [t["valid"], t["test"], t["margin"]],
               loc="lower center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=0.6, rect=(0, 0.07, 1, 1))
    return fig, [CSV_DIR / "ensemble_D.csv"], notes


def fig_damage(lang: str):
    t = LABELS[lang]
    auc = {(r["split"], r["system"], r["damage"]): num(r["auc"]) for r in read_csv("vision_damage_auc.csv")}
    deltas = [r for r in read_csv("vision_damage_deltas.csv") if r["split"] == "test"]
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.4))
    notes = []
    w = 0.26
    for i, dmg in enumerate(DAMAGES):
        for j, v in enumerate(VISIONS):
            x = i + (j - 1) * w
            te, va = auc[("test", SYSTEM[v], dmg)], auc[("valid", SYSTEM[v], dmg)]
            ax.bar(x, te, width=w * 0.9, color=COLOR[v], edgecolor=COLOR[v], linewidth=0.6, zorder=2)
            ax.scatter([x], [va], marker="D", s=9, facecolors="white", edgecolors="0.15", linewidths=0.6, zorder=3)
            notes.append(f"{dmg} {v}: test {te:.4f} · valid {va:.4f}")
    for r in deltas:
        notes.append(f"test {r['comparison']} {r['damage']}: {num(r['delta']):+.4f} "
                     f"[{num(r['ci_lo']):+.4f}, {num(r['ci_hi']):+.4f}]")
    ax.axvline(len(DAMAGES) - 1.5, color="0.6", linewidth=0.6, linestyle=":")
    ax.set_xticks(range(len(DAMAGES)))
    ax.set_xticklabels([t[d] for d in DAMAGES], fontsize=6.5)
    ax.set_ylabel(t["auc"])
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    handles = [Patch(facecolor=COLOR[v], edgecolor=COLOR[v]) for v in VISIONS]
    handles += [Line2D([], [], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor="0.15",
                       markersize=3.5)]
    ax.legend(handles, [f"{t[v]} ({t['test']})" for v in VISIONS] + [t["valid"]], loc="upper center",
              bbox_to_anchor=(0.5, 1.2), ncol=4, fontsize=5.8, columnspacing=0.8, handlelength=1.2)
    return fig, [CSV_DIR / "vision_damage_auc.csv", CSV_DIR / "vision_damage_deltas.csv"], notes


FIGURES = {"ensemble": ("f_multiseed_ensemble", fig_ensemble), "damage": ("f_multiseed_damage", fig_damage)}


def main() -> None:
    p = argparse.ArgumentParser(description="Figures of the second round of RESET GRAPHS")
    p.add_argument("figure", choices=(*FIGURES, "all"))
    p.add_argument("--lang", choices=("it", "en"), default="en")
    p.add_argument("--out-dir", default="figures/multiseed/english")
    a = p.parse_args()
    out = Path(a.out_dir)
    keys = list(FIGURES) if a.figure == "all" else [a.figure]
    for key in keys:
        if (out / f"{FIGURES[key][0]}.pdf").exists():
            raise SystemExit(f"!! {out / FIGURES[key][0]}.pdf esiste gia': niente sovrascritture")
    for key in keys:
        name, fn = FIGURES[key]
        fig, sources, notes = fn(a.lang)
        save_figure(fig, name, sources, notes=notes, out_dir=out)


if __name__ == "__main__":
    main()
