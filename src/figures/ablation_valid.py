# src/figures/ablation_valid.py

"""
Figure (F6): every frozen configuration of the vision branch, grouped by
encoder — how much does the choice of ablation actually matter? (VALID)

Why it matters
--------------
The branch was chosen among ~50 combinations of encoder x pooling x
transformation. A ranking table hides the only thing worth knowing: whether the
winner is a real jump or the top of a cloud of near-identical values. A box per
encoder, with every configuration drawn as a point on top of it, answers that at
a glance — including for the encoder that won.

What is on the y axis is the metric of record (`status.md §38`): R, the mean per
query of the three self-recovery AUCs (rooms removed with their walls, crop,
scattered patches). The configuration that was frozen is marked with a star, and
the trained heads are drawn with a different marker because they are not frozen
and are not part of the same family.

Everything is read from the per-query files of wave 2 and cached next to the
figure, so a second run is instant:

    figures/<name>.data.json     R per configuration (rewrite with --refresh)

Usage (CPU; ~2 min the first time, instant afterwards):

    python -m src.figures.ablation_valid
    python -m src.figures.ablation_valid --refresh --lang en
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.robustness_auc import ROBUST_STRATEGIES, load_robust_auc
from src.figures.style import OKABE_ITO, apply_style, new_figure, save_figure

SPLIT = "valid"
DAMAGE_DIR = Path("results/perquery/vision_damage_valid_B")
CHOSEN = "vision_pespatial_gem_whiten-train"        # la config congelata (§44)

LABELS = {
    "it": {
        "y": "R — robustezza media sui tre danni",
        "x": "encoder (ordinati per mediana)",
        "chosen": "config scelta",
        "head": "head allenata",
        "frozen": "config frozen",
        "note": "{n} configurazioni frozen · {h} head allenate",
    },
    "en": {
        "y": "R — mean robustness over the three damages",
        "x": "encoder (ordered by median)",
        "chosen": "chosen config",
        "head": "trained head",
        "frozen": "frozen config",
        "note": "{n} frozen configurations · {h} trained heads",
    },
}


def discover(damage_dir: Path = DAMAGE_DIR) -> list[str]:
    """Config prefixes in a per-query folder (one per `<prefix>_partial-...` family)."""
    names = {p.name.split("_partial-")[0] for p in damage_dir.glob("*_partial-*.npz")}
    return sorted(names)


def compute(damage_dir: Path = DAMAGE_DIR, split: str = SPLIT) -> dict:
    """R per configuration, skipping the ones missing a damage or a fraction.

    Returns:
        {"r": {prefix: value}, "skipped": {prefix: reason}}
    """
    r, skipped = {}, {}
    for prefix in discover(damage_dir):
        try:
            r[prefix] = float(load_robust_auc(damage_dir / prefix, split=split,
                                              strategies=ROBUST_STRATEGIES).mean)
        except (FileNotFoundError, ValueError) as err:
            skipped[prefix] = str(err).split("\n")[0]
    if not r:
        raise ValueError(f"nessuna config completa in {damage_dir}")
    return {"r": r, "skipped": skipped, "split": split, "dir": str(damage_dir)}


def load_or_compute(cache: Path, refresh: bool) -> dict:
    """Read the cached R values, or recompute them and write the cache."""
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    data = compute()
    cache.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def split_prefix(prefix: str) -> tuple[str, str, str]:
    """`vision_<encoder>_<pooling>_<transform>` → its three parts."""
    parts = prefix.split("_", 3)
    if len(parts) < 4:
        raise ValueError(f"prefisso inatteso: {prefix}")
    return parts[1], parts[2], parts[3]


def group(values: dict) -> tuple[list[str], dict, dict]:
    """Group the configurations by encoder, frozen and head kept apart.

    Returns:
        (encoders ordered by median R of their frozen configs, frozen, heads),
        where both dicts map encoder -> [(prefix, R), ...].
    """
    frozen, heads = {}, {}
    for prefix, value in values.items():
        encoder, _, transform = split_prefix(prefix)
        target = heads if transform.startswith("head") else frozen
        target.setdefault(encoder, []).append((prefix, value))
    order = sorted(frozen, key=lambda e: np.median([v for _, v in frozen[e]]))
    return order, frozen, heads


def draw(order, frozen, heads, lang: str, height_in: float):
    """Box per encoder + one point per configuration; the chosen one is a star."""
    text = LABELS[lang]
    fig, ax = new_figure("text", height_in=height_in)
    rng = np.random.default_rng(0)          # jitter riproducibile

    data = [[v for _, v in frozen[e]] for e in order]
    box = ax.boxplot(data, positions=range(len(order)), widths=0.55, showfliers=False,
                     medianprops={"color": "0.15", "linewidth": 1.2},
                     boxprops={"color": "0.45", "linewidth": 0.8},
                     whiskerprops={"color": "0.45", "linewidth": 0.8},
                     capprops={"color": "0.45", "linewidth": 0.8},
                     patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("0.93")

    for i, encoder in enumerate(order):
        for prefix, value in frozen[encoder]:
            x = i + rng.uniform(-0.17, 0.17)
            if prefix == CHOSEN:
                ax.plot(x, value, marker="*", markersize=9, color=OKABE_ITO["green"],
                        markeredgecolor="0.15", markeredgewidth=0.5, zorder=5,
                        linestyle="none", label=text["chosen"])
            else:
                ax.plot(x, value, marker="o", markersize=3, color=OKABE_ITO["blue"],
                        alpha=0.75, linestyle="none", label=text["frozen"])
        for _, value in heads.get(encoder, []):
            ax.plot(i + rng.uniform(-0.17, 0.17), value, marker="x", markersize=4.5,
                    color=OKABE_ITO["vermillion"], linestyle="none", label=text["head"])

    # Linea al valore della config scelta: si vede subito quante altre ci arrivano
    # vicino, cioe' quanto e' (poco) speciale il primo posto.
    chosen_value = next((v for e in order for pfx, v in frozen[e] if pfx == CHOSEN), None)
    if chosen_value is not None:
        ax.axhline(chosen_value, color=OKABE_ITO["green"], linestyle=(0, (4, 3)),
                   linewidth=0.8, zorder=1)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=6.5)
    ax.set_xlabel(text["x"])
    ax.set_ylabel(text["y"])
    ax.set_xlim(-0.6, len(order) - 0.4)

    # Una voce sola per tipo di punto (il ciclo ne ha create molte).
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="lower right", labelspacing=0.3,
              handletextpad=0.4)
    n = sum(len(v) for v in frozen.values())
    ax.annotate(text["note"].format(n=n, h=sum(len(v) for v in heads.values())),
                xy=(0.01, 0.98), xycoords="axes fraction", fontsize=6, color="0.35",
                ha="left", va="top")
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--lang", choices=("it", "en"), default="it")
    p.add_argument("--refresh", action="store_true", help="ricalcola la cache dei valori R")
    p.add_argument("--height", type=float, default=2.6)
    p.add_argument("--name", default="f_ablation_valid")
    p.add_argument("--out-dir", default="figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"{args.name}.data.json"
    data = load_or_compute(cache, args.refresh)
    apply_style()
    order, frozen, heads = group(data["r"])
    fig = draw(order, frozen, heads, args.lang, args.height)

    notes = [f"split: {data['split']} · cartella: {data['dir']}",
             f"config lette: {len(data['r'])} (saltate: {len(data['skipped'])})"]
    for encoder in order:
        values = [v for _, v in frozen[encoder]]
        best = max(frozen[encoder], key=lambda kv: kv[1])
        notes.append(f"{encoder:10s} n={len(values)} mediana {np.median(values):.4f} "
                     f"min {min(values):.4f} max {max(values):.4f} · migliore: {best[0]}")
    for prefix, reason in data["skipped"].items():
        notes.append(f"saltata {prefix}: {reason}")
    for line in notes:
        print(line)
    save_figure(fig, args.name, [str(DAMAGE_DIR), str(cache)], notes=notes, out_dir=out_dir)


if __name__ == "__main__":
    main()
