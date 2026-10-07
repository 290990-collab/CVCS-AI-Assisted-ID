"""Diagnostics of the metrics themselves: what the reported numbers actually measure.

Post-hoc (no GPU, no retrieval): questions 1-2 read the per-query files (`perquery/1`), question 3
rebuilds the relevance features from the `.mat`.

1. Is "Recall" recall? `recall_at_k` divides by `min(k, #relevant)`; with at least `k` relevant it is
   Precision@k. `precision_regime` gives the fraction of queries in that regime, per axis and k.
2. Over how many queries is the mean taken? Queries with a singleton equivalence class have no relevant
   items (`num_relevant == 0`) and leave the Recall/mAP mean: the residual mean describes a subset chosen by
   the ground truth (absolute value optimistic; report coverage next to it).
3. Are the three axes independent? `axis_correlation` is the Pearson correlation of the gains on the
   same pairs; if high, "wins on 2 axes out of 3" is not two independent pieces of evidence.

Usage:

    # 1 and 2: from per-query files on disk
    python -m src.evaluation.metric_diagnostics \
        --perquery results/perquery/vision_valid/*.npz --k 10

    # 3: from the gallery (estimate on a sample of pairs)
    python -m src.evaluation.metric_diagnostics \
        --gallery embeddings/vision/dinov2/natural/image_paths.json \
        --corr-queries 60 --corr-gallery 6000

Question 3 computes axis similarities over the whole sampled sub-gallery per query; the defaults fit the
login node `ulimit -t` (600 s CPU).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.data.rplan_metadata import load_metadata
from src.evaluation.perquery import load_perquery
from src.evaluation.relevance import AXES, DISCRETE_AXES, GalleryAxes

DEFAULT_K = 10
DEFAULT_CORR_QUERIES = 60
DEFAULT_CORR_GALLERY = 6000
CORR_SEED = 0

# special `num_relevant` values in the perquery/1 contract
NO_RELEVANT_SET = -1    # continuous axis: no equivalence class
SKIPPED = 0             # singleton class: query excluded from Recall/mAP


# --- 1 + 2: diagnostics from per-query files ---

def axis_diagnostics(data, axis: str, k: int) -> dict | None:
    """Metric regime and mean coverage for a discrete axis (axis in DISCRETE_AXES).

    Returns a dict (n_query, n_skipped, coverage, n_prec = queries with #relevant >= k,
    precision_regime = fraction of scored queries in that regime, median_relevant, p90_relevant),
    or None if the file has no relevant set for the axis.
    """
    a = data.axis_index(axis)
    num_rel = np.asarray(data.num_relevant[a], dtype=np.int64)
    if num_rel.size == 0 or np.all(num_rel == NO_RELEVANT_SET):
        return None

    n_query = int(num_rel.size)
    n_skipped = int((num_rel == SKIPPED).sum())
    scored = num_rel[num_rel > SKIPPED]
    n_scored = int(scored.size)
    n_prec = int((scored >= k).sum())

    return {
        "n_query": n_query,
        "n_skipped": n_skipped,
        "coverage": n_scored / n_query if n_query else float("nan"),
        "n_prec": n_prec,
        "precision_regime": n_prec / n_scored if n_scored else float("nan"),
        "median_relevant": float(np.median(scored)) if n_scored else float("nan"),
        "p90_relevant": float(np.percentile(scored, 90)) if n_scored else float("nan"),
    }


def run_diagnostics(data, k: int) -> dict[str, dict]:
    """`axis_diagnostics` on every discrete axis in the file."""
    out = {}
    for axis in DISCRETE_AXES:
        try:
            d = axis_diagnostics(data, axis, k)
        except KeyError:              # axis absent from the file
            continue
        if d is not None:
            out[axis] = d
    return out


def print_diagnostics_table(rows: list[tuple[str, dict[str, dict]]], k: int) -> None:
    """Run x axis table: mean coverage and metric regime; rows = (run label, `run_diagnostics` result)."""
    print(f"=== Diagnostica delle metriche a k={k} ===")
    print("copertura = query su cui Recall/mAP sono definiti (le altre sono")
    print("            singleton ed escono dalla media)")
    print(f"prec@{k}   = frazione delle query valutate con #rilevanti >= {k}:")
    print(f"            lì min(k,#rel)=k e 'Recall@{k}' È Precision@{k}\n")

    header = (f"{'run':<34} {'asse':<12} {'copertura':>10} {'escluse':>9} "
              f"{f'prec@{k}':>9} {'#rel med':>9} {'#rel p90':>9}")
    print(header)
    print("-" * len(header))
    for label, res in rows:
        for axis, d in res.items():
            print(f"{label:<34} {axis:<12} "
                  f"{100*d['coverage']:>9.1f}% {d['n_skipped']:>9d} "
                  f"{100*d['precision_regime']:>8.1f}% "
                  f"{d['median_relevant']:>9.0f} {d['p90_relevant']:>9.0f}")
    print()


# --- 3: correlation between relevance axes ---

def axis_correlation(
    axes: GalleryAxes,
    num_queries: int = DEFAULT_CORR_QUERIES,
    seed: int = CORR_SEED,
) -> tuple[np.ndarray, dict[str, dict], int]:
    """Pearson correlation of the three axis gains over the same (query, row) pairs, self excluded.

    `axes` may be a sub-sample of the gallery. Returns (corr [3,3] in AXES order,
    per-axis stats mean/std/p1/p99, n_pairs).
    """
    rng = random.Random(seed)
    valid_rows = [i for i in range(len(axes)) if axes.valid[i]]
    if not valid_rows:
        raise ValueError("nessuna riga con metadati .mat: correlazione non calcolabile")
    qs = rng.sample(valid_rows, min(num_queries, len(valid_rows)))

    columns: dict[str, list[np.ndarray]] = {ax: [] for ax in AXES}
    for qi in qs:
        not_self = np.ones(len(axes), dtype=bool)
        not_self[qi] = False
        for ax in AXES:
            columns[ax].append(axes.sim(ax, qi)[not_self])

    stacked = {ax: np.concatenate(v) for ax, v in columns.items()}
    matrix = np.vstack([stacked[ax] for ax in AXES])
    corr = np.corrcoef(matrix)

    stats = {
        ax: {
            "mean": float(stacked[ax].mean()),
            "std": float(stacked[ax].std()),
            "p1": float(np.percentile(stacked[ax], 1)),
            "p99": float(np.percentile(stacked[ax], 99)),
        }
        for ax in AXES
    }
    return corr, stats, int(matrix.shape[1])


def print_correlation(corr: np.ndarray, stats: dict, n_pairs: int, n_gallery: int) -> None:
    """Correlation matrix + marginals, with the reading."""
    print(f"=== Correlazione fra i gain dei tre assi ({n_pairs} coppie, "
          f"gallery di {n_gallery} righe) ===\n")

    header = f"{'':<13}" + "".join(f"{ax:>13}" for ax in AXES)
    print(header)
    print("-" * len(header))
    for i, ax in enumerate(AXES):
        print(f"{ax:<13}" + "".join(f"{corr[i, j]:>13.3f}" for j in range(len(AXES))))
    print()

    header2 = f"{'asse':<13} {'media':>9} {'std':>9} {'p1':>9} {'p99':>9} {'p1..p99':>9}"
    print(header2)
    print("-" * len(header2))
    for ax in AXES:
        s = stats[ax]
        print(f"{ax:<13} {s['mean']:>9.3f} {s['std']:>9.3f} "
              f"{s['p1']:>9.3f} {s['p99']:>9.3f} {s['p99']-s['p1']:>9.3f}")
    print()
    print("Lettura: correlazioni alte -> i tre numeri della tabella NON sono tre")
    print("evidenze indipendenti ('vince su 2 assi su 3' non è un'affermazione")
    print("forte). Un intervallo p1..p99 stretto -> asse saturo: il gain usa solo")
    print("una parte della scala [0,1] e anche un ranking cieco prende molto.")
    print()


# --- CLI ---

def build_gallery_axes(gallery_path: str, sample: int, seed: int = CORR_SEED):
    """`GalleryAxes` from an `image_paths.json`/`names.json`, optionally on a row sample; returns (axes, n_rows)."""
    entries = json.loads(Path(gallery_path).read_text())
    if sample and sample < len(entries):
        rng = random.Random(seed)
        entries = [entries[i] for i in rng.sample(range(len(entries)), sample)]
    print(f"[diagnostics] costruzione feature per-asse su {len(entries)} righe...")
    axes = GalleryAxes([load_metadata(e) for e in entries])
    print(f"[diagnostics] metadati .mat presenti: {int(axes.valid.sum())}/{len(entries)}\n")
    return axes, len(entries)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnostica delle metriche: regime, copertura, indipendenza degli assi."
    )
    p.add_argument("--perquery", nargs="*", default=[],
                   help="file .npz per-query da diagnosticare (punti 1 e 2)")
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help=f"profondità per il regime precision (default {DEFAULT_K})")
    p.add_argument("--gallery", default=None,
                   help="image_paths.json/names.json per la correlazione fra assi (punto 3)")
    p.add_argument("--corr-queries", type=int, default=DEFAULT_CORR_QUERIES,
                   help=f"query campionate per la correlazione (default {DEFAULT_CORR_QUERIES})")
    p.add_argument("--corr-gallery", type=int, default=DEFAULT_CORR_GALLERY,
                   help=f"righe di gallery campionate (default {DEFAULT_CORR_GALLERY}, "
                        "0 = intera: pesante)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.perquery and not args.gallery:
        raise SystemExit("niente da fare: passa --perquery e/o --gallery")

    if args.perquery:
        rows = []
        for path in sorted(args.perquery):
            data = load_perquery(path)
            rows.append((Path(path).stem, run_diagnostics(data, args.k)))
        print_diagnostics_table(rows, args.k)

    if args.gallery:
        axes, n_rows = build_gallery_axes(args.gallery, args.corr_gallery)
        corr, stats, n_pairs = axis_correlation(axes, args.corr_queries)
        print_correlation(corr, stats, n_pairs, n_rows)


if __name__ == "__main__":
    main()
