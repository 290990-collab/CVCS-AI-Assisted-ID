"""Sensitivity of the system ranking on the geometry axis to the weights of its components.

`GalleryAxes.geometry_sim` averages area, aspect ratio and per-room-type area distribution with
weights (1,1,1). Weights enter the gain (ground truth), not the model ranking, so nDCG is recomputed
post-hoc from the saved `ret_rows` (no retrieval, no GPU). The 0.5 in `dist_sim` is not a weight.

Claim "A beats B on geometry" holds if the paired delta has the same sign in all weightings and its
95% CI excludes zero; REGGE / NON REGGE (significant inversion) / INCONCLUSIVO otherwise.
Read within a row only (each weighting is a different GT). `dist_sim` is a function of `rType` +
`boxes` (the graph node features): (0,0,1) is the most circular for the graph branch, (1,1,0) the least.

Usage:

    python -m src.evaluation.geometry_variants \
        --gallery embeddings/<...>/image_paths.json \
        --perquery run_A.npz run_B.npz --k 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.rplan_metadata import load_metadata
from src.evaluation.metrics import ndcg_at_k
from src.evaluation.perquery import load_perquery
from src.evaluation.relevance import GalleryAxes
from src.evaluation.significance import BOOTSTRAP_B, BOOTSTRAP_SEED, bootstrap_ci

# baseline, isolated components, area+aspect (pure shape), doubled distribution
DEFAULT_WEIGHTS = (
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 1.0, 2.0),
)
BASELINE_WEIGHTS = (1.0, 1.0, 1.0)
DEFAULT_K = 10


# --- weighted geometry gain ---

def geometry_sim_weighted(axes: GalleryAxes, qi: int, w) -> np.ndarray:
    """`GalleryAxes.geometry_sim` with weights w=(w_area, w_aspect, w_dist), normalised to [0,1].

    Bitwise identical to the original for w=(1,1,1). Returns float32 [N] (self included, no-.mat plans 0).
    """
    w_area, w_aspect, w_dist = (float(x) for x in w)
    total = w_area + w_aspect + w_dist
    if total <= 0:
        raise ValueError(f"pesi geometria non validi (somma {total}): {w}")

    area_sim = 1.0 - np.abs(axes.area - axes.area[qi])
    aq = axes.aspect[qi]
    aspect_sim = np.minimum(axes.aspect, aq) / np.maximum(axes.aspect, aq)
    dist_sim = 1.0 - 0.5 * np.abs(axes.dist - axes.dist[qi]).sum(axis=1)

    sim = (w_area * area_sim + w_aspect * aspect_sim + w_dist * dist_sim) / total
    sim[~axes.valid] = 0.0
    return sim.astype(np.float32)


# --- nDCG recomputation from a per-query file ---

def perquery_ndcg_geometry(axes: GalleryAxes, data, w, k: int):
    """Per-query geometry nDCG@k recomputed with weights `w` from the saved `ret_rows`.

    Returns (names, values float64 [Q]).
    """
    if data.ret_rows is None:
        raise ValueError("il file per-query non contiene ret_rows: ricalcolo impossibile")

    exclude_self = bool(data.meta.get("exclude_self", True))
    values = []
    for q, qi in enumerate(np.asarray(data.qi, dtype=int)):
        sims = geometry_sim_weighted(axes, int(qi), w)
        not_self = np.ones(len(axes), dtype=bool)
        if exclude_self:
            not_self[qi] = False
        rows = np.asarray(data.ret_rows[q][:int(data.n_ret[q])], dtype=int)
        retrieved_gains = sims[rows] if len(rows) else np.empty(0)
        values.append(ndcg_at_k(retrieved_gains, sims[not_self], k))
    return [str(n) for n in data.names], np.asarray(values, dtype=float)


def ndcg_geometry_weighted(axes: GalleryAxes, data, w, k: int) -> float:
    """Mean of `perquery_ndcg_geometry`."""
    _, values = perquery_ndcg_geometry(axes, data, w, k)
    return float(values.mean()) if len(values) else float("nan")


def paired_delta(names_a, values_a, names_b, values_b) -> dict:
    """Delta paired by name (not position) between two systems + 95% bootstrap CI (`significance.bootstrap_ci`)."""
    col_b = {name: i for i, name in enumerate(names_b)}
    pairs = [(values_a[i], values_b[col_b[name]])
             for i, name in enumerate(names_a) if name in col_b]
    if not pairs:
        return {"n_pairs": 0, "mean_a": float("nan"), "mean_b": float("nan"),
                "delta": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}

    a = np.asarray([p[0] for p in pairs], dtype=float)
    b = np.asarray([p[1] for p in pairs], dtype=float)
    lo, hi = bootstrap_ci(a - b)
    return {"n_pairs": len(pairs), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "delta": float((a - b).mean()), "ci_lo": lo, "ci_hi": hi}


def stored_geometry_ndcg(data, k: int) -> float:
    """Stored geometry nDCG@k (consistency check for the recomputation)."""
    row = data.ndcg[data.axis_index("geometry"), data.k_index(k)]
    return float(np.nanmean(row))


def check_gallery(axes: GalleryAxes, data, label: str) -> None:
    """Raise if the per-query file belongs to a different gallery (`ret_rows` are row indices)."""
    n_meta = data.meta.get("gallery", {}).get("n")
    if n_meta is not None and int(n_meta) != len(axes):
        raise ValueError(
            f"{label}: gallery del file ({n_meta} righe) diversa da quella caricata "
            f"({len(axes)}): i ret_rows non sono interpretabili"
        )


# --- weighting x system table + claim verdict ---

def print_table(results: dict, systems: list[str], k: int) -> None:
    """Weighting x system table of geometry nDCG@k."""
    print(f"=== nDCG@{k} geometria, ricalcolato per pesatura (area, aspect, dist) ===")
    header = f"{'pesatura':<16}" + "".join(f"{s:>16}" for s in systems)
    print(header)
    print("-" * len(header))
    for w, scores in results.items():
        label = "(" + ",".join(f"{x:g}" for x in w) + ")"
        print(f"{label:<16}" + "".join(f"{scores[s]:>16.4f}" for s in systems))
    print()


def claim_verdict(rows: list[dict]) -> str:
    """Verdict on "A beats B on geometry".

    REGGE: same delta sign in all weightings, CI excludes 0. NON REGGE: some weighting has the
    opposite sign with CI excluding 0. INCONCLUSIVO: otherwise.
    """
    excludes_zero = [r["ci_lo"] > 0 or r["ci_hi"] < 0 for r in rows]
    signs = [np.sign(r["delta"]) for r in rows]
    reference = signs[0]

    if any(e and s != reference for e, s in zip(excludes_zero, signs)):
        return "NON REGGE"
    if all(excludes_zero) and len(set(signs)) == 1 and reference != 0:
        return "REGGE"
    return "INCONCLUSIVO"


def print_paired_table(rows: list[dict], system_a: str, system_b: str, k: int) -> None:
    """Paired delta + 95% CI per weighting, and the verdict."""
    print(f"=== delta appaiato nDCG@{k} geometria: {system_a} - {system_b} ===")
    print(f"criterio (fissato prima dei numeri): il claim regge se in TUTTE le "
          f"{len(rows)} pesature\nil segno del delta e' invariato E il CI 95% esclude lo zero.")
    header = (f"{'pesatura':<16} {'n':>6} {system_a[:10]:>10} {system_b[:10]:>10} "
              f"{'delta':>9} {'CI 95%':>21}")
    print(header)
    print("-" * len(header))
    for r in rows:
        label = "(" + ",".join(f"{x:g}" for x in r["weights"]) + ")"
        flag = "" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "   <- CI contiene 0"
        print(f"{label:<16} {r['n_pairs']:>6} {r['mean_a']:>10.4f} {r['mean_b']:>10.4f} "
              f"{r['delta']:>+9.4f} [{r['ci_lo']:>+9.4f},{r['ci_hi']:>+9.4f}]{flag}")
    print(f"\nVERDETTO: {claim_verdict(rows)}  "
          f"(bootstrap appaiato B={BOOTSTRAP_B}, seed {BOOTSTRAP_SEED})\n")


# --- entrypoint ---

def parse_weights(values: list[str] | None):
    """Parse CLI weightings, each 'wa,wb,wc'."""
    if not values:
        return list(DEFAULT_WEIGHTS)
    parsed = []
    for item in values:
        parts = item.split(",")
        if len(parts) != 3:
            raise ValueError(f"pesatura '{item}': attesi 3 valori separati da virgola")
        parsed.append(tuple(float(x) for x in parts))
    return parsed


def load_gallery_axes(gallery_path: str) -> GalleryAxes:
    """Per-axis features from the `.mat`, aligned to the gallery JSON rows."""
    entries = json.loads(Path(gallery_path).read_text())
    print(f"[geometry_variants] gallery: {len(entries)} righe ({gallery_path})")
    axes = GalleryAxes([load_metadata(e) for e in entries])
    print(f"[geometry_variants] metadati .mat presenti: {int(axes.valid.sum())}/{len(entries)}")
    return axes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sensibilita' dell'asse geometria ai pesi delle sue componenti."
    )
    p.add_argument("--gallery", required=True,
                   help="image_paths.json (vision) o names.json (graph) della gallery")
    p.add_argument("--gallery-b", default=None, dest="gallery_b",
                   help="gallery del SECONDO --perquery, per il confronto CROSS-RAMO "
                        "(vision 67.453 vs graph 67.405). Richiede esattamente due "
                        "--perquery. ATTENZIONE: l'IDCG viene da gallery diverse, "
                        "quindi il confronto e' appaiato sui nomi ma non perfettamente "
                        "equo finche' non c'e' la gallery comune (fase B.3).")
    p.add_argument("--perquery", nargs="+", required=True,
                   help="uno o piu' .npz per-query (un sistema ciascuno)")
    p.add_argument("--label", nargs="+", default=None,
                   help="etichette dei sistemi (default: run_tag del meta)")
    p.add_argument("--weights", nargs="+", default=None,
                   help="pesature 'wa,wb,wc' (default: 6 pesature esplorative)")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.label and len(args.label) != len(args.perquery):
        raise ValueError("--label deve avere tanti valori quanti --perquery")

    weights = parse_weights(args.weights)
    axes = load_gallery_axes(args.gallery)

    axes_b = None
    if args.gallery_b:
        if len(args.perquery) != 2:
            raise ValueError("--gallery-b richiede esattamente due --perquery")
        axes_b = load_gallery_axes(args.gallery_b)
        print("[geometry_variants] CROSS-RAMO: A e B hanno gallery diverse. L'IDCG di "
              "ciascun sistema e' calcolato sulla PROPRIA gallery; il delta resta "
              "appaiato sui nomi, ma il confronto non e' perfettamente equo (fase B.3).")

    systems, data_by_system, axes_by_system = [], {}, {}
    for i, path in enumerate(args.perquery):
        data = load_perquery(path)
        ax = axes_b if (axes_b is not None and i == 1) else axes
        check_gallery(ax, data, Path(path).name)
        label = args.label[i] if args.label else str(data.meta.get("run_tag", Path(path).stem))
        systems.append(label)
        data_by_system[label] = data
        axes_by_system[label] = ax

    # with baseline weights the recomputation must match the stored value
    for label, data in data_by_system.items():
        if args.k in list(data.meta.get("k_values", [])):
            recomputed = ndcg_geometry_weighted(
                axes_by_system[label], data, BASELINE_WEIGHTS, args.k)
            stored = stored_geometry_ndcg(data, args.k)
            print(f"[geometry_variants] {label}: ricalcolo (1,1,1) {recomputed:.6f} "
                  f"vs salvato {stored:.6f} (delta {recomputed - stored:+.2e})")
    print()

    # one recomputation per (weighting, system), kept per query
    per_query = {}
    results = {}
    for w in weights:
        key = tuple(w)
        results[key] = {}
        for label, data in data_by_system.items():
            names, values = perquery_ndcg_geometry(axes_by_system[label], data, w, args.k)
            per_query[(key, label)] = (names, values)
            results[key][label] = float(values.mean()) if len(values) else float("nan")

    print_table(results, systems, args.k)

    # the verdict compares exactly two systems
    if len(systems) == 2:
        a_label, b_label = systems
        rows = []
        for w in weights:
            key = tuple(w)
            names_a, values_a = per_query[(key, a_label)]
            names_b, values_b = per_query[(key, b_label)]
            rows.append({"weights": key,
                         **paired_delta(names_a, values_a, names_b, values_b)})
        print_paired_table(rows, a_label, b_label, args.k)
    else:
        print(f"[geometry_variants] {len(systems)} sistemi: il criterio di decisione "
              "(delta appaiato + CI) vale su una COPPIA.\n"
              "Rilancia con due soli --perquery per avere il verdetto; la tabella "
              "sopra e' solo descrittiva.\n")

    print("I pesi cambiano il GAIN (il ground truth), non il ranking dei modelli: "
          "l'analisi e' post-hoc sui ret_rows salvati.")
    print("Righe NON confrontabili fra loro (ogni pesatura e' una GT diversa); "
          "(0,0,1) e' la pesatura piu' circolare per il ramo graph, (1,1,0) la meno.")


if __name__ == "__main__":
    main()
