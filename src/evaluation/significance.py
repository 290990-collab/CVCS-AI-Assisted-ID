"""
Paired significance test (Wilcoxon + bootstrap CI) between two per-query files.

Both systems see the same queries, so the variability compared is that of the per-query differences.
Wilcoxon signed-rank (`scipy.stats.wilcoxon`) is non-parametric; the paired percentile bootstrap
(B=10000, seed 0) resamples pairs and gives a 95% CI on the mean delta.

Constraints:
- pair on `names`, never on `qi` (galleries may number rows differently);
- keep only queries non-NaN in both files on that axis (singleton-class skip holds in AND).

Primary endpoint: nDCG@10 on the three axes with Holm correction; everything else is exploratory.

Usage:

    python -m src.evaluation.significance --a run_A.npz --b run_B.npz --k 10

Partial mode adds `--metric self_rr` (reciprocal rank of the source plan, one number per query,
no axis or k):

    python -m src.evaluation.significance --metric self_rr \
        --a <A>_partial-random-f0.75_valid.npz --b <B>_partial-random-f0.75_valid.npz

Both files must share the same `partial_label` (`check_compatible` enforces it).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.evaluation.perquery import load_perquery
from src.evaluation.relevance import AXES

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CHUNK = 500          # resample rows per block (memory)
PRIMARY_K = 10
PRIMARY_METRIC = "ndcg"
METRICS = ("ndcg", "recall", "map")

# Partial-only metric, shape [Q] (no axis, no k); kept out of METRICS, whose arrays are [axes, k, Q].
SELF_RR_METRIC = "self_rr"
CLI_METRICS = METRICS + (SELF_RR_METRIC,)

# Meta fields that must match: otherwise the files measure different things. `query_seed` and
# `num_queries` guard against differently sampled queries (name pairing would silently keep the intersection).
STRICT_META_KEYS = ("split", "exclude_self", "mode", "partial_label",
                    "geometry_weights", "query_seed", "num_queries")

# `num_queries` follows the gallery; excluded from the strict check when a gallery mismatch is allowed.
GALLERY_DEPENDENT_KEYS = ("num_queries",)

# Minimum paired fraction for a readable comparison; 0.90 tolerates the singleton-class skip
# and catches a broken name join.
DEFAULT_MIN_PAIR_FRACTION = 0.90

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    HAVE_SCIPY = True
except ImportError:                     # no scipy: degrade, do not fake
    _scipy_wilcoxon = None
    HAVE_SCIPY = False


def _normalized(key: str, value):
    """Canonical meta field: empty `partial_label` -> None, lists -> tuples."""
    if key == "partial_label" and not value:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def check_compatible(meta_a: dict, meta_b: dict, k: int, allow_gallery_mismatch: bool) -> None:
    """Raise ValueError if the two files are not comparable.

    `allow_gallery_mismatch` permits different galleries (not needed when both runs share `gallery_names`,
    see `src/evaluation/gallery_join.py`; only for older files with different gallery sizes).
    """
    for key in STRICT_META_KEYS:
        if allow_gallery_mismatch and key in GALLERY_DEPENDENT_KEYS:
            continue
        va, vb = _normalized(key, meta_a.get(key)), _normalized(key, meta_b.get(key))
        if va != vb:
            raise ValueError(
                f"meta incompatibili su '{key}': A={va!r} vs B={vb!r} — "
                "i due file non misurano la stessa cosa"
            )

    for tag, meta in (("A", meta_a), ("B", meta_b)):
        k_values = list(meta.get("k_values", []))
        if k not in k_values:
            raise ValueError(f"k={k} assente da k_values di {tag}: {k_values}")

    ga, gb = meta_a.get("gallery", {}), meta_b.get("gallery", {})
    if ga.get("sha1") != gb.get("sha1") or ga.get("n") != gb.get("n"):
        msg = (f"gallery diverse: A n={ga.get('n')} sha1={str(ga.get('sha1'))[:12]} vs "
               f"B n={gb.get('n')} sha1={str(gb.get('sha1'))[:12]}")
        if not allow_gallery_mismatch:
            raise ValueError(msg + " — usa --allow-gallery-mismatch se e' voluto")
        print(f"[significance] ATTENZIONE: {msg} (consentito esplicitamente)")


def paired_values(data_a, data_b, axis: str, metric: str, k: int):
    """Paired (a, b, names) over queries common to and non-NaN in both systems, in A's order.

    `axis` in composition|topology|geometry; `metric` in ndcg|recall|map.
    """
    row_a = getattr(data_a, metric)[data_a.axis_index(axis), data_a.k_index(k)]
    row_b = getattr(data_b, metric)[data_b.axis_index(axis), data_b.k_index(k)]

    col_b = {str(name): i for i, name in enumerate(data_b.names)}
    a_vals, b_vals, names = [], [], []
    for i, name in enumerate(data_a.names):
        j = col_b.get(str(name))
        if j is None:
            continue
        x, y = float(row_a[i]), float(row_b[j])
        if np.isnan(x) or np.isnan(y):
            continue                   # singleton skip, AND on both files
        a_vals.append(x)
        b_vals.append(y)
        names.append(str(name))

    return (np.asarray(a_vals, dtype=float),
            np.asarray(b_vals, dtype=float),
            names)


def paired_self_rr(data_a, data_b):
    """Paired `self_rr` (partial runs), same name pairing as `paired_values`.

    Raises ValueError if a file has no `self_rr`.
    """
    for tag, d in (("A", data_a), ("B", data_b)):
        if getattr(d, "self_rr", None) is None:
            raise ValueError(
                f"{tag}: il file non contiene `self_rr` (mode={d.meta.get('mode')!r}). "
                "La metrica self_rr esiste solo nelle run partial: rilancia la "
                "valutazione con partial.enabled=true e eval.perquery_dir valorizzato"
            )

    row_a, row_b = data_a.self_rr, data_b.self_rr
    col_b = {str(name): i for i, name in enumerate(data_b.names)}
    a_vals, b_vals, names = [], [], []
    for i, name in enumerate(data_a.names):
        j = col_b.get(str(name))
        if j is None:
            continue
        x, y = float(row_a[i]), float(row_b[j])
        if np.isnan(x) or np.isnan(y):
            continue
        a_vals.append(x)
        b_vals.append(y)
        names.append(str(name))

    return (np.asarray(a_vals, dtype=float),
            np.asarray(b_vals, dtype=float),
            names)


def bootstrap_ci(diff: np.ndarray, b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of the mean delta, resampling pairs with replacement (chunked, no [B, P] allocation)."""
    n = len(diff)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(b, dtype=float)
    for start in range(0, b, BOOTSTRAP_CHUNK):
        stop = min(start + BOOTSTRAP_CHUNK, b)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Paired Wilcoxon signed-rank p-value; returns (p, note), p is NaN with a note when not computable."""
    if len(a) == 0:
        return float("nan"), "nessuna coppia"
    if np.allclose(a - b, 0.0):
        return float("nan"), "differenze tutte nulle (test non definito)"
    if not HAVE_SCIPY:
        return float("nan"), "scipy non disponibile: Wilcoxon non calcolato"
    result = _scipy_wilcoxon(a, b)
    return float(result.pvalue), ""


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni correction; NaN p-values stay NaN and are not counted."""
    valid = {key: p for key, p in pvalues.items() if not np.isnan(p)}
    ordered = sorted(valid.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    for key, p in pvalues.items():
        adjusted.setdefault(key, float("nan"))
    return adjusted


def compare(data_a, data_b, axis: str, metric: str, k: int,
            min_pair_fraction: float = DEFAULT_MIN_PAIR_FRACTION) -> dict:
    """Paired comparison on (axis, metric, k) as a dict; `reliable` is False when pairs < `min_pair_fraction` of the smaller file."""
    a, b, names = paired_values(data_a, data_b, axis, metric, k)
    diff = a - b
    lo, hi = bootstrap_ci(diff)
    p, note = wilcoxon_p(a, b)
    n_ref = min(len(data_a.names), len(data_b.names))
    fraction = len(names) / n_ref if n_ref else 0.0
    return {
        "axis": axis, "metric": metric, "k": k,
        "n_pairs": len(names), "n_ref": n_ref,
        "pair_fraction": fraction,
        "reliable": fraction >= min_pair_fraction,
        "min_pair_fraction": min_pair_fraction,
        "mean_a": float(a.mean()) if len(a) else float("nan"),
        "mean_b": float(b.mean()) if len(b) else float("nan"),
        "delta": float(diff.mean()) if len(diff) else float("nan"),
        "ci_lo": lo, "ci_hi": hi,
        "p": p, "p_note": note,
    }


def compare_self_rr(data_a, data_b,
                    min_pair_fraction: float = DEFAULT_MIN_PAIR_FRACTION) -> dict:
    """Paired comparison on `self_rr`; same dict as `compare` (axis "self-recovery", k "-")."""
    a, b, names = paired_self_rr(data_a, data_b)
    diff = a - b
    lo, hi = bootstrap_ci(diff)
    p, note = wilcoxon_p(a, b)
    n_ref = min(len(data_a.names), len(data_b.names))
    fraction = len(names) / n_ref if n_ref else 0.0
    return {
        "axis": "self-recovery", "metric": SELF_RR_METRIC, "k": "-",
        "n_pairs": len(names), "n_ref": n_ref,
        "pair_fraction": fraction,
        "reliable": fraction >= min_pair_fraction,
        "min_pair_fraction": min_pair_fraction,
        "mean_a": float(a.mean()) if len(a) else float("nan"),
        "mean_b": float(b.mean()) if len(b) else float("nan"),
        "delta": float(diff.mean()) if len(diff) else float("nan"),
        "ci_lo": lo, "ci_hi": hi,
        "p": p, "p_note": note,
    }


def print_rows(rows: list[dict], title: str, adjusted: dict[str, float] | None = None) -> None:
    """Comparison table; `adjusted` adds the Holm column. Unreliable rows are flagged `!` and repeated at the end."""
    print(f"=== {title} ===")
    header = (f"{'':<1} {'asse':<13} {'metrica':<7} {'k':>4} {'n':>6} {'A':>8} {'B':>8} "
              f"{'delta':>9} {'CI 95%':>19} {'p':>10}")
    if adjusted is not None:
        header += f" {'p Holm':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (f"{'!' if not r['reliable'] else ' ':<1} "
                f"{r['axis']:<13} {r['metric']:<7} {r['k']:>4} {r['n_pairs']:>6} "
                f"{r['mean_a']:>8.4f} {r['mean_b']:>8.4f} {r['delta']:>+9.4f} "
                f"[{r['ci_lo']:>+8.4f},{r['ci_hi']:>+8.4f}] {r['p']:>10.2e}")
        if adjusted is not None:
            line += f" {adjusted.get(r['axis'], float('nan')):>10.2e}"
        print(line)
        if r["p_note"]:
            print(f"{'':<15} └─ Wilcoxon: {r['p_note']}")
    for r in rows:
        if not r["reliable"]:
            print(f"! {r['axis']} {r['metric']}@{r['k']}: solo {r['n_pairs']}/{r['n_ref']} "
                  f"query appaiate ({100 * r['pair_fraction']:.1f}% < "
                  f"{100 * r['min_pair_fraction']:.0f}%) — risultato NON AFFIDABILE, "
                  "le due medie non sono sullo stesso insieme di query")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test appaiato (Wilcoxon + bootstrap) fra due file per-query."
    )
    p.add_argument("--a", required=True, help="file per-query .npz del sistema A")
    p.add_argument("--b", required=True, help="file per-query .npz del sistema B")
    p.add_argument("--axis", nargs="+", default=list(AXES), choices=list(AXES),
                   help="assi da confrontare (default: tutti)")
    p.add_argument("--metric", default=PRIMARY_METRIC, choices=list(CLI_METRICS),
                   help="ndcg|recall|map sono per-asse; `self_rr` (self-recovery, "
                        "solo run partial) ignora --axis e --k: e' un numero per "
                        "query. E' l'endpoint del criterio A.5 (status.md § 23)")
    p.add_argument("--k", type=int, default=PRIMARY_K)
    p.add_argument("--allow-gallery-mismatch", action="store_true",
                   dest="allow_gallery_mismatch",
                   help="consente gallery diverse (serve solo a vision vs graph)")
    p.add_argument("--min-pair-fraction", type=float, default=DEFAULT_MIN_PAIR_FRACTION,
                   dest="min_pair_fraction",
                   help="frazione minima di query appaiate perche' il confronto sia "
                        f"affidabile (default {DEFAULT_MIN_PAIR_FRACTION})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_a, data_b = load_perquery(args.a), load_perquery(args.b)
    check_compatible(data_a.meta, data_b.meta, args.k, args.allow_gallery_mismatch)

    print(f"[significance] A = {Path(args.a).name} "
          f"(run_tag {data_a.meta.get('run_tag')}, Q={len(data_a.names)})")
    print(f"[significance] B = {Path(args.b).name} "
          f"(run_tag {data_b.meta.get('run_tag')}, Q={len(data_b.names)})")
    print(f"[significance] bootstrap B={BOOTSTRAP_B} seed={BOOTSTRAP_SEED}, "
          f"Wilcoxon {'scipy' if HAVE_SCIPY else 'NON disponibile'}, "
          f"copertura minima {100 * args.min_pair_fraction:.0f}% delle query\n")

    # `self_rr` has no axis: single test, no Holm
    if args.metric == SELF_RR_METRIC:
        row = compare_self_rr(data_a, data_b, args.min_pair_fraction)
        print_rows([row], "self-recovery (MRR) — endpoint del criterio A.5 "
                          "«migliore = piu' robusto», test singolo")
        print(f"masking: A partial_label={data_a.meta.get('partial_label')!r} · "
              f"B partial_label={data_b.meta.get('partial_label')!r} "
              "(devono coincidere: lo garantisce check_compatible)")
        print("delta = media(A) - media(B); CI percentile appaiato sulle differenze.")
        print("NB: le metriche per-asse non sono stampate qui: sono un'altra "
              "domanda (--metric ndcg). Dal 25 ago (B.4) i file partial nuovi le "
              "hanno con exclude_self=True, quindi confrontabili col full; i file "
              "piu' vecchi hanno False e vengono rifiutati dal check di compatibilita'.")
        return

    requested = [compare(data_a, data_b, ax, args.metric, args.k, args.min_pair_fraction)
                 for ax in args.axis]

    # primary endpoint: 3 tests, Holm
    is_primary = (args.metric == PRIMARY_METRIC and args.k == PRIMARY_K
                  and set(args.axis) == set(AXES))
    if is_primary:
        adjusted = holm({r["axis"]: r["p"] for r in requested})
        print_rows(requested, f"endpoint primario: {PRIMARY_METRIC}@{PRIMARY_K}, "
                              "3 assi, p corretti con Holm", adjusted)
    else:
        print_rows(requested, f"{args.metric}@{args.k} — ESPLORATIVO "
                              "(nessuna correzione per test multipli)")
        primary_ok = all(PRIMARY_K in list(d.meta.get("k_values", []))
                         for d in (data_a, data_b))
        if primary_ok:
            primary = [compare(data_a, data_b, ax, PRIMARY_METRIC, PRIMARY_K,
                               args.min_pair_fraction) for ax in AXES]
            adjusted = holm({r["axis"]: r["p"] for r in primary})
            print_rows(primary, f"endpoint primario: {PRIMARY_METRIC}@{PRIMARY_K}, "
                                "3 assi, p corretti con Holm", adjusted)
        else:
            print(f"[significance] endpoint primario non calcolabile: "
                  f"k={PRIMARY_K} assente dai k_values\n")

    print("delta = media(A) - media(B); CI percentile appaiato sulle differenze.")


if __name__ == "__main__":
    main()
