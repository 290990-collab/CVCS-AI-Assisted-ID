# src/evaluation/significance.py

"""
Il delta fra due sistemi e' reale o e' rumore di campionamento?

Con 2000 query, una differenza di 0.003 di nDCG puo' benissimo essere
l'ondeggiare del campione. Questo modulo risponde con due strumenti appaiati
(query per query), che e' l'unico modo corretto: i due sistemi vedono le STESSE
query, quindi la variabilita' da confrontare e' quella delle *differenze*, non
quella delle due medie.

- **Wilcoxon signed-rank** (`scipy.stats.wilcoxon`): test non parametrico sui
  ranghi delle differenze. Non assume normalita' — le distribuzioni per-query di
  nDCG sono tutt'altro che gaussiane (tetti a 1.0, code lunghe).
- **Bootstrap percentile appaiato** (B=10000, seed 0): ricampiona le *coppie* e
  da' un intervallo di confidenza al 95% sul delta medio. Il p-value dice "e'
  diverso da zero?", il CI dice "di quanto, al netto del rumore" — nel report
  serve il secondo.

Due regole non negoziabili:

- l'appaiamento e' **sui `names`**, mai sui `qi`: due gallery diverse hanno
  numerazioni di riga diverse (vision 67.453 righe, graph 67.405), e appaiare
  per indice significherebbe confrontare piante diverse;
- si tengono solo le query non-NaN in **entrambi** i file su quell'asse: lo skip
  delle classi singleton deve valere in AND, altrimenti i due sistemi non
  vengono mediati sullo stesso insieme.

Endpoint **primario**: nDCG@10 sui tre assi, con correzione di **Holm** (tre
test sulla stessa domanda -> il p-value grezzo sottostima il rischio di falso
positivo). Tutto il resto e' esplorativo e va dichiarato tale.

Uso tipico:

    python -m src.evaluation.significance --a run_A.npz --b run_B.npz --k 10

Nel **partial** c'e' un endpoint in piu': `--metric self_rr`, il reciprocal rank
della pianta sorgente. E' un numero per query (nessun asse, nessun k) ed e' la
misura su cui si decide «migliore = piu' robusto» (criterio A.5, status.md § 23):

    python -m src.evaluation.significance --metric self_rr \
        --a <A>_partial-random-f0.75_valid.npz --b <B>_partial-random-f0.75_valid.npz

I due file devono avere lo **stesso** `partial_label` (stessa strategia e stessa
frazione mascherata): lo impone `check_compatible`, perche' confrontare due
livelli di masking diversi non misura la robustezza ma il livello.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.evaluation.perquery import load_perquery
from src.evaluation.relevance import AXES

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CHUNK = 500          # righe di resample per blocco: tiene la memoria bassa
PRIMARY_K = 10
PRIMARY_METRIC = "ndcg"
METRICS = ("ndcg", "recall", "map")

# Metrica del solo PARTIAL, con una forma diversa da tutte le altre: `self_rr`
# e' [Q] — un numero per query, senza asse e senza k — perche' il self-recovery
# ("ho ritrovato la pianta sorgente?") non si declina per asse. Non entra in
# METRICS: gli array di quelle sono [assi, k, Q] e il codice che le legge
# indicizza per (asse, k). Va quindi trattata a parte, non aggiunta a un elenco.
# E' l'endpoint del criterio A.5 «migliore = piu' robusto» (status.md § 23).
SELF_RR_METRIC = "self_rr"
CLI_METRICS = METRICS + (SELF_RR_METRIC,)

# Campi del meta che DEVONO coincidere: se differiscono i due file non misurano
# la stessa cosa e il confronto e' privo di senso (non e' una questione di
# rumore, e' un errore di protocollo). `query_seed` e `num_queries` sono qui
# perche' due run campionate con seed diversi hanno query diverse: le medie non
# sono quelle pubblicate e l'appaiamento per nome terrebbe solo l'intersezione,
# in silenzio.
STRICT_META_KEYS = ("split", "exclude_self", "mode", "partial_label",
                    "geometry_weights", "query_seed", "num_queries")

# `num_queries` dipende meccanicamente dalla gallery (pool di query diverso):
# quando la differenza di gallery e' dichiarata col flag, questo campo non e'
# piu' un indizio di errore di protocollo e va escluso dal confronto stretto.
GALLERY_DEPENDENT_KEYS = ("num_queries",)

# Frazione minima di query appaiate perche' il confronto sia leggibile: sotto
# questa soglia le due medie non sono piu' calcolate sullo stesso insieme e il
# delta mescola "differenza fra sistemi" e "differenza fra campioni". 0.90 e'
# permissivo quanto basta per lo skip delle classi singleton (che toglie
# tipicamente pochi punti percentuali) e stretto abbastanza da intercettare un
# join andato male sui nomi.
DEFAULT_MIN_PAIR_FRACTION = 0.90

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    HAVE_SCIPY = True
except ImportError:                     # env senza scipy: si degrada, non si finge
    _scipy_wilcoxon = None
    HAVE_SCIPY = False


# ----------------------------------------------------------------------
# Compatibilita' dei due file.
# ----------------------------------------------------------------------

def _normalized(key: str, value):
    """Forma canonica di un campo del meta, per confronti robusti.

    `partial_label` None e "" significano la stessa cosa (modalita' full) e le
    liste/tuple di `geometry_weights` vanno confrontate per valore.
    """
    if key == "partial_label" and not value:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def check_compatible(meta_a: dict, meta_b: dict, k: int, allow_gallery_mismatch: bool) -> None:
    """Errore duro se i due file non sono confrontabili.

    Args:
        meta_a, meta_b: i `meta` dei due `.npz`.
        k: profondita' richiesta, deve esistere in entrambi i `k_values`.
        allow_gallery_mismatch: unico modo per confrontare gallery diverse.
            ⚠️ Dal 25 ago 2026 (**B.3**) vision vs graph NON ne ha piu' bisogno,
            se entrambe le run usano lo stesso `gallery_names` (inner join,
            `src/evaluation/gallery_join.py`): stessa lista, stesso ordine,
            stesso sha1. Il flag resta per i confronti con i file piu' vecchi,
            dove le due gallery avevano taglie diverse (67.453 vs 67.405) e i due
            IDCG non erano identici — li' il confronto va dichiarato zoppo.

    Raises:
        ValueError: alla prima incompatibilita' trovata.
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


# ----------------------------------------------------------------------
# Appaiamento per nome.
# ----------------------------------------------------------------------

def paired_values(data_a, data_b, axis: str, metric: str, k: int):
    """Vettori appaiati (a, b) delle query comuni e non-NaN in entrambi.

    Args:
        data_a, data_b: `PerQueryData` dei due sistemi.
        axis:   nome dell'asse ("composition" | "topology" | "geometry").
        metric: "ndcg" | "recall" | "map".
        k:      profondita'.

    Returns:
        (a, b, names) — array float64 [P] allineati e i nomi delle P query
        tenute, nell'ordine in cui compaiono in A (deterministico).
    """
    # gli array del contratto si chiamano come la metrica ("ndcg"|"recall"|"map")
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
            continue                   # skip singleton: vale in AND sui due file
        a_vals.append(x)
        b_vals.append(y)
        names.append(str(name))

    return (np.asarray(a_vals, dtype=float),
            np.asarray(b_vals, dtype=float),
            names)


def paired_self_rr(data_a, data_b):
    """Vettori appaiati del reciprocal rank del self, per il partial.

    Come `paired_values` ma senza asse ne' k: `self_rr` e' [Q]. L'appaiamento
    resta **sui nomi**, e restano solo le query presenti e non-NaN in entrambi.

    Args:
        data_a, data_b: `PerQueryData` di due run in `mode="partial"`.

    Returns:
        (a, b, names) — array float64 [P] allineati e i nomi tenuti.

    Raises:
        ValueError: se uno dei due file non contiene `self_rr` (cioe' non e' una
            run partial, oppure e' stato scritto da una versione precedente).
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


# ----------------------------------------------------------------------
# Test statistici.
# ----------------------------------------------------------------------

def bootstrap_ci(diff: np.ndarray, b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED,
                 alpha: float = 0.05) -> tuple[float, float]:
    """CI percentile del delta medio, ricampionando le COPPIE con rimpiazzo.

    Ricampionare le coppie (non i due vettori separatamente) e' cio' che rende
    il bootstrap appaiato: conserva la correlazione fra i due sistemi sulla
    stessa query, che e' proprio l'informazione che rende il test sensibile.
    Il calcolo e' a blocchi per non allocare [B, P] in una volta.
    """
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
    """p-value di Wilcoxon signed-rank appaiato.

    Returns:
        (p, nota) — `p` e' NaN se il test non e' calcolabile; `nota` spiega
        sempre il perche' (scipy assente, differenze tutte nulle, campione
        vuoto), cosi' un NaN non passa mai per un risultato.
    """
    if len(a) == 0:
        return float("nan"), "nessuna coppia"
    if np.allclose(a - b, 0.0):
        return float("nan"), "differenze tutte nulle (test non definito)"
    if not HAVE_SCIPY:
        return float("nan"), "scipy non disponibile: Wilcoxon non calcolato"
    result = _scipy_wilcoxon(a, b)
    return float(result.pvalue), ""


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Correzione di Holm-Bonferroni su una famiglia di test.

    Ordina i p crescenti e moltiplica l'i-esimo per (m - i), imponendo poi la
    monotonia: controlla il family-wise error rate senza l'eccesso di
    conservativita' di Bonferroni. I NaN restano NaN (test non calcolato) e non
    entrano nel conteggio della famiglia.
    """
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
    """Confronto appaiato completo su (asse, metrica, k). Ritorna un dict di numeri.

    Include il controllo di copertura: `reliable` e' False quando le coppie
    appaiate sono meno di `min_pair_fraction` delle query del file piu' piccolo.
    Il risultato viene comunque calcolato (serve a capire cosa e' successo) ma
    marcato: un delta su mezzo campione non e' il delta pubblicabile.
    """
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
    """Confronto appaiato sul self-recovery (`self_rr`), endpoint del criterio A.5.

    Ritorna lo **stesso dict** di `compare`, cosi' `print_rows` lo stampa senza
    saperne nulla: al posto dell'asse c'e' l'etichetta "self-recovery" e la
    profondita' e' "-" perche' il reciprocal rank non ne ha una.
    """
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


# ----------------------------------------------------------------------
# Stampa.
# ----------------------------------------------------------------------

def print_rows(rows: list[dict], title: str, adjusted: dict[str, float] | None = None) -> None:
    """Tabella dei confronti; con `adjusted` aggiunge la colonna p corretto (Holm).

    Le righe con copertura insufficiente sono marcate `!` e ripetute in coda come
    avviso: devono saltare all'occhio, non nascondersi in una colonna.
    """
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


# ----------------------------------------------------------------------
# Entrypoint.
# ----------------------------------------------------------------------

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

    # `self_rr` non si declina per asse: un confronto solo, nessuna famiglia di
    # test e quindi nessuna correzione di Holm. Le metriche per-asse non si
    # stampano qui comunque: sono un'altra domanda, si chiedono con
    # `--metric ndcg`. ⚠️ Dal 25 ago 2026 (B.4) i file partial NUOVI hanno
    # `exclude_self=True` e le loro metriche per-asse SONO confrontabili col
    # full; quelli scritti prima hanno False e `check_compatible` li rifiuta —
    # che è il comportamento voluto, non un bug.
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

    # Endpoint primario: nDCG@10 sui tre assi, famiglia di 3 test -> Holm.
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
