# src/evaluation/metric_diagnostics.py

"""
Diagnostica delle METRICHE stesse: non "quanto è bravo il modello", ma "cosa
sta misurando davvero il numero che riportiamo".

Tre domande, tre risposte quantitative. Tutte e tre si rispondono **post-hoc**:
la prima e la seconda leggono i file per-query già scritti (`perquery/1`), la
terza ricostruisce le feature di rilevanza dai `.mat`. Nessuna GPU, nessun
retrieval, nessuna run da rifare.

1. **Il "Recall" è davvero recall?** `recall_at_k` normalizza per
   `min(k, #rilevanti)`. Quando una query ha almeno `k` rilevanti quel `min` vale
   `k`, la formula diventa `|top-k ∩ rilevanti| / k` e cioè **Precision@k**. Non
   è un errore di calcolo (è la scelta dichiarata in `metrics.py`), ma cambia il
   NOME di ciò che si riporta: `precision_regime` dice su quale frazione di query
   accade, per asse e per k.

2. **Su quante query è calcolata la media?** Le query la cui classe di
   equivalenza è singleton non hanno rilevanti: Recall/mAP non sono definiti e
   quelle query escono dalla media (`num_relevant == 0` nel contratto). Se ne
   esce una frazione grande, la media residua descrive un SOTTOINSIEME scelto
   dalla ground truth — tipicamente le query "facili", quelle con struttura
   comune. Il confronto fra sistemi resta appaiato (l'esclusione non dipende dal
   modello), ma il valore assoluto è ottimista e va riportato con la copertura
   accanto.

3. **I tre assi sono indipendenti?** `relevance.py` li presenta come tre segnali
   distinti, e la tentazione è leggere "vince su 2 assi su 3" come due evidenze.
   `axis_correlation` misura la correlazione di Pearson fra i gain sulle STESSE
   coppie: se è alta, i tre numeri non sono tre prove indipendenti e la frase va
   evitata.

Uso tipico:

    # 1 e 2: dai file per-query già su disco
    python -m src.evaluation.metric_diagnostics \
        --perquery results/perquery/vision_valid/*.npz --k 10

    # 3: dalla gallery (stima su un campione di coppie)
    python -m src.evaluation.metric_diagnostics \
        --gallery embeddings/vision/dinov2/natural/image_paths.json \
        --corr-queries 60 --corr-gallery 6000

⚠️ Il punto 3 costa: per ogni query calcola le similarità d'asse su tutta la
sotto-gallery campionata. I default sono tarati per stare sotto il `ulimit -t`
del login node (600 s di CPU); alzarli vuol dire passare a un nodo di calcolo.
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

# Valori speciali di `num_relevant` nel contratto perquery/1.
NO_RELEVANT_SET = -1    # asse continuo: nessuna classe di equivalenza
SKIPPED = 0             # classe singleton: query esclusa da Recall/mAP


# ----------------------------------------------------------------------
# 1 + 2 — diagnostica dai file per-query.
# ----------------------------------------------------------------------

def axis_diagnostics(data, axis: str, k: int) -> dict | None:
    """Regime della metrica e copertura della media, per un asse discreto.

    Args:
        data: `PerQueryData` di una run.
        axis: nome dell'asse (deve essere in DISCRETE_AXES).
        k:    profondità a cui valutare il regime.

    Returns:
        dict con:
          n_query   query presenti nel file
          n_skipped query escluse da Recall/mAP (classe singleton)
          coverage  frazione di query su cui la media è calcolata
          n_prec    query con #rilevanti >= k (Recall degenera in Precision@k)
          precision_regime  frazione delle query VALUTATE in quel regime
          median_relevant / p90_relevant  dimensione della classe di equivalenza
        Oppure None se l'asse non ha un insieme di rilevanti nel file.
    """
    a = data.axis_index(axis)
    num_rel = np.asarray(data.num_relevant[a], dtype=np.int64)
    if num_rel.size == 0 or np.all(num_rel == NO_RELEVANT_SET):
        return None

    n_query = int(num_rel.size)
    n_skipped = int((num_rel == SKIPPED).sum())
    scored = num_rel[num_rel > SKIPPED]        # le query che entrano nella media
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
    """`axis_diagnostics` su tutti gli assi discreti presenti nel file."""
    out = {}
    for axis in DISCRETE_AXES:
        try:
            d = axis_diagnostics(data, axis, k)
        except KeyError:              # asse assente dal file
            continue
        if d is not None:
            out[axis] = d
    return out


def print_diagnostics_table(rows: list[tuple[str, dict[str, dict]]], k: int) -> None:
    """Tabella run x asse: copertura della media e regime della metrica.

    Args:
        rows: lista di (etichetta_run, risultato di `run_diagnostics`).
        k:    profondità usata per il regime (compare nell'intestazione).
    """
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


# ----------------------------------------------------------------------
# 3 — correlazione fra gli assi di rilevanza.
# ----------------------------------------------------------------------

def axis_correlation(
    axes: GalleryAxes,
    num_queries: int = DEFAULT_CORR_QUERIES,
    seed: int = CORR_SEED,
) -> tuple[np.ndarray, dict[str, dict], int]:
    """Correlazione di Pearson fra i gain dei tre assi, sulle stesse coppie.

    Per ogni query campionata si prendono le similarità verso TUTTA la gallery
    passata (self escluso) sui tre assi: ogni coppia (query, riga) contribuisce
    un punto con tre coordinate. La correlazione è quindi calcolata esattamente
    sulle quantità che le metriche usano come gain.

    Args:
        axes:        gallery su cui misurare (può essere un sottocampione: la
                     correlazione è una statistica di coppia, non serve l'intera).
        num_queries: quante query campionare.
        seed:        seme del campionamento (riproducibilità).

    Returns:
        (corr, stats, n_pairs) — matrice [3,3] nell'ordine di AXES, statistiche
        marginali per asse (media/std/p1/p99) e il numero di coppie usate.
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
    """Matrice di correlazione + marginali, con la lettura già fatta."""
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


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------

def build_gallery_axes(gallery_path: str, sample: int, seed: int = CORR_SEED):
    """`GalleryAxes` da un `image_paths.json`/`names.json`, opzionalmente su un
    sottocampione di righe (la correlazione non richiede l'intera gallery).

    Returns:
        (axes, n_rows_usate).
    """
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
