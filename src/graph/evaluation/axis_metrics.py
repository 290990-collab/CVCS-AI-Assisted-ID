# src/graph/evaluation/axis_metrics.py

"""
Accumulo e stampa delle metriche per-asse del ramo graph.

Sono gli helper che stavano dentro `graph_evaluate.py`: qui vivono in un modulo
a se' perche' servono a DUE chiamanti che devono misurare la stessa identica
cosa:

- `graph_evaluate.py`  -> valutazione finale (gallery = intero snapshot, query
                          dallo split test);
- `retrieval_probe.py` -> sonda usata DURANTE il training per scegliere il
                          checkpoint (gallery piccola dal valid).

Tenerli in un solo posto e' il punto: se la sonda misurasse il retrieval in modo
anche solo leggermente diverso dalla valutazione finale, selezionare i pesi su di
essa non avrebbe senso. Le funzioni NUMERICHE (nDCG, Recall, AP) restano quelle
del core condiviso `src/evaluation/metrics.py`, usate as-is e non duplicate.

Nota: questi helper sono la copia locale del ramo graph (il ramo vision ha i
suoi, deliberatamente separati: i due rami restano autonomi, niente refactor
cross-ramo).
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.evaluation.relevance import AXES, DISCRETE_AXES


def new_metrics(k_values):
    """Contenitore vuoto delle metriche: asse -> metrica -> K -> lista di valori.

    Una lista per (asse, metrica, K) perche' ogni query contribuisce un valore e
    la media si fa alla fine (le query senza rilevanti non entrano affatto,
    invece di entrare come zero).
    """
    return {
        ax: {"ndcg": {k: [] for k in k_values},
             "recall": {k: [] for k in k_values},
             "map": {k: [] for k in k_values}}
        for ax in AXES
    }


def accumulate_axes(metrics, skipped, axes, qi, ret_rows, k_values, exclude_self):
    """Aggiorna `metrics`/`skipped` con le metriche per-asse di UNA query.

    Args:
        metrics:      contenitore da `new_metrics` (modificato in place).
        skipped:      dict asse -> conteggio query escluse (modificato in place).
        axes:         `GalleryAxes` allineata alle righe della gallery.
        qi:           indice di riga della query dentro la gallery.
        ret_rows:     righe recuperate, nell'ordine del retriever.
        k_values:     profondita' da valutare.
        exclude_self: True (full) -> il self esce da gallery, rilevanti e
                      risultati. False -> riservato al futuro partial su grafo,
                      dove ritrovare la pianta sorgente E' il compito.
    """
    ret_rows = np.asarray(ret_rows, dtype=int)
    not_self = np.ones(len(axes), dtype=bool)
    if exclude_self:
        not_self[qi] = False

    for ax in AXES:
        sims = axes.sim(ax, qi)
        gallery_gains = sims[not_self]
        retrieved_gains = sims[ret_rows] if len(ret_rows) else np.empty(0)
        for k in k_values:
            metrics[ax]["ndcg"][k].append(ndcg_at_k(retrieved_gains, gallery_gains, k))

        if ax in DISCRETE_AXES:
            rel = axes.relevant(ax, qi)
            if exclude_self:
                rel[qi] = False
            num_rel = int(rel.sum())
            if num_rel == 0:
                skipped[ax] += 1
                continue
            retrieved_rel = rel[ret_rows] if len(ret_rows) else np.empty(0, dtype=bool)
            for k in k_values:
                metrics[ax]["recall"][k].append(
                    recall_at_k(retrieved_rel, num_rel, k)
                )
                metrics[ax]["map"][k].append(
                    average_precision_at_k(retrieved_rel, num_rel, k)
                )


def mean_metric(values: list[float]) -> float:
    """Media sulle query, NaN se nessuna query ha contribuito."""
    return float(np.mean(values)) if values else float("nan")


def print_axis_tables(metrics, k_values) -> None:
    """Stampa una tabella asse x metrica per ogni K."""
    for k in k_values:
        print(f"================  K = {k}  ================")
        header = f"{'asse':<13} {'nDCG':>8} {'Recall':>8} {'mAP':>8}"
        print(header)
        print("-" * len(header))
        for ax in AXES:
            ndcg = mean_metric(metrics[ax]["ndcg"][k])
            if ax in DISCRETE_AXES:
                rec = mean_metric(metrics[ax]["recall"][k])
                mapk = mean_metric(metrics[ax]["map"][k])
                print(f"{ax:<13} {ndcg:>8.3f} {rec:>8.3f} {mapk:>8.3f}")
            else:
                print(f"{ax:<13} {ndcg:>8.3f} {'—':>8} {'—':>8}")
        print()
