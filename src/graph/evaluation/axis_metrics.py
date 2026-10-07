"""
Per-axis metric accumulation and printing for the graph branch.

Shared by `graph_evaluate.py` (final evaluation) and `retrieval_probe.py` (checkpoint selection during training)
so both measure the same thing. The numeric functions (nDCG, Recall, AP) come from `src/evaluation/metrics.py`.
Local to the graph branch: the vision branch keeps its own helpers.
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
    """Empty container: axis -> metric -> K -> list of per-query values (queries without relevants are skipped, not zero)."""
    return {
        ax: {"ndcg": {k: [] for k in k_values},
             "recall": {k: [] for k in k_values},
             "map": {k: [] for k in k_values}}
        for ax in AXES
    }


def accumulate_axes(metrics, skipped, axes, qi, ret_rows, k_values, exclude_self):
    """Add the per-axis metrics of ONE query to `metrics`; `skipped[axis]` counts queries without relevants.

    `qi` is the query row in the gallery, `ret_rows` the retrieved rows in retriever order.
    `exclude_self=True` (full) removes the query from gallery, relevants and results.
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
    """Mean over queries, NaN if empty."""
    return float(np.mean(values)) if values else float("nan")


def print_axis_tables(metrics, k_values) -> None:
    """Print an axis x metric table per K."""
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
