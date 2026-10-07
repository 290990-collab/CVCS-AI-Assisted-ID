"""Per-axis retrieval metrics evaluated against the whole gallery.

Pure numeric functions, one evaluation per query; the mean over queries is done by `evaluate.py`.

- graded (`ndcg_at_k`): axis similarity [0,1] as gain; IDCG from the best K candidates of the whole gallery.
- binary set (`recall_at_k`, `average_precision_at_k`): need the relevant set (exact equivalence class of
  the axis), normalise by `min(k, #relevant_total)`, return None for a query with no relevant item
  (singleton class); such queries leave the mean instead of counting as zero.

With at least `k` relevant items the denominator is `k` and `recall_at_k` is `|top-k ∩ relevant| / k`,
i.e. Precision@k. On RPLAN the composition equivalence classes hold thousands of plans, so the
"Recall" of that axis is almost always a precision: tables and reports must say so. Name and `"recall"`
key of the `perquery/1` contract stay unchanged. `src/evaluation/metric_diagnostics.py` measures
how often, per axis and k.
"""

import numpy as np


def _dcg(gains: np.ndarray) -> float:
    """DCG: sum of gains discounted by 1/log2(rank+1), gains in ranked order."""
    gains = np.asarray(gains, dtype=float)
    ranks = np.arange(1, len(gains) + 1)
    return float((gains / np.log2(ranks + 1)).sum())


def ndcg_at_k(retrieved_gains, gallery_gains, k: int) -> float:
    """nDCG@k with graded relevance, in [0,1].

    retrieved_gains: axis similarities in retrieval order (self excluded);
    gallery_gains: gains of the whole gallery for this query (self excluded), for the IDCG.
    """
    rg = np.asarray(retrieved_gains, dtype=float)[:k]
    dcg = _dcg(rg)

    ideal = np.sort(np.asarray(gallery_gains, dtype=float))[::-1][:k]
    idcg = _dcg(ideal)

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_relevant, num_relevant_total: int, k: int) -> float | None:
    """Truncated Recall@k: `|top-k ∩ relevant| / min(k, #relevant_total)`; None if no relevant item.

    With `#relevant_total >= k` this is Precision@k (see module docstring).
    retrieved_relevant: bool array in retrieval order; num_relevant_total excludes self.
    """
    if num_relevant_total <= 0:
        return None
    rr = np.asarray(retrieved_relevant, dtype=bool)[:k]
    return float(rr.sum()) / min(k, num_relevant_total)


def average_precision_at_k(
    retrieved_relevant, num_relevant_total: int, k: int
) -> float | None:
    """AP@k: mean of precision@rank at the relevant positions, normalised by `min(k, #relevant_total)`.

    None if no relevant item. With classes larger than `k` the maximum 1 needs all top-k relevant.
    """
    if num_relevant_total <= 0:
        return None
    rr = np.asarray(retrieved_relevant, dtype=bool)[:k]

    hits = 0
    precision_sum = 0.0
    for rank, is_rel in enumerate(rr, start=1):
        if is_rel:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(k, num_relevant_total)
