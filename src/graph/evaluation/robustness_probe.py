"""
Robustness probe on valid: selects the graph checkpoint on self-recovery under masking, not on full-plan nDCG.

`retrieval_probe.py` picks the epoch on full-plan topology nDCG@10, a different metric from the evaluation one;
this probe measures the final evaluation quantity at each epoch, scaled down.

Protocol (as `graph_evaluate --partial`, restricted to 2000 queries):
- gallery = the whole shared gallery (`gallery_names` of configs/graph_retrieval.yaml), canonical order,
  so the same `qi` and per-query seed;
- queries = **valid** rows disjoint from the evaluation queries (`graph_evaluate.sample_query_rows`);
- damage = `make_partial_graph(meta, "random", {"fraction": f}, Random(partial_seed + qi))`, for f in `AUC_FRACTIONS`;
- score = self RR (1/rank, 0 beyond max(k_values)), MRR per f, AUC = mean over f.

Differences from `graph_evaluate`:
- a query emptied at any f is dropped at all f, so the AUC averages MRR over the same queries;
- ranking = exact dot product, ties won by the self (rank = 1 + strictly more similar rows), as in the vision probe;
  FAISS does not guarantee tie order, so exact ties may differ.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from src.data.rplan_metadata import load_metadata
from src.evaluation.perquery import gallery_sha1
from src.evaluation.robustness_auc import AUC_FRACTIONS
from src.graph.graph_partial_query import make_partial_graph

# probe queries always come from valid
PROBE_SPLIT = "valid"
# default of `--partial-seed` in graph_evaluate: same removed rooms
DEFAULT_PARTIAL_SEED = 42


def probe_query_rows(valid_rows, eval_rows, n: int, seed: int) -> list[int]:
    """`n` rows of `valid_rows` not in `eval_rows`; the pool is sorted first so the result is input-order independent."""
    pool = sorted(set(valid_rows) - set(eval_rows))
    return random.Random(seed).sample(pool, n)


def partial_query_graphs(metas, rows, fractions, seed: int, transform, lost_marker: bool):
    """Damaged graphs per fraction on the same queries for all f.

    `metas` are `RoomMeta` (or None) aligned to `rows` (`qi` in the shared gallery); `seed` is the evaluation
    `partial_seed` (per query `seed + qi`); `transform` None = raw graphs; `lost_marker` as in `graph_evaluate.evaluate_partial`.
    Returns (graphs, kept_rows, n_dropped): `graphs[f]` is aligned to `kept_rows`; dropped = emptied at some f or no metadata.
    """
    per_f = {f: [] for f in fractions}
    kept_rows, n_dropped = [], 0
    for meta, qi in zip(metas, rows):
        if meta is None:
            n_dropped += 1
            continue
        built = {}
        for f in fractions:
            graph, _ = make_partial_graph(
                meta, "random", {"fraction": float(f)}, random.Random(seed + qi),
                lost_marker=lost_marker,
            )
            if graph is None:
                break
            built[f] = transform(graph) if transform is not None else graph
        if len(built) != len(fractions):
            n_dropped += 1
            continue
        for f in fractions:
            per_f[f].append(built[f])
        kept_rows.append(qi)
    return per_f, kept_rows, n_dropped


@torch.no_grad()
def self_reciprocal_ranks(q: torch.Tensor, gallery: torch.Tensor, rows: torch.Tensor,
                          max_rank: int, chunk: int = 256) -> torch.Tensor:
    """Self reciprocal rank per query, 0 beyond `max_rank`.

    Same semantics as `src/vision/training/retrieval_probe.self_reciprocal_ranks` (duplicated, branches stay independent).
    `q` [P, d] and `gallery` [N, d] are L2-normalised; rank = 1 + rows strictly above the self, so ties go to the self.
    """
    out = []
    for i in range(0, q.shape[0], chunk):
        s = q[i:i + chunk] @ gallery.T                                # [c, N]
        own = s.gather(1, rows[i:i + chunk, None])                    # [c, 1]
        rank = 1 + (s > own).sum(dim=1)
        out.append(torch.where(rank <= max_rank, 1.0 / rank.float(),
                               torch.zeros_like(rank, dtype=torch.float)))
    return torch.cat(out)


class RobustnessProbe:
    """Self-recovery under random masking on a fixed shared gallery.

    Build with `RobustnessProbe.build(...)`; `score(encoder, device)` returns `{"auc", "mrr_f0.25", "mrr_f0.5", "mrr_f0.75"}`.
    """

    def __init__(self, dataset_all, gallery_ds_rows, names, query_graphs, query_rows,
                 fractions, max_rank: int, batch_size: int, info: dict):
        self.dataset_all = dataset_all
        self.gallery_ds_rows = list(gallery_ds_rows)   # gallery row -> dataset index
        self.names = names
        self.query_graphs = query_graphs               # {f: [Data]} aligned to query_rows
        self.query_rows = list(query_rows)
        self.fractions = tuple(fractions)
        self.max_rank = int(max_rank)
        self.batch_size = int(batch_size)
        self.info = info
        self._gallery_batches = None                   # prepared on first score()
        self._query_batches = None
        self._rows_t = None

    @classmethod
    def build(cls, dataset_all, transform, eval_config_path, num_queries: int, seed: int,
              lost_marker: bool, batch_size: int) -> "RobustnessProbe":
        """Rebuild the evaluation gallery and queries and pick the probe queries.

        `dataset_all` is the whole `RplanGraphDataset` (no split) with `transform`; `eval_config_path` is
        configs/graph_retrieval.yaml; `seed` is both the evaluation `partial_seed` (42) and the probe row sampling seed.
        """
        # local import: graph_evaluate imports train_gnn, which imports this module
        from src.graph.evaluation import graph_evaluate

        ecfg = OmegaConf.load(eval_config_path)
        if not ecfg.get("gallery_names"):
            raise ValueError(
                f"{eval_config_path}: gallery_names e' null — la sonda di robustezza "
                "richiede la gallery condivisa (stessi qi, stessi seed per-query della valutazione)"
            )
        if dataset_all._indices is not None:
            raise ValueError("RobustnessProbe.build richiede il dataset INTERO (split=None)")

        # same restricted gallery as graph_evaluate.main (dataset order, then restrict_gallery); embeddings unused, width-0 placeholder
        ds_names = list(dataset_all._data.name)
        placeholder = np.empty((len(ds_names), 0), dtype=np.float32)
        _, names, row_of = graph_evaluate.restrict_gallery(
            placeholder, ds_names, str(ecfg.gallery_names))
        gallery_ds_rows = [0] * len(names)
        for ds_idx, row in row_of.items():
            gallery_ds_rows[row] = ds_idx

        eval_rows = graph_evaluate.sample_query_rows(
            dataset_all, int(ecfg.num_queries), int(ecfg.seed), str(ecfg.split), row_of)
        pool = sorted(row_of[i] for i in dataset_all.split_indices(PROBE_SPLIT) if i in row_of)
        rows = probe_query_rows(pool, eval_rows, num_queries, seed)
        overlap = len(set(rows) & set(eval_rows))
        assert overlap == 0, f"sonda e valutazione condividono {overlap} query"

        metas = [load_metadata(names[qi]) for qi in rows]
        fractions = tuple(float(f) for f in AUC_FRACTIONS)
        query_graphs, kept_rows, n_dropped = partial_query_graphs(
            metas, rows, fractions, seed, transform, lost_marker)

        info = {
            "gallery_n": len(names),
            "gallery_sha1": gallery_sha1(names),
            "eval_excluded": len(set(pool) & set(eval_rows)),
            "overlap": overlap,
            "n_queries": len(kept_rows),
            "n_dropped": n_dropped,
            "max_rank": int(max(ecfg.k_values)),
            "fractions": list(fractions),
            "seed": int(seed),
        }
        return cls(dataset_all, gallery_ds_rows, names, query_graphs, kept_rows,
                    fractions, info["max_rank"], batch_size, info)

    def _batches(self, graphs, device):
        return [Batch.from_data_list(graphs[i:i + self.batch_size]).to(device)
                for i in range(0, len(graphs), self.batch_size)]

    def _prepare(self, device) -> None:
        graphs = [self.dataset_all[i] for i in self.gallery_ds_rows]
        mism = sum(g.name != n for g, n in zip(graphs, self.names))
        if mism:
            raise RuntimeError(f"RobustnessProbe: {mism} grafi della gallery non allineati ai nomi")
        self._gallery_batches = self._batches(graphs, device)
        self._query_batches = {f: self._batches(self.query_graphs[f], device)
                               for f in self.fractions}
        self._rows_t = torch.as_tensor(self.query_rows, dtype=torch.long, device=device)

    @torch.no_grad()
    def score(self, encoder, device) -> dict[str, float]:
        """Self-recovery AUC (mean over f of MRR); embeddings are not re-normalised (encoder output is L2-norm) and no DataLoader is used (global torch RNG untouched)."""
        was_training = encoder.training
        encoder.eval()
        if self._gallery_batches is None:
            self._prepare(device)

        gallery = torch.cat([encoder(b) for b in self._gallery_batches])
        out = {}
        for f in self.fractions:
            q = torch.cat([encoder(b) for b in self._query_batches[f]])
            rr = self_reciprocal_ranks(q, gallery, self._rows_t, self.max_rank)
            out[f"mrr_f{f}"] = float(rr.mean())

        if was_training:
            encoder.train()
        return {"auc": float(np.mean([out[f"mrr_f{f}"] for f in self.fractions])), **out}
