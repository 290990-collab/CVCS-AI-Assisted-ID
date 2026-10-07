"""
Retrieval probe on valid: per-axis nDCG measured during training, instead of the loss.

InfoNCE pushes apart every plan in the batch, including architecturally similar ones, so val-loss ranks
configurations in the opposite order of retrieval quality (the best-loss GAT was last in nDCG).
The probe selects checkpoints with a mini-retrieval under the final evaluation protocol.

Gallery and per-axis ground truth are fixed: computed once at construction, per epoch only
forward + dot product + metric accumulation.

Protocol (as `graph_evaluate`, scaled down):
- gallery = sample of the **valid** split (never test);
- queries = gallery rows, self-match excluded;
- ranking = exact dot product on L2-normalised embeddings (same as FAISS `IndexFlatIP`);
- metrics = `accumulate_axes`, same code as the final evaluation.

Values are not comparable with the final report (smaller gallery, different IDCG); only across epochs and configs.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.rplan_metadata import load_metadata
from src.evaluation.relevance import AXES, DISCRETE_AXES, GalleryAxes
from src.graph.evaluation.axis_metrics import accumulate_axes, mean_metric, new_metrics

# a single axis or the mean of the three
SELECTION_CRITERIA = ("mean",) + AXES


class RetrievalProbe:
    """Mini-retrieval on a fixed gallery: `score(encoder)` -> per-axis nDCG. Build with `RetrievalProbe.build(...)`."""

    def __init__(self, graphs, names, query_rows, k: int, batch_size: int):
        self.graphs = graphs
        self.names = names
        self.query_rows = list(query_rows)
        self.k = k
        self.batch_size = batch_size

        # per-axis ground truth: the expensive part, computed once
        self.axes = GalleryAxes([load_metadata(n) for n in names])

        # queries without .mat metadata are not evaluable (in practice none)
        self.query_rows = [qi for qi in self.query_rows if self.axes.valid[qi]]

    @classmethod
    def build(
        cls,
        dataset,
        num_queries: int = 500,
        gallery_size: int = 5000,
        k: int = 10,
        seed: int = 42,
        batch_size: int = 512,
    ) -> "RetrievalProbe":
        """Sample gallery and query rows from `dataset` (normally the valid split, with the training `transform`).

        `gallery_size` 0 = whole dataset; `seed` makes the sample fixed so epochs are comparable.
        """
        pool = list(range(len(dataset)))
        rng = random.Random(seed)

        size = len(pool) if gallery_size <= 0 else min(gallery_size, len(pool))
        gallery_rows = rng.sample(pool, size)

        graphs = [dataset[i] for i in gallery_rows]
        names = [g.name for g in graphs]

        # queries = first gallery rows (already a random sample)
        query_rows = range(min(num_queries, len(graphs)))

        return cls(graphs, names, query_rows, k=k, batch_size=batch_size)

    def __len__(self) -> int:
        return len(self.graphs)

    @property
    def num_queries(self) -> int:
        return len(self.query_rows)

    @torch.no_grad()
    def score(self, encoder, device) -> dict[str, float]:
        """Run the mini-retrieval: `ndcg_<axis>` (3 axes), `recall_<axis>`/`map_<axis>` (discrete axes), `mean` of the three nDCG."""
        was_training = encoder.training
        encoder.eval()

        # gallery embeddings (encoder output is L2-normalised)
        loader = DataLoader(self.graphs, batch_size=self.batch_size, shuffle=False)
        emb = torch.cat([encoder(batch.to(device)) for batch in loader])

        # exact inner product = cosine
        q_idx = torch.as_tensor(self.query_rows, dtype=torch.long, device=emb.device)
        sims = emb[q_idx] @ emb.t()
        # +1: the self-match takes the first position and is dropped
        top = torch.topk(sims, min(self.k + 1, sims.size(1)), dim=1).indices.cpu().numpy()

        if was_training:
            encoder.train()

        k_values = (self.k,)
        metrics = new_metrics(k_values)
        skipped = {ax: 0 for ax in DISCRETE_AXES}
        for j, qi in enumerate(self.query_rows):
            ret_rows = [int(r) for r in top[j] if r != qi][: self.k]
            accumulate_axes(metrics, skipped, self.axes, qi, ret_rows, k_values,
                            exclude_self=True)

        out: dict[str, float] = {}
        for ax in AXES:
            out[f"ndcg_{ax}"] = mean_metric(metrics[ax]["ndcg"][self.k])
            if ax in DISCRETE_AXES:
                out[f"recall_{ax}"] = mean_metric(metrics[ax]["recall"][self.k])
                out[f"map_{ax}"] = mean_metric(metrics[ax]["map"][self.k])

        out["mean"] = float(np.mean([out[f"ndcg_{ax}"] for ax in AXES]))
        return out

    @staticmethod
    def selection_value(scores: dict[str, float], criterion: str) -> float:
        """Single number to select weights on: "mean" or an axis name."""
        if criterion not in SELECTION_CRITERIA:
            raise ValueError(
                f"criterio '{criterion}' non valido (scegli tra {SELECTION_CRITERIA})"
            )
        return scores["mean"] if criterion == "mean" else scores[f"ndcg_{criterion}"]

    @staticmethod
    def format_scores(scores: dict[str, float]) -> str:
        """Compact log line: nDCG of the three axes + mean."""
        return (
            f"C {scores['ndcg_composition']:.3f} "
            f"T {scores['ndcg_topology']:.3f} "
            f"G {scores['ndcg_geometry']:.3f} "
            f"| mean {scores['mean']:.4f}"
        )
