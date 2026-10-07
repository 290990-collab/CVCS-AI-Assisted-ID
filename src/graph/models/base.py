"""Common contract for graph encoders (GCN / GAT / GraphSAGE).

Unlike the frozen vision encoders these are trained: `forward` propagates gradients and
returns one L2-normalised embedding per graph for FAISS.

Subclasses differ only in the convolution: the layer stack, node->graph pooling,
projection and L2-norm live in `BaseGraphEncoder`; subclasses implement `build_conv`.

Default 2 layers: RPLAN graphs have 4-8 nodes and diameter 2-3; more hops over-smooth.
Pooling: `add` preserves room count, `mean` normalises it away, `mean_max` concatenates both.

`raw_skip`: with random weights the add-pool of raw node features alone reaches mean nDCG
0.805 (trained model 0.862), since type one-hot (13) + geometry (6) sums to the type
histogram plus aggregated geometry. A random linear projection degrades composition
(0.884 raw add-pool, 0.850 projected) and InfoNCE pushes against it. With `raw_skip` the
raw add-pool is concatenated to the learned embedding before the projection, so the
network capacity goes to topology (+0.100 from training, vs +0.056 and +0.012 for the
other two axes). Default False to keep saved checkpoints loadable (changes `proj` shape).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

POOLINGS = ("add", "mean", "max", "mean_max")


def global_pool(x: torch.Tensor, batch: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool node features x [N, D] into [G, D] per graph (`batch` maps node -> graph); [G, 2D] for "mean_max"."""
    if pooling == "add":
        return global_add_pool(x, batch)
    if pooling == "mean":
        return global_mean_pool(x, batch)
    if pooling == "max":
        return global_max_pool(x, batch)
    if pooling == "mean_max":
        return torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1
        )
    raise ValueError(f"pooling '{pooling}' non valido (scegli tra {POOLINGS})")


class BaseGraphEncoder(nn.Module, ABC):
    """Trainable graph encoder: nodes -> L2-normalised graph embedding.

    Args:
        in_dim: node feature size (19 for RPLAN).
        out_dim: final (FAISS) embedding size.
        raw_skip: concatenate the raw-feature add-pool to the pooled embedding.

    Subclasses implement `build_conv`; set `uses_edge_attr = True` if the conv consumes edge_attr.
    """

    uses_edge_attr: bool = False

    def __init__(
        self,
        in_dim: int = 19,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 2,
        pooling: str = "add",
        dropout: float = 0.0,
        raw_skip: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling '{pooling}' non valido (scegli tra {POOLINGS})")

        self.pooling = pooling
        self.dropout = dropout
        self.raw_skip = raw_skip
        self._out_dim = out_dim

        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in = in_dim if layer_idx == 0 else hidden_dim
            self.convs.append(self.build_conv(layer_in, hidden_dim))

        # "mean_max" doubles the projection input
        pooled_dim = hidden_dim * (2 if pooling == "mean_max" else 1)
        if raw_skip:
            pooled_dim += in_dim

        self.proj = nn.Linear(pooled_dim, out_dim)

    @abstractmethod
    def build_conv(self, in_channels: int, out_channels: int) -> MessagePassing:
        """One PyG message-passing layer (GCNConv | GATv2Conv | SAGEConv)."""
        raise NotImplementedError

    def forward(self, data) -> torch.Tensor:
        """`Data`/`Batch` -> unit-norm embeddings [G, out_dim]."""
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        # batch is absent for a single uncollated graph
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # ReLU + dropout between layers, not after the last
        last = len(self.convs) - 1
        for i, conv in enumerate(self.convs):
            if self.uses_edge_attr:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            if i < last:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # L2-norm: cosine similarity as FAISS inner product
        graph_emb = global_pool(x, batch, self.pooling)

        # raw add-pool of `data.x` = type histogram + geometry, independent of learned weights
        if self.raw_skip:
            graph_emb = torch.cat(
                [graph_emb, global_add_pool(data.x, batch)], dim=-1
            )

        z = self.proj(graph_emb)
        return F.normalize(z, p=2, dim=-1)

    @property
    def embedding_dim(self) -> int:
        """Embedding size of forward()."""
        return self._out_dim
