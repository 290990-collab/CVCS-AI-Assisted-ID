"""GATv2 encoder (Brody et al. 2021).

Learns an attention weight per edge; the only encoder using `edge_attr` (RPLAN relation type).
`heads` are averaged (`concat=False`) so the output width stays `out_channels`, as the
base stack assumes. `heads` and `edge_dim` are set before super().__init__ because the
base builds the layers there.
"""

from __future__ import annotations

from torch_geometric.nn import GATv2Conv, MessagePassing

from src.graph.graph_builder import NUM_RELATION_TYPES
from src.graph.models.base import BaseGraphEncoder


class GATEncoder(BaseGraphEncoder):
    """GATv2Conv encoder with typed-edge attention.

    Args (besides BaseGraphEncoder's):
        heads: attention heads (averaged).
        edge_dim: edge_attr size (10 = relation one-hot).
        attn_dropout: dropout on attention coefficients.
    """

    uses_edge_attr = True

    def __init__(
        self,
        heads: int = 4,
        edge_dim: int = NUM_RELATION_TYPES,
        attn_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        self.heads = heads
        self.edge_dim = edge_dim
        self.attn_dropout = attn_dropout
        super().__init__(**kwargs)

    def build_conv(self, in_channels: int, out_channels: int) -> MessagePassing:
        return GATv2Conv(
            in_channels,
            out_channels,
            heads=self.heads,
            concat=False,
            edge_dim=self.edge_dim,
            dropout=self.attn_dropout,
        )
