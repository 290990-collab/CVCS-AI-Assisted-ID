"""GCN encoder (Kipf & Welling 2017).

Isotropic symmetric weighted neighbour mean (weight ~ 1/sqrt(deg_i * deg_j)); no `edge_attr`.
GCNConv adds its own self-loops, so spurious ones are removed by `RemoveSpuriousSelfLoops`
(src/graph/transforms.py).
"""

from __future__ import annotations

from torch_geometric.nn import GCNConv, MessagePassing

from src.graph.models.base import BaseGraphEncoder


class GCNEncoder(BaseGraphEncoder):
    """GCNConv encoder; arguments as BaseGraphEncoder."""

    uses_edge_attr = False

    def build_conv(self, in_channels: int, out_channels: int) -> MessagePassing:
        return GCNConv(in_channels, out_channels)
