"""GraphSAGE encoder (Hamilton et al. 2017).

Keeps node and aggregated-neighbour representations separate,
`h_i' = W_self h_i + W_neigh aggr(h_j : j in N(i))`, preserving the room-type signal.
Default aggregator "mean"; no `edge_attr`.
"""

from __future__ import annotations

from torch_geometric.nn import MessagePassing, SAGEConv

from src.graph.models.base import BaseGraphEncoder


class GraphSAGEEncoder(BaseGraphEncoder):
    """SAGEConv encoder.

    Args (besides BaseGraphEncoder's):
        aggr: neighbour aggregator ("mean" | "max" | "add" | "lstm").
    """

    uses_edge_attr = False

    def __init__(self, aggr: str = "mean", **kwargs) -> None:
        self.aggr = aggr
        super().__init__(**kwargs)

    def build_conv(self, in_channels: int, out_channels: int) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, aggr=self.aggr)
