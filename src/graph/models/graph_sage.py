# src/graph/models/graph_sage.py

"""
Encoder GraphSAGE (Hamilton et al. 2017).

SAGEConv tiene SEPARATE la rappresentazione del nodo e quella aggregata dai
vicini: `h_i' = W_self * h_i + W_neigh * aggr(h_j : j in N(i))`. Rispetto a GCN
(che mescola se' stesso e i vicini in un'unica media pesata dal grado), questo
preserva meglio l'identita' del nodo -- utile qui, dove il TIPO della stanza
(l'one-hot) e' un segnale forte che non vogliamo diluire nella media dei vicini.

Aggregatore di default: "mean". Non usa `edge_attr`.
"""

from __future__ import annotations

from torch_geometric.nn import MessagePassing, SAGEConv

from src.graph.models.base import BaseGraphEncoder


class GraphSAGEEncoder(BaseGraphEncoder):
    """Encoder basato su SAGEConv.

    Args (oltre a quelli di BaseGraphEncoder):
        aggr: aggregatore dei vicini ("mean" | "max" | "add" | "lstm").
    """

    # SAGE non consuma il tipo di relazione sugli archi.
    uses_edge_attr = False

    def __init__(self, aggr: str = "mean", **kwargs) -> None:
        # Impostato prima di super().__init__: build_conv lo legge durante la
        # costruzione dello stack.
        self.aggr = aggr
        super().__init__(**kwargs)

    def build_conv(self, in_channels: int, out_channels: int) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, aggr=self.aggr)
