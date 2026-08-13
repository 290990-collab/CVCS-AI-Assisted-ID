# src/graph/models/gcn.py

"""
Encoder GCN (Graph Convolutional Network, Kipf & Welling 2017).

Il layer GCN aggiorna ogni nodo con una **media pesata simmetrica** delle feature
dei vicini (peso ~ 1/sqrt(deg_i * deg_j)) seguita da una trasformazione lineare.
E' l'aggregatore piu' semplice: isotropo (tutti i vicini contano allo stesso modo,
a meno del grado) e senza attenzione. Buona baseline prima di GAT/SAGE.

GCNConv aggiunge da se' i self-loop, quindi ha senso rimuovere quelli spurii con
`RemoveSpuriousSelfLoops` (vedi src/graph/transforms.py). Non usa `edge_attr`.
"""

from __future__ import annotations

from torch_geometric.nn import GCNConv, MessagePassing

from src.graph.models.base import BaseGraphEncoder


class GCNEncoder(BaseGraphEncoder):
    """Encoder basato su GCNConv. Vedi BaseGraphEncoder per gli argomenti."""

    # GCN non consuma il tipo di relazione sugli archi.
    uses_edge_attr = False

    def build_conv(self, in_channels: int, out_channels: int) -> MessagePassing:
        return GCNConv(in_channels, out_channels)
