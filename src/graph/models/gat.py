# src/graph/models/gat.py

"""
Encoder GAT (Graph Attention Network) basato su GATv2Conv (Brody et al. 2021).

A differenza di GCN (isotropo), GAT impara un PESO DI ATTENZIONE per ogni arco:
un nodo puo' dare piu' importanza a certi vicini che ad altri. GATv2 usa
attenzione "dinamica" (piu' espressiva della GAT originale). E' l'unico dei tre
encoder che sfrutta `edge_attr`, cioe' il TIPO di relazione RPLAN (0..9): puo'
quindi pesare diversamente, ad es., una porta rispetto a un semplice confine.

Note di dimensione:
- `heads` teste di attenzione calcolate in parallelo; con `concat=False` vengono
  MEDIATE, cosi' l'uscita resta `out_channels` (compatibile con lo stack della
  base, che assume larghezza costante). Con concat=True l'uscita sarebbe
  heads*out_channels e romperebbe il layer successivo.
- `heads` ed `edge_dim` vanno impostati PRIMA di super().__init__, perche' e' la
  base a costruire i layer (chiamando build_conv) dentro il proprio __init__.
"""

from __future__ import annotations

from torch_geometric.nn import GATv2Conv, MessagePassing

from src.graph.graph_builder import NUM_RELATION_TYPES
from src.graph.models.base import BaseGraphEncoder


class GATEncoder(BaseGraphEncoder):
    """Encoder basato su GATv2Conv (con attenzione sugli archi tipizzati).

    Args (oltre a quelli di BaseGraphEncoder):
        heads:    numero di teste di attenzione (mediate, non concatenate).
        edge_dim: dimensione di edge_attr (10 = one-hot della relazione RPLAN).
        attn_dropout: dropout sui coefficienti di attenzione.
    """

    # GAT consuma il tipo di relazione sugli archi.
    uses_edge_attr = True

    def __init__(
        self,
        heads: int = 4,
        edge_dim: int = NUM_RELATION_TYPES,
        attn_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        # Impostati PRIMA di super().__init__: build_conv li legge durante la
        # costruzione dello stack (che avviene dentro BaseGraphEncoder.__init__).
        self.heads = heads
        self.edge_dim = edge_dim
        self.attn_dropout = attn_dropout
        super().__init__(**kwargs)

    def build_conv(self, in_channels: int, out_channels: int) -> MessagePassing:
        return GATv2Conv(
            in_channels,
            out_channels,
            heads=self.heads,
            concat=False,  # media delle teste -> uscita = out_channels
            edge_dim=self.edge_dim,
            dropout=self.attn_dropout,
        )
