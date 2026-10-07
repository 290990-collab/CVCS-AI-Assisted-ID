"""Interchangeable graph encoders for retrieval.

All follow the BaseGraphEncoder contract (nodes -> L2-normalised graph embedding) and
are selected by name via GRAPH_ENCODER_REGISTRY.
  - gcn: symmetric weighted neighbour mean (isotropic baseline).
  - gat: edge attention; only one using edge_attr.
  - sage: separates self from neighbours.
"""

from src.graph.models.base import BaseGraphEncoder
from src.graph.models.gat import GATEncoder
from src.graph.models.gcn import GCNEncoder
from src.graph.models.graph_sage import GraphSAGEEncoder

GRAPH_ENCODER_REGISTRY = {
    "gcn": GCNEncoder,
    "gat": GATEncoder,
    "sage": GraphSAGEEncoder,
}


def build_graph_encoder(name: str, **kwargs) -> BaseGraphEncoder:
    """Instantiate the registered graph encoder; raises ValueError if unknown."""
    if name not in GRAPH_ENCODER_REGISTRY:
        raise ValueError(
            f"Encoder di grafo '{name}' non supportato. "
            f"Disponibili: {sorted(GRAPH_ENCODER_REGISTRY)}"
        )
    return GRAPH_ENCODER_REGISTRY[name](**kwargs)


__all__ = [
    "BaseGraphEncoder",
    "GCNEncoder",
    "GATEncoder",
    "GraphSAGEEncoder",
    "GRAPH_ENCODER_REGISTRY",
    "build_graph_encoder",
]
