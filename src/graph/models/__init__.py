# src/graph/models/__init__.py

"""
Encoder di grafo intercambiabili per il retrieval.

Tutti rispettano il contratto BaseGraphEncoder (nodi -> embedding per grafo,
L2-norm), quindi sono interscambiabili senza toccare la pipeline FAISS. La
selezione avviene per nome tramite GRAPH_ENCODER_REGISTRY (via config YAML,
analogo al VisionModelManager del ramo vision).

I tre encoder (richiesti dai prof) coprono aggregatori diversi:
  - gcn   -> media pesata simmetrica dei vicini (isotropo, baseline).
  - gat   -> attenzione sugli archi; unico a usare edge_attr (tipo relazione).
  - sage  -> separa se' stesso dai vicini (preserva l'identita' del nodo/tipo).
"""

from src.graph.models.base import BaseGraphEncoder
from src.graph.models.gat import GATEncoder
from src.graph.models.gcn import GCNEncoder
from src.graph.models.graph_sage import GraphSAGEEncoder

# nome (usato nei config) -> classe encoder
GRAPH_ENCODER_REGISTRY = {
    "gcn": GCNEncoder,
    "gat": GATEncoder,
    "sage": GraphSAGEEncoder,
}


def build_graph_encoder(name: str, **kwargs) -> BaseGraphEncoder:
    """Istanzia l'encoder di grafo registrato sotto 'name'.

    Args:
        name:   chiave del registry ("gcn" | "gat" | "sage").
        kwargs: argomenti del costruttore (in_dim, hidden_dim, out_dim,
                num_layers, pooling, dropout, + specifici: heads/edge_dim per GAT,
                aggr per SAGE).
    Raises:
        ValueError: se `name` non e' un encoder registrato.
    """
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
