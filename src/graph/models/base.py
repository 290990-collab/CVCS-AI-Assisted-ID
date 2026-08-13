# src/graph/models/base.py

"""
Contratto comune agli encoder di grafo (GCN / GAT / GraphSAGE).

A differenza degli encoder vision (frozen, feature extraction pura), le GNN qui
si ADDESTRANO: `forward` deve poter propagare gradienti. Il contratto e' pero'
lo stesso spirito -- ogni encoder produce un embedding per grafo, L2-normalizzato,
da mandare a FAISS.

Design "un solo punto di variazione"
------------------------------------
GCN/GAT/GraphSAGE differiscono quasi solo nel TIPO di convoluzione. Quindi lo
stack di layer, il pooling nodi->grafo, la proiezione finale e la L2-norm vivono
tutti qui in `BaseGraphEncoder`; ogni sottoclasse implementa solo `build_conv`,
che restituisce UN layer di message passing del suo tipo. Aggiungere un encoder
nuovo = una classe di poche righe.

Perche' pochi layer (default 2): i grafi RPLAN hanno 4-8 nodi e diametro 2-3.
Con troppi hop tutti i nodi convergono alla stessa rappresentazione
(over-smoothing) e il grafo diventa indistinguibile. Due layer bastano a coprire
il vicinato utile.

Perche' il pooling conta: `add` somma i nodi -> preserva il CONTEGGIO delle
stanze (che e' un asse di rilevanza); `mean` lo normalizza via; `mean_max`
concatena media e massimo (piu' espressivo, raddoppia la dimensione).

Perche' esiste `raw_skip` (misurato, 28 lug 2026)
-------------------------------------------------
Un'ablation con pesi CASUALI ha mostrato che quasi tutto il punteggio di
retrieval viene dall'architettura, non dal training: il solo add-pool delle
feature grezze, senza alcuna rete, fa gia' nDCG medio 0.805 contro lo 0.862 del
modello allenato. Il motivo e' che le feature dei nodi sono
`one-hot del tipo (13) + geometria (6)`, quindi **sommarle E' l'istogramma dei
tipi** (l'asse composizione) piu' la geometria aggregata.

Il problema e' che la proiezione lineare *casuale* ne distrugge una parte
(composizione: 0.884 con l'add-pool grezzo, 0.850 dopo la proiezione casuale) e
il training deve poi ricostruirla, mentre InfoNCE la spinge nella direzione
opposta (allontana piante della stessa classe di equivalenza). Con `raw_skip`
l'add-pool grezzo viene **concatenato** all'embedding appreso prima della
proiezione: composizione e geometria arrivano all'uscita per costruzione e non
sono piu' degradabili dalla loss, e la capacita' della rete resta libera di
lavorare sulla topologia -- l'unico asse che il message passing deve davvero
imparare (+0.100 dal training, contro +0.056 e +0.012 degli altri due).

Default `False` per compatibilita' con i checkpoint gia' salvati (cambia la
forma di `proj`); si accende dai YAML, un asse di ablation alla volta.
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

# Pooling nodi -> grafo supportati.
POOLINGS = ("add", "mean", "max", "mean_max")


def global_pool(x: torch.Tensor, batch: torch.Tensor, pooling: str) -> torch.Tensor:
    """Aggrega le feature dei nodi in un vettore per grafo.

    Args:
        x:       feature dei nodi [num_nodi_batch, D].
        batch:   vettore [num_nodi_batch] che assegna ogni nodo al suo grafo.
        pooling: "add" | "mean" | "max" | "mean_max".
    Output: [num_grafi, D] (o [num_grafi, 2D] per "mean_max").
    """
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
    """Encoder di grafo: nodi -> embedding per grafo (L2-norm), addestrabile.

    Args:
        in_dim:     dimensione delle feature dei nodi in ingresso (19 per RPLAN).
        hidden_dim: dimensione nascosta dei layer di convoluzione.
        out_dim:    dimensione dell'embedding finale (quello indicizzato in FAISS).
        num_layers: numero di layer di message passing (2 = default, vedi modulo).
        pooling:    strategia di pooling nodi->grafo (vedi POOLINGS).
        dropout:    dropout tra i layer (regolarizzazione durante il training).
        raw_skip:   se True concatena l'add-pool delle feature GREZZE dei nodi
                    all'embedding poolato, prima della proiezione (vedi modulo).

    Le sottoclassi implementano SOLO `build_conv`; opzionalmente impostano
    `uses_edge_attr = True` se la loro convoluzione consuma `edge_attr`.
    """

    # Se True, forward passa edge_attr alla convoluzione (solo GAT lo usa).
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

        # Stack di convoluzioni: il primo layer porta da in_dim a hidden_dim, i
        # successivi restano a hidden_dim. `build_conv` decide il tipo (GCN/GAT/SAGE).
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in = in_dim if layer_idx == 0 else hidden_dim
            self.convs.append(self.build_conv(layer_in, hidden_dim))

        # Il pooling "mean_max" raddoppia la larghezza in ingresso alla proiezione.
        pooled_dim = hidden_dim * (2 if pooling == "mean_max" else 1)
        # La skip aggiunge le in_dim feature grezze aggregate (add-pool).
        if raw_skip:
            pooled_dim += in_dim

        # Testa di proiezione: dallo spazio poolato all'embedding finale.
        self.proj = nn.Linear(pooled_dim, out_dim)

    @abstractmethod
    def build_conv(self, in_channels: int, out_channels: int) -> MessagePassing:
        """Costruisce UN layer di message passing del tipo dell'encoder.

        Input:  in_channels, out_channels.
        Output: un layer PyG (GCNConv | GATv2Conv | SAGEConv).
        """
        raise NotImplementedError

    def forward(self, data) -> torch.Tensor:
        """Nodi -> embedding per grafo, L2-normalizzato.

        Input:  data -> `Data`/`Batch` PyG con x, edge_index, (edge_attr,) batch.
        Output: embeddings [num_grafi, out_dim], norma 1 per riga.
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        # `batch` mappa nodo->grafo; assente se e' un singolo grafo non collato.
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # Message passing: ReLU + dropout tra i layer, non dopo l'ultimo.
        last = len(self.convs) - 1
        for i, conv in enumerate(self.convs):
            if self.uses_edge_attr:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            if i < last:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Nodi -> grafo -> proiezione -> L2-norm (per usare la cosine similarity
        # come inner product in FAISS, coerente col ramo vision).
        graph_emb = global_pool(x, batch, self.pooling)

        # Skip: l'add-pool delle feature GREZZE (`data.x`, non toccato dai conv
        # perche' `x` sopra e' un nome locale ri-assegnato). Con l'one-hot del
        # tipo questo E' l'istogramma delle stanze -> composizione e geometria
        # raggiungono l'uscita senza dipendere dai pesi appresi.
        if self.raw_skip:
            graph_emb = torch.cat(
                [graph_emb, global_add_pool(data.x, batch)], dim=-1
            )

        z = self.proj(graph_emb)
        return F.normalize(z, p=2, dim=-1)

    @property
    def embedding_dim(self) -> int:
        """Dimensione D dell'embedding prodotto da forward()."""
        return self._out_dim
