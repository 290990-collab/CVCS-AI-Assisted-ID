# src/graph/graph_builder.py

"""
Adapter da RoomMeta (metadati .mat di RPLAN) a grafo PyTorch Geometric.

NON legge i .mat: si appoggia a `src.data.rplan_metadata.load_metadata`, che
restituisce un `RoomMeta` gia' pronto (composizione, topologia, geometria).
Qui il lavoro e' solo *tradurre* quella struttura in un `torch_geometric.data.Data`
che gli encoder GNN (GCN/GAT/GraphSAGE) possono consumare.

Mappatura RoomMeta -> Data
--------------------------
- room_types  -> feature dei nodi: one-hot a 13 dim (il tipo di stanza).
- boxes       -> feature dei nodi: geometria per stanza (centro, w, h, area, aspect),
                 normalizzata sulla griglia 256 (mai sul footprint -> vedi nota assi).
- edges       -> edge_index (i,j) + edge_attr (tipo di relazione, one-hot a 10 dim).
                 Il grafo e' NON diretto: gli archi vengono simmetrizzati.
- footprint   -> feature a livello di grafo (area/aspect dell'appartamento).
- type_histogram -> feature a livello di grafo (conteggio stanze per tipo), utile
                 come descrittore per la baseline training-free.

Nota sugli assi (trappola nota di RPLAN)
----------------------------------------
`boxes` (da gtBoxNew) e' [x0,y0,x1,y1]; `footprint` (da gtBox) ha gli assi
scambiati [y0,x0,y1,x1]. Per evitare del tutto il rischio, la geometria dei nodi
viene normalizzata sulla griglia fissa 256 usando SOLO `boxes`. Del footprint si
usano solo area e aspect ratio, che sono invarianti allo scambio dei due assi.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from src.data.rplan_metadata import (
    NUM_ROOM_TYPES,
    RoomMeta,
    load_metadata,
)

# Numero di tipi di relazione nella 3a colonna di rEdge (0..9).
NUM_RELATION_TYPES = 10

# Griglia su cui sono espresse le coordinate RPLAN (coerente con rplan_metadata).
_GRID = 256.0

# Dimensione delle feature geometriche per nodo:
# [cx, cy, w, h, area, aspect] -> 6.
_NODE_GEOM_DIM = 6

# Dimensione totale della feature di un nodo (one-hot tipo + geometria).
NODE_FEATURE_DIM = NUM_ROOM_TYPES + _NODE_GEOM_DIM


def _node_features(meta: RoomMeta) -> torch.Tensor:
    """Costruisce la matrice delle feature dei nodi x = [N, NODE_FEATURE_DIM].

    Ogni riga (una stanza) e' la concatenazione di:
    - one-hot del tipo di stanza (13 dim);
    - geometria normalizzata sulla griglia 256: centro (cx, cy), larghezza,
      altezza, area, aspect ratio.

    Input:  meta -> RoomMeta della pianta.
    Output: tensore float [N, NODE_FEATURE_DIM].
    """
    num_nodes = meta.num_rooms
    x = torch.zeros(num_nodes, NODE_FEATURE_DIM, dtype=torch.float)

    for i, (room_type, box) in enumerate(zip(meta.room_types, meta.boxes)):
        # --- parte semantica: one-hot del tipo (colonne 0..12) ---
        x[i, room_type] = 1.0

        # --- parte geometrica: normalizzata su griglia 256 (colonne 13..18) ---
        x0, y0, x1, y1 = box
        w = max(x1 - x0, 0) / _GRID
        h = max(y1 - y0, 0) / _GRID
        cx = ((x0 + x1) / 2.0) / _GRID
        cy = ((y0 + y1) / 2.0) / _GRID
        area = w * h
        # aspect >= 1 e simmetrico all'orientamento; guardia contro lati nulli.
        long_side, short_side = max(w, h), max(min(w, h), 1e-6)
        aspect = long_side / short_side

        geom_start = NUM_ROOM_TYPES
        x[i, geom_start:geom_start + _NODE_GEOM_DIM] = torch.tensor(
            [cx, cy, w, h, area, aspect], dtype=torch.float
        )

    return x


def _edge_index_and_attr(
    meta: RoomMeta,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Costruisce edge_index [2, E'] e edge_attr [E', NUM_RELATION_TYPES].

    Gli archi di RPLAN sono diretti (una tripla per adiacenza); qui il grafo e'
    non diretto, quindi vengono simmetrizzati con `to_undirected`. La relazione
    (0..9) diventa un one-hot come attributo d'arco.

    Input:  meta -> RoomMeta della pianta.
    Output: (edge_index long [2, E'], edge_attr float [E', NUM_RELATION_TYPES]).
            Se la pianta non ha archi, ritorna tensori vuoti coerenti.
    """
    num_nodes = meta.num_rooms

    # Pianta senza archi: edge_index/edge_attr vuoti ma con la forma giusta.
    if len(meta.edges) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, NUM_RELATION_TYPES, dtype=torch.float)
        return edge_index, edge_attr

    sources: list[int] = []
    targets: list[int] = []
    relations: list[int] = []
    for i, j, relation in meta.edges:
        sources.append(i)
        targets.append(j)
        relations.append(relation)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    # One-hot della relazione (clamp per robustezza se comparissero id fuori range).
    relation_ids = torch.tensor(relations, dtype=torch.long).clamp(
        0, NUM_RELATION_TYPES - 1
    )
    edge_attr = F.one_hot(relation_ids, num_classes=NUM_RELATION_TYPES).float()

    # Simmetrizzazione: aggiunge la direzione opposta; archi paralleli con lo
    # stesso attributo vengono fusi con la media (reduce='mean').
    edge_index, edge_attr = to_undirected(
        edge_index, edge_attr, num_nodes=num_nodes, reduce="mean"
    )
    return edge_index, edge_attr


def build_graph(meta: RoomMeta) -> Data:
    """Traduce un RoomMeta in un grafo PyG `Data`.

    Oltre a x/edge_index/edge_attr allega attributi a livello di grafo utili in
    valutazione e per la baseline training-free (name/split/num_rooms/footprint/
    type_histogram). Questi non entrano nel message passing.

    Input:  meta -> RoomMeta della pianta.
    Output: torch_geometric.data.Data.
    """
    x = _node_features(meta)
    edge_index, edge_attr = _edge_index_and_attr(meta)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # --- attributi a livello di grafo (metadati, non feature di message passing) ---
    data.name = meta.name
    data.split = meta.split
    data.num_rooms = meta.num_rooms
    # Descrittori globali dell'appartamento (simmetrici allo scambio d'assi).
    data.footprint_area = float(meta.footprint_area)
    data.footprint_aspect = float(meta.footprint_aspect)
    # Istogramma dei tipi (conteggio per tipo): base della baseline training-free.
    data.type_histogram = torch.tensor(
        meta.type_histogram, dtype=torch.float
    ).unsqueeze(0)

    return data


def build_graph_from_png(
    png_path: str | Path,
) -> Data | None:
    """Costruisce il grafo della pianta corrispondente a un PNG della gallery.

    Input:  png_path -> path (o stem) del PNG.
    Output: Data, oppure None se il PNG non ha un record .mat collegabile
            (~0.1% dei casi, coerente con load_metadata).
    """
    meta = load_metadata(png_path)
    if meta is None:
        return None
    return build_graph(meta)
