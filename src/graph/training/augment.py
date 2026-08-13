# src/graph/training/augment.py

"""
Augmentation di grafo per il training contrastivo (InfoNCE).

Il contrastivo self-supervised ha bisogno di **due viste** dello stesso grafo che
il modello deve riconoscere come vicine (positivi), contro le altre piante del
batch (negativi). Le viste si ottengono perturbando leggermente il grafo.

Quali invarianze stiamo insegnando (e quali sono giuste)
-------------------------------------------------------
Ogni augmentation dice al modello "questa perturbazione NON cambia la pianta".
Conviene quindi chiedersi se la ground truth di valutazione e' davvero invariante
a quella perturbazione, perche' insegnare un'invarianza falsa costa punteggio:

- **rotazione e riflessione -> invarianza VERA, la migliore che abbiamo.**
  Tutti e tre gli assi di rilevanza sono *esattamente* invarianti: la
  composizione e' un istogramma di tipi (non guarda le coordinate), la topologia
  e' l'adiacenza tipizzata (idem), e le tre componenti geometriche sono
  simmetriche (area del footprint, aspect = lato lungo/lato corto, distribuzione
  di area per tipo). Ruotare una pianta di 90 gradi non cambia *nessuna* label:
  il modello puo' solo guadagnarci.
- **jitter geometrico -> invarianza quasi vera.** Uno spostamento piccolo cambia
  la similarita' geometrica di pochissimo e non tocca composizione/topologia.
- **edge drop -> invarianza parziale.** Toglie adiacenze, che sono l'asse
  topologia: da tenere basso.
- **node drop -> invarianza FALSA per il retrieval full, vera per il partial.**
  Togliere una stanza cambia davvero la composizione, che e' un asse di
  rilevanza. Serve pero' a rendere il modello robusto alle query incomplete
  (e' la controparte grafo del masking del ramo vision), quindi resta - ma va
  dosato sapendo che sul full retrieval lavora *contro* l'asse composizione.
- **feature mask -> dipende da cosa maschera.** Sulle colonne geometriche e'
  innocuo; sull'one-hot del tipo insegna a ignorare il tipo di stanza, che di
  nuovo e' l'asse composizione.

⚠️ Su grafi RPLAN (4-8 nodi) le probabilita' vanno lette con attenzione: con 5
nodi e `node_drop=0.1` la probabilita' che NESSUN nodo venga tolto e' 0.9^5 ~
59%, quindi nella maggioranza dei casi le due viste hanno struttura identica e
il compito contrastivo diventa quasi banale. E' uno dei motivi per cui il
training saturava in poche epoche.

Scelta di design importante (allineamento dei positivi): le viste **mantengono il
numero di nodi e il vettore `batch`** (i nodi "tolti" vengono azzerati e isolati,
non rimossi). Cosi' entrambe le viste producono esattamente gli stessi B grafi
nello stesso ordine -> le righe di InfoNCE restano allineate. Con pooling `add`
(default) un nodo azzerato e isolato contribuisce 0, quindi equivale a rimuoverlo.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from src.data.rplan_metadata import NUM_ROOM_TYPES

# Le 6 colonne geometriche [cx, cy, w, h, area, aspect] seguono l'one-hot del tipo
# (coerente con graph_builder._node_features e transforms._GEOM_START).
_GEOM_START = NUM_ROOM_TYPES
_GEOM_DIM = 6
_CX, _CY, _W, _H = 0, 1, 2, 3   # indici DENTRO il blocco geometrico


@dataclass(frozen=True)
class AugmentParams:
    """Intensita' delle augmentation (una vista).

    Args:
        node_drop:   frazione attesa di nodi azzerati/isolati.
        edge_drop:   frazione attesa di archi (superstiti) rimossi.
        feat_mask:   probabilita' di azzerare una singola cella (nodo, colonna).
                     ⚠️ per-cella, non piu' per-colonna-globale: vedi `augment_view`.
        geom_jitter: dev.std del rumore gaussiano sulle 6 colonne geometriche,
                     nelle unita' in cui arrivano (con `normalize=True` sono
                     z-score, quindi 0.1 = 10% di una deviazione standard).
        flip_prob:   probabilita' di riflettere la pianta (asse verticale).
        rot_prob:    probabilita' di ruotarla di 90/180/270 gradi (k estratto
                     uniformemente tra 1, 2, 3).
    """

    node_drop: float = 0.1
    edge_drop: float = 0.1
    feat_mask: float = 0.1
    geom_jitter: float = 0.0
    flip_prob: float = 0.0
    rot_prob: float = 0.0

    @classmethod
    def from_cfg(cls, cfg) -> "AugmentParams":
        """Costruisce i parametri dalla config del training (Namespace/oggetto)."""
        return cls(
            node_drop=cfg.node_drop,
            edge_drop=cfg.edge_drop,
            feat_mask=cfg.feat_mask,
            geom_jitter=getattr(cfg, "geom_jitter", 0.0),
            flip_prob=getattr(cfg, "flip_prob", 0.0),
            rot_prob=getattr(cfg, "rot_prob", 0.0),
        )

    @property
    def uses_geometry(self) -> bool:
        """True se almeno una augmentation tocca il blocco geometrico."""
        return self.geom_jitter > 0.0 or self.flip_prob > 0.0 or self.rot_prob > 0.0


def _rand(shape, generator, device) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device=device)


def _apply_geometric_symmetry(
    geom: torch.Tensor,
    node_graph: torch.Tensor,
    num_graphs: int,
    params: AugmentParams,
    generator: torch.Generator,
) -> torch.Tensor:
    """Riflessione/rotazione della pianta, applicate in coordinate GREZZE.

    Le trasformazioni si estraggono **per grafo** (non per nodo): ruotare meta'
    stanze di una pianta non avrebbe senso. `node_graph` mappa nodo->grafo, quindi
    la scelta fatta per il grafo viene propagata a tutti i suoi nodi.

    Convenzioni (coordinate normalizzate sulla griglia 256, quindi in [0,1]):
    - riflessione: cx -> 1 - cx  (w, h, area, aspect invariati);
    - rotazione di 90 gradi: (cx, cy) -> (1 - cy, cx) e (w, h) -> (h, w);
      area = w*h e aspect = lato_lungo/lato_corto sono simmetrici, non cambiano.

    Input:  geom [N, 6] GREZZO, node_graph [N], num_graphs, params, generator.
    Output: geom [N, 6] trasformato.
    """
    device = geom.device

    # --- riflessione (una decisione per grafo) ---
    if params.flip_prob > 0.0:
        flip = (_rand(num_graphs, generator, device) < params.flip_prob)[node_graph]
        geom[flip, _CX] = 1.0 - geom[flip, _CX]

    # --- rotazione di k*90 gradi, k in {1,2,3} (una decisione per grafo) ---
    if params.rot_prob > 0.0:
        do_rot = _rand(num_graphs, generator, device) < params.rot_prob
        # k uniforme in {1,2,3} sui grafi selezionati, 0 sugli altri.
        k_graph = torch.randint(1, 4, (num_graphs,), generator=generator, device=device)
        k_graph = torch.where(do_rot, k_graph, torch.zeros_like(k_graph))
        k_node = k_graph[node_graph]

        # Applica la rotazione elementare fino a 3 volte, ogni giro solo ai nodi
        # a cui ne restano da fare: piu' leggibile che precalcolare le 4 matrici.
        for _ in range(3):
            todo = k_node > 0
            if not bool(todo.any()):
                break
            cx = geom[todo, _CX].clone()
            cy = geom[todo, _CY].clone()
            geom[todo, _CX] = 1.0 - cy
            geom[todo, _CY] = cx
            w = geom[todo, _W].clone()
            h = geom[todo, _H].clone()
            geom[todo, _W] = h
            geom[todo, _H] = w
            k_node = k_node - todo.long()

    return geom


def augment_view(
    batch,
    params: AugmentParams,
    generator: torch.Generator,
    geom_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> Data:
    """Produce UNA vista aumentata di un batch di grafi.

    Input:
        batch:      `Batch` PyG con x, edge_index, (edge_attr,) batch.
        params:     intensita' delle augmentation (vedi `AugmentParams`).
        generator:  torch.Generator per riproducibilita'.
        geom_stats: (mean, std) delle 6 colonne geometriche, ciascuno [1,6], o
                    None se il dataset e' grezzo (`--no-normalize`). Servono SOLO
                    a riflessione/rotazione: le simmetrie sono definite sulle
                    coordinate grezze in [0,1], quindi con feature z-scorate si
                    de-normalizza, si trasforma e si ri-normalizza. ⚠️ Non e' un
                    dettaglio pedante: su RPLAN std(cy)/std(cx) = 1.24 e
                    std(h)/std(w) = 1.26 (le piante non sono isotrope), quindi
                    scambiare direttamente le colonne z-scorate falserebbe la
                    scala di circa il 25%.
    Output: un `Data` con x/edge_index/edge_attr perturbati e lo stesso `batch`
            (stesso numero di grafi, stesso ordine).
    """
    device = batch.x.device
    x = batch.x.clone()
    edge_index = batch.edge_index
    edge_attr = getattr(batch, "edge_attr", None)

    num_nodes = x.size(0)
    num_edges = edge_index.size(1)

    # --- simmetrie geometriche + jitter (prima del node drop: operano sui valori) ---
    if params.uses_geometry:
        # `.clone()`: le simmetrie modificano in place, e senza clone il
        # comportamento dipenderebbe da quale ramo crea una copia. Meglio
        # lavorare sempre su una copia e riscrivere alla fine.
        geom = x[:, _GEOM_START:_GEOM_START + _GEOM_DIM].clone()

        node_graph = getattr(batch, "batch", None)
        if node_graph is None:
            node_graph = x.new_zeros(num_nodes, dtype=torch.long)

        if params.flip_prob > 0.0 or params.rot_prob > 0.0:
            # Le simmetrie sono definite sulle coordinate grezze in [0,1]. Se le
            # feature sono z-scorate si de-normalizza, si trasforma e si
            # ri-normalizza; con dataset grezzo (`--no-normalize`) si applicano
            # direttamente.
            if geom_stats is not None:
                mean, std = geom_stats
                geom = geom * std + mean
            geom = _apply_geometric_symmetry(
                geom, node_graph, int(node_graph.max()) + 1, params, generator
            )
            if geom_stats is not None:
                geom = (geom - mean) / std

        if params.geom_jitter > 0.0:
            noise = torch.randn(
                geom.shape, generator=generator, device=device
            ) * params.geom_jitter
            geom = geom + noise

        x[:, _GEOM_START:_GEOM_START + _GEOM_DIM] = geom

    # --- node drop: scegli i nodi da tenere, azzera gli altri ---
    keep_node = _rand(num_nodes, generator, device) >= params.node_drop
    x[~keep_node] = 0.0

    # --- archi: tieni solo quelli con ENTRAMBI gli estremi vivi ---
    edge_keep = keep_node[edge_index[0]] & keep_node[edge_index[1]]

    # --- edge drop: rimuovi in piu' una frazione degli archi superstiti ---
    if params.edge_drop > 0.0:
        edge_rand = _rand(num_edges, generator, device) >= params.edge_drop
        edge_keep = edge_keep & edge_rand

    # edge_index ed edge_attr si filtrano con la STESSA maschera -> restano allineati.
    new_edge_index = edge_index[:, edge_keep]
    new_edge_attr = edge_attr[edge_keep] if edge_attr is not None else None

    # --- feature mask: azzera singole celle (nodo, colonna) ---
    # ⚠️ Prima la maschera era per-COLONNA e condivisa da tutto il batch: azzerava
    # p.es. "la colonna bagno" per ogni pianta contemporaneamente, cioe' spostava
    # lo spazio in modo uniforme invece di creare una difficolta' discriminativa.
    # Per-cella ogni nodo perde informazione diversa -> vista davvero diversa.
    if params.feat_mask > 0.0:
        cell_keep = _rand(x.shape, generator, device) >= params.feat_mask
        x = x * cell_keep

    # Costruisce la vista: stesso vettore `batch` -> stessi B grafi, allineati.
    view = Data(x=x, edge_index=new_edge_index)
    if new_edge_attr is not None:
        view.edge_attr = new_edge_attr
    view.batch = batch.batch
    return view


def two_views(
    batch,
    params: AugmentParams,
    generator: torch.Generator,
    geom_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[Data, Data]:
    """Due viste aumentate indipendenti dello stesso batch (coppia positiva).

    Input/Output: vedi `augment_view`; ritorna la coppia (vista_a, vista_b).
    """
    return (
        augment_view(batch, params, generator, geom_stats),
        augment_view(batch, params, generator, geom_stats),
    )
