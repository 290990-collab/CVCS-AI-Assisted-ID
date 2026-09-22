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

Coppie ASIMMETRICHE (`pair_mode="asym_partial"`, fase C.0 → opzione D, 10 set 2026)
-----------------------------------------------------------------------------------
Con il self-recovery sotto masking il graph crollava (status.md § 30): le coppie
simmetriche non insegnano mai che "pianta con meta' delle stanze" e "pianta
intera" sono lo stesso oggetto. In questo modo la coppia diventa quella della
head del vision: vista A = grafo **intero** (solo rotazioni e riflessioni), vista
B = lo stesso grafo con una frazione f ~ U[`partial_frac_min`, `partial_frac_max`]
di stanze **rimosse davvero** (sottografo indotto, archi rinumerati), come fa il
partial in valutazione (`graph_partial_query.make_partial_graph`: stesso
`round(f*n)`, feature dei superstiti identiche). Almeno una stanza resta sempre,
cosi' ogni grafo del batch sopravvive e le righe di InfoNCE restano allineate.
node_drop/edge_drop/feat_mask/geom_jitter NON si applicano in questa modalita':
la sola variabile che cambia e' la forma delle coppie. Default `symmetric` =
comportamento storico, bit per bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from src.data.rplan_metadata import NUM_ROOM_TYPES
from src.graph.transforms import LOST_MARKER_COL, lost_neighbor_count

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
    pair_mode: str = "symmetric"        # symmetric | asym_partial (opzione D)
    partial_frac_min: float = 0.25
    partial_frac_max: float = 0.75
    lost_marker: bool = False           # colonna "vicini persi" nella vista B (asymlost)

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
            pair_mode=getattr(cfg, "pair_mode", "symmetric"),
            partial_frac_min=getattr(cfg, "partial_frac_min", 0.25),
            partial_frac_max=getattr(cfg, "partial_frac_max", 0.75),
            lost_marker=getattr(cfg, "lost_marker", False),
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


def keep_subgraph(batch, keep: torch.Tensor, lost_marker: bool = False) -> Data:
    """Sottografo indotto dai nodi `keep` [N] bool: nodi tolti DAVVERO, archi fra
    superstiti rinumerati, `batch` filtrato. Le feature dei superstiti non cambiano.

    Con `lost_marker=True` (variante asymlost) la colonna LOST_MARKER_COL, che la
    transform ha riempito di zeri, riceve il numero di vicini distinti tolti,
    contato sul grafo INTERO prima del sottografo: stesso valore di
    `graph_partial_query.lost_marker_for` in valutazione.
    """
    if lost_marker:
        if batch.x.size(1) != LOST_MARKER_COL + 1:
            raise ValueError(
                f"keep_subgraph(lost_marker=True): x ha {batch.x.size(1)} colonne, "
                f"attese {LOST_MARKER_COL + 1} (manca AppendLostMarker nella transform?)"
            )
        count = lost_neighbor_count(batch.edge_index, keep)
    edge_attr = getattr(batch, "edge_attr", None)
    edge_index, edge_attr = subgraph(keep, batch.edge_index, edge_attr,
                                     relabel_nodes=True, num_nodes=batch.x.size(0))
    if lost_marker:
        x = batch.x[keep].clone()
        x[:, LOST_MARKER_COL] = count[keep].to(x.dtype)
        view = Data(x=x, edge_index=edge_index)
    else:
        view = Data(x=batch.x[keep], edge_index=edge_index)
    if edge_attr is not None:
        view.edge_attr = edge_attr
    node_graph = getattr(batch, "batch", None)
    if node_graph is None:
        node_graph = batch.x.new_zeros(batch.x.size(0), dtype=torch.long)
    view.batch = node_graph[keep]
    return view


def remove_rooms_view(batch, params: AugmentParams, generator: torch.Generator) -> Data:
    """Vista PARZIALE: per ogni grafo toglie round(f*n) stanze a caso, con
    f ~ U[partial_frac_min, partial_frac_max] estratta per grafo.

    Come `select_rooms_to_remove` (strategia `random`) del partial in valutazione:
    stesso arrotondamento (`torch.round` = `round` di Python, half-to-even). In
    piu' si lascia sempre almeno una stanza: un grafo vuoto sparirebbe dal pooling
    e disallineerebbe le righe di InfoNCE.
    """
    device = batch.x.device
    node_graph = getattr(batch, "batch", None)
    if node_graph is None:
        node_graph = batch.x.new_zeros(batch.x.size(0), dtype=torch.long)
    num_graphs = int(node_graph.max()) + 1
    counts = torch.bincount(node_graph, minlength=num_graphs)

    frac = params.partial_frac_min + (params.partial_frac_max - params.partial_frac_min) \
        * _rand(num_graphs, generator, device)
    n_remove = torch.round(frac * counts.float()).long()
    n_remove = torch.minimum(n_remove, counts - 1).clamp_min(0)

    # Rango casuale di ogni nodo DENTRO il suo grafo: si ordina per (grafo, u).
    u = _rand(node_graph.numel(), generator, device)
    order = torch.argsort(node_graph.double() * 2.0 + u.double())
    ptr = torch.cumsum(counts, 0) - counts                       # primo nodo di ogni grafo
    rank = torch.empty_like(node_graph)
    rank[order] = torch.arange(node_graph.numel(), device=device) - ptr[node_graph[order]]
    keep = rank >= n_remove[node_graph]
    # Il marcatore non estrae numeri casuali: a parita' di seed tolte le stesse
    # stanze della variante `asym`.
    return keep_subgraph(batch, keep, lost_marker=params.lost_marker)


def two_views(
    batch,
    params: AugmentParams,
    generator: torch.Generator,
    geom_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[Data, Data]:
    """Due viste aumentate dello stesso batch (coppia positiva).

    - `symmetric` (default): due viste indipendenti con le stesse augmentation.
    - `asym_partial` (opzione D): vista A = grafo intero con le sole simmetrie
      (flip/rot), vista B = grafo con una frazione di stanze rimosse.

    Input/Output: vedi `augment_view`; ritorna la coppia (vista_a, vista_b).
    """
    if params.lost_marker and params.pair_mode != "asym_partial":
        raise ValueError(
            f"lost_marker=True richiede pair_mode='asym_partial' (dato: '{params.pair_mode}')"
        )
    if params.pair_mode == "asym_partial":
        whole = AugmentParams(node_drop=0.0, edge_drop=0.0, feat_mask=0.0, geom_jitter=0.0,
                              flip_prob=params.flip_prob, rot_prob=params.rot_prob)
        return (augment_view(batch, whole, generator, geom_stats),
                remove_rooms_view(batch, params, generator))
    if params.pair_mode != "symmetric":
        raise ValueError(f"pair_mode='{params.pair_mode}' (attesi: symmetric | asym_partial)")
    return (
        augment_view(batch, params, generator, geom_stats),
        augment_view(batch, params, generator, geom_stats),
    )
