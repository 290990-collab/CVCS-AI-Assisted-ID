# src/graph/graph_partial_query.py

"""
Query di grafo DEGRADATA — il gemello grafo del partial del ramo vision (fase **C.0**).

Perche' serve
-------------
Dal 24 ago «migliore» = **piu' robusto** sotto masking (`status.md §23`), e la
misura e' l'AUC della curva di self-recovery. Il ramo graph non aveva alcun
partial: il criterio non gli si applicava, la sua config restava provvisoria
(B.7) e la fase D non aveva il suo vincolo d'ingresso. Questo modulo produce il
pezzo mancante: da una pianta e una strategia di masking, il grafo della pianta
**incompleta**.

La scelta di protocollo che rende il confronto appaiato
------------------------------------------------------
La selezione delle stanze da togliere **non viene reimplementata**: si importa
`select_rooms_to_remove` dal ramo vision. Il progetto di norma non incrocia i due
rami, ma qui l'oggetto condiviso *e' il protocollo*: se le due implementazioni
divergessero anche di una riga, i due rami toglierebbero stanze diverse e il
confronto sotto masking misurerebbe il masking, non i modelli. Una funzione pura
importata e' piu' sicura di una copia che sembra uguale.

⚠️ **Il pairing richiede B.3.** Il seed per-query e' `seed + qi`, con `qi` la riga
della gallery: solo con la gallery condivisa (ordine canonico, `gallery_join.py`)
la stessa pianta ha lo stesso `qi` nei due rami, quindi lo stesso `rng`, quindi
le **stesse stanze** rimosse.

Decisioni dichiarate
--------------------
- **Il footprint resta quello della pianta INTERA.** Nel ramo vision l'immagine
  degradata mantiene la tela e le stanze superstiti restano dove sono (le
  rimosse diventano bianco): il grafo fa lo stesso, non ricalcola l'ingombro
  sulle stanze rimaste. Ricalcolarlo sarebbe un altro esperimento (un crop con
  riscalatura), non questo.
- **Le feature dei nodi superstiti non cambiano**: sono normalizzate sulla
  griglia 256 fissa (`graph_builder._node_features`), non sul footprint. Togliere
  una stanza non perturba le altre — verificato in test.
- **Grafo vuoto**: se la strategia toglie tutte le stanze non esiste una query
  valida e si ritorna `None`. Il chiamante la conta e la salta, come il ramo
  vision conta le query in cui non è stato rimosso nulla.
"""

from __future__ import annotations

import random

import torch
from torch_geometric.data import Data

from src.data.rplan_metadata import RoomMeta
from src.graph.graph_builder import build_graph
from src.graph.transforms import lost_neighbor_count
from src.vision.data.vision_partial_query import select_rooms_to_remove


def filter_meta(meta: RoomMeta, removed: list[int] | set[int]) -> RoomMeta | None:
    """`RoomMeta` senza le stanze in `removed`, con gli archi rimappati.

    Gli archi che toccano una stanza rimossa spariscono (non esiste piu' la
    relazione); quelli fra due superstiti restano, con gli indici rinumerati
    sulla nuova posizione — altrimenti `edge_index` punterebbe a nodi sbagliati,
    che e' il modo silenzioso di valutare un grafo diverso da quello voluto.

    Returns:
        Il `RoomMeta` ridotto, o None se non resta nessuna stanza.
    """
    removed = set(removed)
    keep = [i for i in range(meta.num_rooms) if i not in removed]
    if not keep:
        return None

    remap = {old: new for new, old in enumerate(keep)}
    return RoomMeta(
        name=meta.name,
        split=meta.split,
        room_types=tuple(meta.room_types[i] for i in keep),
        edges=tuple(
            (remap[a], remap[b], rel)
            for a, b, rel in meta.edges
            if a in remap and b in remap
        ),
        boxes=tuple(meta.boxes[i] for i in keep),
        footprint=meta.footprint,        # invariato: vedi «Decisioni dichiarate»
        entrance=meta.entrance,
    )


def lost_marker_for(meta: RoomMeta, removed: list[int] | set[int]) -> torch.Tensor:
    """Numero di vicini DISTINTI tolti per ogni stanza superstite (asymlost).

    Si conta sul grafo INTERO (`build_graph(meta).edge_index`, gia' simmetrizzato:
    lo stesso edge_index che il training vede nel batch), non su `meta.edges`
    (diretti). Le righe seguono l'ordine crescente dell'indice originale, lo
    stesso di `filter_meta`.

    Returns: float [n_superstiti].
    """
    removed = set(removed)
    keep = torch.tensor([i not in removed for i in range(meta.num_rooms)], dtype=torch.bool)
    count = lost_neighbor_count(build_graph(meta).edge_index, keep)
    return count[keep].float()


def make_partial_graph(
    meta: RoomMeta,
    strategy: str,
    params: dict,
    rng: random.Random,
    lost_marker: bool = False,
) -> tuple[Data | None, list[int]]:
    """Grafo della pianta degradata + indici delle stanze rimosse.

    Stessa firma e stesso ordine di ritorno di `make_partial_query` del ramo
    vision (grafo al posto dell'immagine), cosi' i due loop di valutazione si
    leggono uguali.

    Args:
        meta:     `RoomMeta` della pianta intera.
        strategy: "random" | "semantic" | "topology" (le stesse del vision).
        params:   parametri della strategia (`fraction`, `keep_types`, `max_degree`).
        rng:      `random.Random` gia' seedato per-query (`seed + qi`).
        lost_marker: se True allega `graph.lost_marker` [n_superstiti] (vedi
                  `lost_marker_for`), che `AppendLostMarker` trasforma in colonna.

    Returns:
        (grafo, rimosse). Il grafo e' None se la strategia svuota la pianta.
    """
    removed = select_rooms_to_remove(meta, strategy, params, rng)
    reduced = filter_meta(meta, removed)
    if reduced is None:
        return None, removed
    graph = build_graph(reduced)
    if lost_marker:
        graph.lost_marker = lost_marker_for(meta, removed)
    return graph, removed
