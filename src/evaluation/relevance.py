# src/evaluation/relevance.py

"""
Rilevanza *architettonica* tra piante, dai metadati .mat RPLAN, **scomposta per asse**.

La rilevanza non è un singolo numero, ma TRE segnali distinti, ciascuno in [0,1],
usati separatamente dalle metriche:
- COMPOSIZIONE: stesse stanze? (Weighted Jaccard sull'istogramma dei tipi)
- TOPOLOGIA:    stesse connessioni? (Weighted Jaccard sull'adiacenza tipizzata)
- GEOMETRIA:    stessa forma/proporzioni? (footprint + distribuzione di area)

⚠️ **Distinti non vuol dire indipendenti**, e la differenza conta quando si
leggono i risultati. Misurato sui `.mat` (vedi `src/evaluation/
metric_diagnostics.py`, che rifà il conto): i gain di composizione e topologia
correlano a ~0.7, e la terza componente della geometria (`type_area_distribution`)
correla a ~0.5 con la composizione, perché è costruita sugli stessi `rType`.
Conseguenza: "il sistema A vince su 2 assi su 3" NON sono due evidenze
indipendenti — un guadagno sulla composizione trascina gli altri due assi.

⚠️ Gli assi hanno anche **saturazione molto diversa**: la geometria vive di
fatto in una banda stretta (p1..p99 ≈ [0.71, 0.95]) perché `footprint_area` è
normalizzata sulla griglia 256 ma varia solo fra ~0.19 e ~0.68. È il motivo per
cui su quell'asse anche un ranking casuale prende un nDCG alto, e per cui il
punteggio grezzo va sempre letto contro il floor misurato.

Per gli assi DISCRETI (composizione, topologia) `GalleryAxes` espone anche la
**classe di equivalenza esatta**: l'insieme delle piante con *esattamente* lo
stesso istogramma (o la stessa adiacenza). È la ground truth binaria per
Recall/mAP, e la sua dimensione emerge dai dati.
"""

from __future__ import annotations

import numpy as np

from src.data.rplan_metadata import NUM_ROOM_TYPES, RoomMeta

# Assi di rilevanza. I primi due sono "discreti" (ammettono classi di
# equivalenza esatta -> ground truth per Recall/mAP); la geometria è continua
# (solo nDCG graduato).
AXES = ("composition", "topology", "geometry")
DISCRETE_AXES = ("composition", "topology")


# ----------------------------------------------------------------------
# Indicizzazione delle coppie di tipi per la topologia.
# L'adiacenza tipizzata è una coppia ORDINATA (min_tipo, max_tipo); con 13 tipi
# ci sono 91 coppie possibili. Le mappiamo su un vettore denso di 91
# interi, così il Weighted Jaccard topologico si vettorializza come quello
# composizionale.
# ----------------------------------------------------------------------

def _build_pair_index() -> dict[tuple[int, int], int]:
    idx: dict[tuple[int, int], int] = {}
    k = 0
    for i in range(NUM_ROOM_TYPES):
        for j in range(i, NUM_ROOM_TYPES):
            idx[(i, j)] = k
            k += 1
    return idx


_PAIR_INDEX = _build_pair_index()
NUM_PAIRS = len(_PAIR_INDEX)  # 91


# ----------------------------------------------------------------------
# Estrazione delle feature per-asse da una singola pianta.
# ----------------------------------------------------------------------

def composition_vector(meta: RoomMeta) -> np.ndarray:
    """Istogramma dei tipi di stanza, vettore di 13 conteggi."""
    return np.asarray(meta.type_histogram, dtype=np.float32)


def topology_vector(meta: RoomMeta) -> np.ndarray:
    """Adiacenza tipizzata come vettore denso di 91 conteggi (una entry per
    coppia di tipi). Invariante alla permutazione delle stanze."""
    vec = np.zeros(NUM_PAIRS, dtype=np.float32)
    for (ti, tj), count in meta.typed_adjacency(include_relation=False).items():
        vec[_PAIR_INDEX[(ti, tj)]] = count
    return vec


# ----------------------------------------------------------------------
# Similarità vettorializzata su tutta la gallery.
# ----------------------------------------------------------------------

def _weighted_jaccard_matrix(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Weighted Jaccard tra il vettore `query` e ogni riga di `matrix`.

    sum(min)/sum(max) riga per riga, in [0,1]. Dove l'unione è 0 (entrambi i
    multiset vuoti su quell'asse) la similarità è 1.0 (concordi sul "niente").
    Ritorna un vettore (N,).
    """
    inter = np.minimum(matrix, query).sum(axis=1)
    union = np.maximum(matrix, query).sum(axis=1)
    sim = np.ones_like(union, dtype=np.float32)
    np.divide(inter, union, out=sim, where=union > 0)
    return sim


class GalleryAxes:
    """Feature per-asse di tutta la gallery + similarità/classi di equivalenza.

    Le righe sono allineate all'ordine di `metas` (che chi costruisce
    l'oggetto allinea all'ordine della gallery FAISS): la riga `i` corrisponde
    alla i-esima pianta. Le piante senza metadati .mat (`None`) hanno feature
    nulle e `valid[i] == False`: non contribuiscono come rilevanti e hanno
    similarità 0.

    Una query si identifica con il suo indice di riga `qi`. Tutti i metodi
    `*_sim` ritornano un vettore (N,) di similarità verso l'intera gallery
    (self incluso: l'esclusione del self è responsabilità del chiamante).
    """

    def __init__(self, metas: list[RoomMeta | None]):
        n = len(metas)
        self.valid = np.array([m is not None for m in metas], dtype=bool)

        self.comp = np.zeros((n, NUM_ROOM_TYPES), dtype=np.float32)
        self.topo = np.zeros((n, NUM_PAIRS), dtype=np.float32)
        self.area = np.zeros(n, dtype=np.float32)
        self.aspect = np.ones(n, dtype=np.float32)        # default 1 -> aspect_sim ben definito
        self.dist = np.zeros((n, NUM_ROOM_TYPES), dtype=np.float32)

        for i, m in enumerate(metas):
            if m is None:
                continue
            self.comp[i] = composition_vector(m)
            self.topo[i] = topology_vector(m)
            self.area[i] = m.footprint_area
            self.aspect[i] = m.footprint_aspect
            self.dist[i] = np.asarray(m.type_area_distribution, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.valid)

    # ---- similarità graduata per asse (per nDCG) --------------------------

    def composition_sim(self, qi: int) -> np.ndarray:
        sim = _weighted_jaccard_matrix(self.comp, self.comp[qi])
        sim[~self.valid] = 0.0
        return sim

    def topology_sim(self, qi: int) -> np.ndarray:
        sim = _weighted_jaccard_matrix(self.topo, self.topo[qi])
        sim[~self.valid] = 0.0
        return sim

    def geometry_sim(self, qi: int) -> np.ndarray:
        """Media di tre componenti (area, aspect, distribuzione area per tipo),
        ciascuna in [0,1]."""
        area_sim = 1.0 - np.abs(self.area - self.area[qi])
        aq = self.aspect[qi]
        aspect_sim = np.minimum(self.aspect, aq) / np.maximum(self.aspect, aq)
        dist_sim = 1.0 - 0.5 * np.abs(self.dist - self.dist[qi]).sum(axis=1)
        sim = (area_sim + aspect_sim + dist_sim) / 3.0
        sim[~self.valid] = 0.0
        return sim.astype(np.float32)

    def sim(self, axis: str, qi: int) -> np.ndarray:
        """Dispatch per nome d'asse."""
        return {
            "composition": self.composition_sim,
            "topology": self.topology_sim,
            "geometry": self.geometry_sim,
        }[axis](qi)

    # ---- classi di equivalenza esatta (per Recall/mAP, assi discreti) -----

    def composition_relevant(self, qi: int) -> np.ndarray:
        """Maschera booleana: piante con istogramma-tipo IDENTICO alla query."""
        mask = (self.comp == self.comp[qi]).all(axis=1) & self.valid
        return mask

    def topology_relevant(self, qi: int) -> np.ndarray:
        """Maschera booleana: piante con adiacenza tipizzata IDENTICA alla query."""
        mask = (self.topo == self.topo[qi]).all(axis=1) & self.valid
        return mask

    def relevant(self, axis: str, qi: int) -> np.ndarray:
        """Maschera dei rilevanti per un asse discreto (composition|topology)."""
        return {
            "composition": self.composition_relevant,
            "topology": self.topology_relevant,
        }[axis](qi)
