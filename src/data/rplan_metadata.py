# src/data/rplan_metadata.py

"""
Metadati strutturali delle piante RPLAN (dai file .mat) collegati ai PNG.

- COMPOSIZIONE (semantica): quali stanze ci sono e quante.
    -> `room_types`, `type_histogram`
- TOPOLOGIA (struttura): quali stanze sono collegate e con che relazione.
    -> `edges` (i, j, tipo_relazione), `typed_adjacency(...)`
- GEOMETRIA (forma/proporzioni): dimensioni, posizioni, footprint, ingresso.
    -> `boxes`, `footprint`, `entrance`, e le proprietà derivate
       (`footprint_area`, `footprint_aspect`, `type_area_distribution`).

Questi metadati servono in valutazione come "ground truth" contro cui misurare il ranking dell'encoder.

Campi .mat usati (forma per una pianta con R stanze):
- name      : scalare, id = stem del PNG.
- rType     : (R,)   tipo di ogni stanza (0..12).
- rEdge     : (E,3)  [stanza_i, stanza_j, tipo_relazione]; la 3a colonna (0..9)
              è il tipo di adiacenza.
- gtBoxNew  : (R,4)  bbox per stanza [xmin, ymin, xmax, ymax] su griglia 256.
- gtBox     : (R+1,4) bbox per stanza + 1 riga extra = bbox globale (footprint).
- boundary  : (P,4)  contorno esterno [x, y, direzione(0..3), flag_porta(0/1)];
              le righe con flag==1 marcano il segmento d'ingresso.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

# I tre split RPLAN sono file .mat aggregati (una struct array per file), non
# un file per pianta. Li uniamo tutti in un unico indice name -> metadati.
DEFAULT_MAT_DIR = Path(
    "/work/cvcs2026/ai_interior_design/datasets/RPLAN/Network/data"
)
_SPLITS = ("train", "valid", "test")

# Griglia su cui sono espresse tutte le coordinate RPLAN (boundary/box).
_GRID = 256.0

NUM_ROOM_TYPES = 13

ROOM_TYPES: dict[int, str] = {
    0: "LivingRoom", 1: "MasterRoom", 2: "Kitchen", 3: "Bathroom",
    4: "DiningRoom", 5: "ChildRoom", 6: "StudyRoom", 7: "SecondRoom",
    8: "GuestRoom", 9: "Balcony", 10: "Entrance", 11: "Storage", 12: "Wall-in",
}


@dataclass(frozen=True)
class RoomMeta:
    """Descrizione strutturale completa di una pianta (immutabile, hashabile)."""

    name: str
    # Split ufficiale RPLAN di provenienza ("train" | "valid" | "test").
    split: str
    # COMPOSIZIONE: un id-tipo per stanza.
    room_types: tuple[int, ...]
    # TOPOLOGIA: (stanza_i, stanza_j, tipo_relazione).
    edges: tuple[tuple[int, int, int], ...]
    # GEOMETRIA: bbox per stanza [xmin, ymin, xmax, ymax], allineata a room_types.
    boxes: tuple[tuple[int, int, int, int], ...]
    # GEOMETRIA: bbox globale dell'appartamento [xmin, ymin, xmax, ymax].
    footprint: tuple[int, int, int, int]
    # GEOMETRIA/FUNZIONALE: centro dell'ingresso normalizzato in [0,1]^2, o None.
    entrance: tuple[float, float] | None

    # ---- composizione -----------------------------------------------------

    @property
    def num_rooms(self) -> int:
        return len(self.room_types)

    @property
    def type_histogram(self) -> tuple[int, ...]:
        """Conteggio di stanze per ciascun tipo, vettore di lunghezza fissa 13, 
        direttamente confrontabile tra piante diverse con una distanza vettoriale.
        """
        counts = Counter(self.room_types)
        return tuple(counts.get(t, 0) for t in range(NUM_ROOM_TYPES))

    # ---- topologia --------------------------------------------------------

    def typed_adjacency(self, include_relation: bool = False) -> Counter:
        """Adiacenze a livello di *tipo* di stanza (invariante alla permutazione).

        Ogni arco (i, j) viene tradotto in una coppia di TIPI ordinata
        (min, max) così che "Cucina-Soggiorno" sia lo stesso item a prescindere
        da quale indice abbia ciascuna stanza in questa specifica pianta.

        Args:
            include_relation: se True include anche il tipo di relazione (0..9)
                nella tripla -> adiacenza più stretta. Se False (default) usa
                solo la coppia di tipi -> più robusta.
        """
        out: Counter = Counter()
        for i, j, rel in self.edges:
            ti, tj = self.room_types[i], self.room_types[j]
            pair = (min(ti, tj), max(ti, tj))
            key = (*pair, rel) if include_relation else pair
            out[key] += 1
        return out

    # ---- geometria --------------------------------------------------------

    @property
    def footprint_area(self) -> float:
        """Area del bounding box globale, normalizzata su griglia 256 -> [0,1]."""
        xmin, ymin, xmax, ymax = self.footprint
        return max(xmax - xmin, 0) * max(ymax - ymin, 0) / (_GRID * _GRID)

    @property
    def footprint_aspect(self) -> float:
        """Aspect ratio (lato lungo / lato corto) del footprint, >= 1."""
        xmin, ymin, xmax, ymax = self.footprint
        w, h = max(xmax - xmin, 1), max(ymax - ymin, 1)
        return max(w, h) / min(w, h)

    @property
    def type_area_distribution(self) -> tuple[float, ...]:
        """Frazione di area occupata da ciascun tipo-stanza, vettore di 13."""
        areas = [0.0] * NUM_ROOM_TYPES
        for t, (xmin, ymin, xmax, ymax) in zip(self.room_types, self.boxes):
            areas[t] += max(xmax - xmin, 0) * max(ymax - ymin, 0)
        total = sum(areas)
        if total <= 0:
            return tuple(areas)
        return tuple(a / total for a in areas)


def _parse_entrance(boundary: np.ndarray) -> tuple[float, float] | None:
    """Centro normalizzato del segmento d'ingresso (righe con flag==1)."""
    b = np.atleast_2d(boundary)
    if b.shape[1] < 4:
        return None
    door = b[b[:, 3] == 1]
    if door.size == 0:
        return None
    cx = float(door[:, 0].mean()) / _GRID
    cy = float(door[:, 1].mean()) / _GRID
    return (cx, cy)


@lru_cache(maxsize=1)
def _load_index(mat_dir: str) -> dict[str, RoomMeta]:
    """Legge i 3 .mat una sola volta e costruisce {name -> RoomMeta}"""
    
    index: dict[str, RoomMeta] = {}
    for split in _SPLITS:
        mat = sio.loadmat(
            Path(mat_dir) / f"data_{split}.mat",
            squeeze_me=True,
            struct_as_record=False,
        )
        for rec in mat["data"]:
            room_types = tuple(int(x) for x in np.atleast_1d(rec.rType))

            edge_rows = np.atleast_2d(rec.rEdge)
            edges = (
                tuple((int(a), int(b), int(rel)) for a, b, rel in edge_rows)
                if edge_rows.size
                else ()
            )

            # gtBoxNew: una bbox per stanza, allineata a room_types.
            box_rows = np.atleast_2d(rec.gtBoxNew)
            boxes = tuple(
                (int(x0), int(y0), int(x1), int(y1)) for x0, y0, x1, y1 in box_rows
            )

            # gtBox: stesse bbox per stanza + 1 riga extra = footprint globale.
            gtbox = np.atleast_2d(rec.gtBox)
            fp = gtbox[-1]
            footprint = (int(fp[0]), int(fp[1]), int(fp[2]), int(fp[3]))

            entrance = _parse_entrance(rec.boundary)

            index[str(rec.name)] = RoomMeta(
                name=str(rec.name),
                split=split,
                room_types=room_types,
                edges=edges,
                boxes=boxes,
                footprint=footprint,
                entrance=entrance,
            )
    return index


def load_metadata(
    png_path: str | Path,
    mat_dir: str | Path = DEFAULT_MAT_DIR,
) -> RoomMeta | None:
    """
    Ritorna i metadati strutturali della pianta corrispondente a `png_path`,
    oppure None se quel PNG non ha un record .mat (0,1% non collegabile).

    Args:
        png_path: path (o stem) del PNG della gallery.
        mat_dir:  cartella dei .mat (default: dataset RPLAN condiviso).
    """
    stem = Path(png_path).stem
    return _load_index(str(mat_dir)).get(stem)


def get_split(
    png_path: str | Path,
    mat_dir: str | Path = DEFAULT_MAT_DIR,
) -> str | None:
    """
    Ritorna lo split ufficiale RPLAN ("train" | "valid" | "test") della pianta
    corrispondente a `png_path`, oppure None se il PNG non ha record .mat.

    Nota: `snapshot_train/` mescola tutti e tre gli split (il nome è fuorviante),
    quindi questo lookup è il modo corretto per separare le query val/test.
    """
    meta = load_metadata(png_path, mat_dir)
    return meta.split if meta is not None else None


def split_row_indices(
    paths: list[str | Path],
    split: str,
    mat_dir: str | Path = DEFAULT_MAT_DIR,
) -> list[int]:
    """
    Indici delle righe di `paths` che appartengono allo split RPLAN `split`.

    Serve a stimare statistiche (es. il whitening) sul solo train senza toccare
    la gallery: le righe indicizzate restano TUTTE, cambia solo l'insieme su cui
    si stima. Le piante senza record .mat non appartengono a nessuno split e
    vengono escluse.

    Args:
        paths:   path dei PNG nell'ordine delle righe (image_paths della gallery).
        split:   "train" | "valid" | "test".
        mat_dir: cartella dei .mat (default: dataset RPLAN condiviso).

    Returns:
        Lista crescente di indici di riga.
    """
    if split not in _SPLITS:
        raise ValueError(f"split {split!r} non valido (usa {_SPLITS})")
    return [i for i, p in enumerate(paths) if get_split(p, mat_dir) == split]
