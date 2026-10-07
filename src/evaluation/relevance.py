"""Architectural relevance between plans from the RPLAN .mat metadata, split by axis.

Three distinct signals in [0,1], used separately by the metrics:
- composition: same rooms? (Weighted Jaccard on the type histogram)
- topology:    same connections? (Weighted Jaccard on the typed adjacency)
- geometry:    same shape/proportions? (footprint + area distribution)

Distinct does not mean independent: measured on the `.mat` (`src/evaluation/metric_diagnostics.py`),
composition and topology gains correlate at ~0.7 and the third geometry component
(`type_area_distribution`) at ~0.5 with composition (same `rType`). "A wins on 2 axes out of 3" is
not two independent pieces of evidence.

Axes also saturate differently: geometry lives in a narrow band (p1..p99 ~ [0.71, 0.95]) because
`footprint_area` is normalised on the 256 grid but only spans ~0.19-0.68, so even a random ranking gets a
high nDCG there; read the raw score against the measured floor.

For the discrete axes (composition, topology) `GalleryAxes` also exposes the exact equivalence class:
plans with exactly the same histogram (or adjacency), the binary ground truth for Recall/mAP.
"""

from __future__ import annotations

import numpy as np

from src.data.rplan_metadata import NUM_ROOM_TYPES, RoomMeta

# relevance axes: the first two are discrete (exact equivalence classes for Recall/mAP), geometry is continuous (graded nDCG only)
AXES = ("composition", "topology", "geometry")
DISCRETE_AXES = ("composition", "topology")


# --- type-pair indexing for topology ---
# typed adjacency = ordered pair (min_type, max_type); 13 types -> 91 pairs, mapped on a dense vector
# so the topological Weighted Jaccard vectorises like the compositional one

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


# --- per-axis features of a single plan ---

def composition_vector(meta: RoomMeta) -> np.ndarray:
    """Room-type histogram, 13 counts."""
    return np.asarray(meta.type_histogram, dtype=np.float32)


def topology_vector(meta: RoomMeta) -> np.ndarray:
    """Typed adjacency as a dense vector of 91 counts (one per type pair); invariant to room permutation."""
    vec = np.zeros(NUM_PAIRS, dtype=np.float32)
    for (ti, tj), count in meta.typed_adjacency(include_relation=False).items():
        vec[_PAIR_INDEX[(ti, tj)]] = count
    return vec


# --- vectorised similarity over the whole gallery ---

def _weighted_jaccard_matrix(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Weighted Jaccard sum(min)/sum(max) between `query` and each row of `matrix`, shape (N,); 1.0 where the union is 0 (both empty)."""
    inter = np.minimum(matrix, query).sum(axis=1)
    union = np.maximum(matrix, query).sum(axis=1)
    sim = np.ones_like(union, dtype=np.float32)
    np.divide(inter, union, out=sim, where=union > 0)
    return sim


class GalleryAxes:
    """Per-axis features of the whole gallery + similarities / equivalence classes.

    Rows follow the order of `metas` (the FAISS gallery order). Plans without .mat (`None`) have null
    features, `valid[i] == False`, similarity 0 and are never relevant. A query is its row index `qi`;
    `*_sim` return an (N,) vector over the whole gallery, self included (excluding it is the caller's job).
    """

    def __init__(self, metas: list[RoomMeta | None]):
        n = len(metas)
        self.valid = np.array([m is not None for m in metas], dtype=bool)

        self.comp = np.zeros((n, NUM_ROOM_TYPES), dtype=np.float32)
        self.topo = np.zeros((n, NUM_PAIRS), dtype=np.float32)
        self.area = np.zeros(n, dtype=np.float32)
        self.aspect = np.ones(n, dtype=np.float32)        # default 1 keeps aspect_sim defined
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

    # --- graded similarity per axis (nDCG) ---

    def composition_sim(self, qi: int) -> np.ndarray:
        sim = _weighted_jaccard_matrix(self.comp, self.comp[qi])
        sim[~self.valid] = 0.0
        return sim

    def topology_sim(self, qi: int) -> np.ndarray:
        sim = _weighted_jaccard_matrix(self.topo, self.topo[qi])
        sim[~self.valid] = 0.0
        return sim

    def geometry_sim(self, qi: int) -> np.ndarray:
        """Mean of three components in [0,1]: area, aspect, per-type area distribution."""
        area_sim = 1.0 - np.abs(self.area - self.area[qi])
        aq = self.aspect[qi]
        aspect_sim = np.minimum(self.aspect, aq) / np.maximum(self.aspect, aq)
        dist_sim = 1.0 - 0.5 * np.abs(self.dist - self.dist[qi]).sum(axis=1)
        sim = (area_sim + aspect_sim + dist_sim) / 3.0
        sim[~self.valid] = 0.0
        return sim.astype(np.float32)

    def sim(self, axis: str, qi: int) -> np.ndarray:
        """Dispatch by axis name."""
        return {
            "composition": self.composition_sim,
            "topology": self.topology_sim,
            "geometry": self.geometry_sim,
        }[axis](qi)

    # --- exact equivalence classes (Recall/mAP, discrete axes) ---

    def composition_relevant(self, qi: int) -> np.ndarray:
        """Boolean mask: plans with a type histogram identical to the query."""
        mask = (self.comp == self.comp[qi]).all(axis=1) & self.valid
        return mask

    def topology_relevant(self, qi: int) -> np.ndarray:
        """Boolean mask: plans with typed adjacency identical to the query."""
        mask = (self.topo == self.topo[qi]).all(axis=1) & self.valid
        return mask

    def relevant(self, axis: str, qi: int) -> np.ndarray:
        """Relevance mask for a discrete axis (composition|topology)."""
        return {
            "composition": self.composition_relevant,
            "topology": self.topology_relevant,
        }[axis](qi)
