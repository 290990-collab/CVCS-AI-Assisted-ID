# src/graph/transforms.py

"""
Trasformazioni PyG per l'adjustment del grafo prima della GNN.

I grafi in cache (`RplanGraphDataset`) sono GREZZI apposta: qui stanno le
correzioni che si applicano *al volo* col meccanismo `transform` di PyG, cosi'
una sola cache serve tutte le varianti e ogni pezzo si puo' accendere/spegnere
in ablation. Nulla di tutto questo va dentro `graph_builder`.

Perche' servono (misurato su 8k piante):

- NormalizeNodeGeometry -- le 6 colonne geometriche hanno scale incompatibili
  tra loro e con l'one-hot del tipo (che vale 0/1). In particolare `aspect`
  arriva a 37 su stanze degeneri, mentre `area` ha media ~0.05: un layer lineare
  inizializzato normalmente sarebbe dominato da `aspect` e ignorerebbe `area`.
  La z-score (media 0, dev.std 1 per colonna) mette tutte le feature sulla stessa
  scala. Le statistiche si calcolano SOLO sul train (mai valid/test) per non
  passare informazione dal test al preprocessing (leakage). Su `aspect` si fa
  prima un clip al percentile alto, perche' la coda estrema (37 = ~45 sigma)
  sposterebbe media e dev.std di tutte le stanze.

- RemoveSpuriousSelfLoops -- nei .mat esistono pochi archi (i,i) spurii (~46/8k).
  Vanno tolti perche' GCNConv/GATConv aggiungono i self-loop da soli: lasciarli
  darebbe al nodo peso doppio su se stesso.

NON si ricentra cx/cy: verificato che RPLAN centra ogni pianta nel canvas
(centro footprint = 0.500 +/- 0.001), quindi le coordinate misurano gia' la
posizione *dentro* l'appartamento, che e' informazione utile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.utils import remove_self_loops

from src.data.rplan_metadata import NUM_ROOM_TYPES

# Le 6 colonne geometriche [cx, cy, w, h, area, aspect] iniziano subito dopo
# l'one-hot del tipo. `aspect` e' l'ultima.
_GEOM_START = NUM_ROOM_TYPES
_GEOM_DIM = 6
_ASPECT_COL = _GEOM_START + _GEOM_DIM - 1  # colonna assoluta di aspect

# Piccolo epsilon per non dividere per zero se una colonna e' costante.
_EPS = 1e-6


class RemoveSpuriousSelfLoops(BaseTransform):
    """Rimuove gli archi (i, i) presenti nei .mat.

    Le convoluzioni (GCN/GAT) aggiungono i self-loop internamente, quindi quelli
    gia' presenti sarebbero duplicati con peso doppio. Toglie coerentemente anche
    la riga corrispondente di edge_attr.
    """

    def forward(self, data: Data) -> Data:
        data.edge_index, data.edge_attr = remove_self_loops(
            data.edge_index, data.edge_attr
        )
        return data


class NormalizeNodeGeometry(BaseTransform):
    """Standardizza (z-score) le 6 colonne geometriche dei nodi.

    Applica, in ordine:
    1. clip di `aspect` al valore `aspect_clip` (statistica robusta della coda);
    2. z-score per colonna: (x - mean) / std, con mean/std pre-calcolati sul
       SOLO train (vedi `compute_geometry_stats`).

    L'one-hot del tipo (colonne 0..12) non viene toccato: e' gia' su scala 0/1 e
    standardizzarlo ne distruggerebbe l'interpretazione.

    Args:
        mean:        media per colonna, shape [6] (ordine cx,cy,w,h,area,aspect).
        std:         dev.std per colonna, shape [6].
        aspect_clip: tetto applicato ad `aspect` prima della z-score.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        aspect_clip: float,
    ) -> None:
        self.mean = mean.view(1, _GEOM_DIM).float()
        # std con guardia: colonne costanti non devono generare divisioni per 0.
        self.std = std.view(1, _GEOM_DIM).float().clamp_min(_EPS)
        self.aspect_clip = float(aspect_clip)

    def forward(self, data: Data) -> Data:
        geom = data.x[:, _GEOM_START:_GEOM_START + _GEOM_DIM].clone()

        # 1. clip della coda di aspect (ultima colonna geometrica).
        geom[:, -1] = geom[:, -1].clamp_max(self.aspect_clip)

        # 2. z-score per colonna.
        geom = (geom - self.mean) / self.std

        data.x = data.x.clone()
        data.x[:, _GEOM_START:_GEOM_START + _GEOM_DIM] = geom
        return data


def compute_geometry_stats(
    train_dataset,
    aspect_percentile: float = 99.0,
) -> dict[str, np.ndarray]:
    """Calcola media/dev.std delle 6 colonne geometriche sul solo train.

    Va invocata UNA volta con il dataset ristretto allo split train. Le
    statistiche cosi' ottenute vengono poi riusate per normalizzare train, valid
    e test (nessun leakage: valid/test non contribuiscono alle statistiche).

    L'ordine e' importante: `aspect` viene prima clippato al `aspect_percentile`,
    poi si calcolano mean/std sui dati clippati -- coerente con cio' che fara'
    `NormalizeNodeGeometry` a runtime.

    Args:
        train_dataset:     RplanGraphDataset(split="train") (o iterable di Data).
        aspect_percentile: percentile a cui clippare `aspect` (default 99).
    Output: dict con chiavi "mean" [6], "std" [6], "aspect_clip" (scalare).
    """
    # Raccoglie tutte le righe geometriche del train in un unico array.
    rows = [
        data.x[:, _GEOM_START:_GEOM_START + _GEOM_DIM].numpy()
        for data in train_dataset
    ]
    geom = np.concatenate(rows, axis=0)  # [num_nodi_train, 6]

    # Tetto di aspect calcolato PRIMA di media/dev.std, poi applicato.
    aspect_clip = float(np.percentile(geom[:, -1], aspect_percentile))
    geom[:, -1] = np.minimum(geom[:, -1], aspect_clip)

    mean = geom.mean(axis=0)
    std = geom.std(axis=0)
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "aspect_clip": np.float32(aspect_clip),
    }


def save_geometry_stats(stats: dict[str, np.ndarray], path: str | Path) -> Path:
    """Serializza le statistiche geometriche su disco (.npz).

    Input:  stats -> dict da compute_geometry_stats; path -> file di uscita.
    Output: il Path scritto.
    Side effects: crea la cartella e scrive il file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stats)
    return path


def load_geometry_stats(path: str | Path) -> dict[str, np.ndarray]:
    """Ricarica le statistiche geometriche salvate con `save_geometry_stats`."""
    data = np.load(path)
    return {k: data[k] for k in ("mean", "std", "aspect_clip")}


def build_node_transform(
    normalize: bool = True,
    drop_self_loops: bool = True,
    stats: dict[str, np.ndarray] | None = None,
) -> BaseTransform | None:
    """Compone la trasformazione dei nodi secondo i flag di ablation.

    Ogni pezzo e' opzionale, cosi' si possono misurare i contributi
    separatamente (grezzo vs z-score vs +self-loop-removal, ...).

    Args:
        normalize:       se True applica NormalizeNodeGeometry (richiede `stats`).
        drop_self_loops: se True applica RemoveSpuriousSelfLoops.
        stats:           statistiche da compute/load_geometry_stats; obbligatorie
                         se normalize=True.
    Output: una trasformazione PyG (singola o Compose), oppure None se nessun
            pezzo e' attivo (dataset grezzo).
    Raises: ValueError se normalize=True ma stats mancano.
    """
    steps: list[BaseTransform] = []

    if drop_self_loops:
        steps.append(RemoveSpuriousSelfLoops())

    if normalize:
        if stats is None:
            raise ValueError("normalize=True richiede `stats` (mean/std/aspect_clip)")
        steps.append(
            NormalizeNodeGeometry(
                mean=torch.as_tensor(stats["mean"]),
                std=torch.as_tensor(stats["std"]),
                aspect_clip=float(stats["aspect_clip"]),
            )
        )

    if not steps:
        return None
    if len(steps) == 1:
        return steps[0]
    return Compose(steps)
