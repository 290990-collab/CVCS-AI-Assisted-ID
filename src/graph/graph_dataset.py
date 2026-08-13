# src/graph/graph_dataset.py

"""
Dataset PyG dei grafi RPLAN, con cache su disco.

Il collo di bottiglia non e' costruire il grafo (e' immediato), ma leggere i tre
.mat aggregati di RPLAN: ~35 s ogni volta che parte un processo. Rifarlo a ogni
epoca di training sarebbe puro spreco, quindi qui i grafi vengono costruiti UNA
volta e serializzati in un unico file (`processed/graphs.pt`), che poi viene
caricato in RAM: 67k grafi da ~7 stanze occupano pochi MB.

I grafi salvati sono GREZZI, esattamente come li produce `graph_builder`:
nessuna normalizzazione delle feature, nessun ritocco della topologia. E' una
scelta voluta -- la cache resta una sola e serve tutte le varianti; eventuali
trasformazioni si applicano al volo passando `transform` (meccanismo nativo PyG),
senza rigenerare nulla. Stessa logica dell'embedding raw sul ramo vision.

Lo split ufficiale RPLAN e' gia' dentro ogni grafo (`data.split`), quindi il
filtro per split e' un'operazione in memoria: NON fa riprocessare la cache.

Uso:
    from src.graph.graph_dataset import RplanGraphDataset

    full  = RplanGraphDataset()                 # tutti i grafi (gallery)
    train = RplanGraphDataset(split="train")    # solo le piante di train
    loader = DataLoader(train, batch_size=64, shuffle=True)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset

from src.graph.graph_builder import build_graph_from_png

# PNG della gallery: stessa cartella usata dal ramo vision.
DEFAULT_SNAPSHOT_DIR = Path(
    "/work/cvcs2026/ai_interior_design/datasets/RPLAN/Interface/static/Data/snapshot_train"
)

# Radice della cache: simmetrica a embeddings/vision/ (ed e' gia' in .gitignore).
# PyG ci crea dentro raw/ e processed/graphs.pt.
DEFAULT_ROOT = Path("embeddings/graph/rplan")

# Split ufficiali RPLAN ammessi dal filtro.
VALID_SPLITS = ("train", "valid", "test")


class RplanGraphDataset(InMemoryDataset):
    """Grafi RPLAN costruiti dai PNG della gallery, cachati su disco.

    Args:
        root:         cartella della cache (dentro ci finiscono raw/ e processed/).
        snapshot_dir: cartella dei PNG; il nome di ogni file e' la chiave verso
                      il record .mat corrispondente.
        split:        se dato ("train"|"valid"|"test"), il dataset espone solo le
                      piante di quello split. Il filtro avviene in memoria dopo il
                      caricamento: la cache su disco resta unica e completa.
        transform:    trasformazione PyG applicata al volo a ogni grafo letto
                      (None = grafi grezzi).
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
        split: str | None = None,
        transform=None,
    ) -> None:
        if split is not None and split not in VALID_SPLITS:
            raise ValueError(f"split {split!r} non valido (usa {VALID_SPLITS} o None)")

        # Serve a process(): va impostato PRIMA di super().__init__, perche' e'
        # super() a decidere se la cache va (ri)generata e quindi a chiamare process().
        self.snapshot_dir = Path(snapshot_dir)
        self.split = split

        super().__init__(root=str(root), transform=transform)
        self.load(self.processed_paths[0])

        # Filtro per split: seleziona gli indici, senza toccare la cache.
        if split is not None:
            self._indices = self.split_indices(split)

    # ---- contratto InMemoryDataset ----------------------------------------

    @property
    def raw_file_names(self) -> list[str]:
        """Nessun raw da scaricare: i PNG e i .mat vivono gia' fuori dal root."""
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Unico file di cache con tutti i grafi collati."""
        return ["graphs.pt"]

    def download(self) -> None:
        """Niente da scaricare: il dataset RPLAN e' gia' sul filesystem condiviso."""
        pass

    def process(self) -> None:
        """Costruisce i grafi da tutti i PNG della gallery e li serializza.

        Viene chiamato da PyG solo se `processed/graphs.pt` non esiste. Le piante
        senza record .mat collegabile (~0.1%) vengono saltate.

        Side effects: scrive `processed/graphs.pt`.
        """
        png_paths = sorted(self.snapshot_dir.glob("*.png"))
        if not png_paths:
            raise FileNotFoundError(f"nessun PNG in {self.snapshot_dir}")

        graphs = []
        skipped = 0
        for png_path in png_paths:
            data = build_graph_from_png(png_path)
            # Un PNG senza record .mat non e' un errore: e' lo 0.1% noto.
            if data is None:
                skipped += 1
                continue
            graphs.append(data)

        print(
            f"[RplanGraphDataset] {len(graphs)} grafi costruiti "
            f"({skipped} PNG senza record .mat, saltati)"
        )
        self.save(graphs, self.processed_paths[0])

    # ---- accessori --------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Nome (stem del PNG) di ogni grafo, nell'ordine del dataset.

        E' la mappa indice -> pianta: serve a risalire dai risultati FAISS alla
        pianta corrispondente.
        """
        return [self[i].name for i in range(len(self))]

    def split_indices(self, split: str) -> list[int]:
        """Indici (sulla cache completa) dei grafi appartenenti a uno split.

        Input:  split -> "train" | "valid" | "test".
        Output: lista di indici interi, riferiti al dataset NON filtrato.
        """
        if split not in VALID_SPLITS:
            raise ValueError(f"split {split!r} non valido (usa {VALID_SPLITS})")

        splits = self._data.split  # lista di stringhe, una per grafo collato
        return [i for i, s in enumerate(splits) if s == split]

    def __repr__(self) -> str:
        suffix = f", split={self.split}" if self.split else ""
        return f"RplanGraphDataset({len(self)} grafi{suffix})"
