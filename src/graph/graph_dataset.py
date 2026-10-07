"""PyG dataset of RPLAN graphs with on-disk cache.

Graphs are built once (reading the three aggregated .mat files takes ~35 s) and
serialised to `processed/graphs.pt`. Cached graphs are raw, as produced by
`graph_builder`; variants are applied on the fly via `transform`. The official
split lives in `data.split`, so split filtering is in-memory and does not
rebuild the cache.

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

# gallery PNGs, shared with the vision branch
DEFAULT_SNAPSHOT_DIR = Path(
    "/work/cvcs2026/ai_interior_design/datasets/RPLAN/Interface/static/Data/snapshot_train"
)

# cache root; PyG creates raw/ and processed/graphs.pt inside
DEFAULT_ROOT = Path("embeddings/graph/rplan")

VALID_SPLITS = ("train", "valid", "test")


class RplanGraphDataset(InMemoryDataset):
    """RPLAN graphs built from gallery PNGs, cached on disk.

    Args:
        root: cache folder.
        snapshot_dir: PNG folder; file name is the key to the .mat record.
        split: "train"|"valid"|"test"; in-memory filter, the cache stays complete.
        transform: PyG transform applied on read (None = raw graphs).
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

        # needed by process(); must be set before super().__init__, which may call it
        self.snapshot_dir = Path(snapshot_dir)
        self.split = split

        super().__init__(root=str(root), transform=transform)
        self.load(self.processed_paths[0])

        if split is not None:
            self._indices = self.split_indices(split)

    # --- InMemoryDataset contract ---

    @property
    def raw_file_names(self) -> list[str]:
        """No raw files."""
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Single cache file with all collated graphs."""
        return ["graphs.pt"]

    def download(self) -> None:
        """No download."""
        pass

    def process(self) -> None:
        """Build graphs from all gallery PNGs (skipping those without .mat) and write `processed/graphs.pt`."""
        png_paths = sorted(self.snapshot_dir.glob("*.png"))
        if not png_paths:
            raise FileNotFoundError(f"nessun PNG in {self.snapshot_dir}")

        graphs = []
        skipped = 0
        for png_path in png_paths:
            data = build_graph_from_png(png_path)
            if data is None:
                skipped += 1
                continue
            graphs.append(data)

        print(
            f"[RplanGraphDataset] {len(graphs)} grafi costruiti "
            f"({skipped} PNG senza record .mat, saltati)"
        )
        self.save(graphs, self.processed_paths[0])

    # --- accessors ---

    @property
    def names(self) -> list[str]:
        """PNG stem of each graph, in dataset order."""
        return [self[i].name for i in range(len(self))]

    def split_indices(self, split: str) -> list[int]:
        """Indices of a split's graphs, relative to the unfiltered dataset."""
        if split not in VALID_SPLITS:
            raise ValueError(f"split {split!r} non valido (usa {VALID_SPLITS})")

        splits = self._data.split  # one string per collated graph
        return [i for i, s in enumerate(splits) if s == split]

    def __repr__(self) -> str:
        suffix = f", split={self.split}" if self.split else ""
        return f"RplanGraphDataset({len(self)} grafi{suffix})"
