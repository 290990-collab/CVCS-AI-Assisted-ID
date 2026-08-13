# src/vision/data/projection_pairs.py

"""
Costruzione della cache di coppie positive per allenare la projection head (Fase 3).

Strategia A+ (self-supervised): il positivo di una pianta è la STESSA pianta
degradata (stanze rimosse) ed eventualmente aumentata (flip/rot90, simmetrie
valide per le planimetrie). Segnale = "è la stessa pianta sotto degrado", che è
DIVERSO dalla rilevanza per-asse su cui valutiamo -> niente circolarità, e allena
proprio la robustezza richiesta dal partial retrieval.

Output: `pairs.npz` nella cartella della feature-variante, con
  anchors   [M, D]    embedding frozen della pianta completa (dal cache raw)
  positives [M, V, D] embedding frozen di V viste degradate per pianta
allineati per riga. M = piante dello split di training (train+valid, test escluso).

Ottimizzazione: il rendering di una vista degradata dipende solo da (pianta, vista,
config di masking), NON dall'encoder -> le immagini renderizzate sono identiche per
tutti i (modello x pooling). Le cachiamo su disco una volta (`_render_cache/`) e le
riusiamo: il costo caro (flood-fill + dilatazione) si paga una sola volta, non 14.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.rplan_metadata import get_split, load_metadata
from src.vision.data.vision_partial_query import make_partial_query
from src.vision.models.vision_model_manager import VisionModelManager
from src.vision.utils.config import load_vision_config


def _augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Simmetrie valide per le piante: flip orizzontale/verticale + rot90."""
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    k = rng.randint(0, 3)
    if k:
        img = img.rotate(90 * k, expand=True)
    return img


class _PartialViewDataset(Dataset):
    """Genera V viste degradate per pianta (rendering parziale + augmentation).

    1 elemento = (tensore_vista, riga_pianta m, indice_vista v). Il rendering
    avviene nei worker del DataLoader (parallelo); l'encoding resta sulla GPU.
    La vista renderizzata e' cachata su disco (`cache_dir`) e condivisa tra tutti
    i (modello x pooling): il seeding e' per (pianta, vista), indipendente
    dall'indice posizionale, cosi' la chiave di cache combacia col determinismo.
    """

    def __init__(self, plan_paths, transform, views, frac_range, augment, open_boundary, seed, cache_dir):
        self.paths = plan_paths
        self.transform = transform
        self.views = views
        self.frac_lo, self.frac_hi = frac_range
        self.augment = augment
        self.open_boundary = open_boundary
        self.seed = seed
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.paths) * self.views

    def __getitem__(self, k: int):
        m, v = divmod(k, self.views)
        return self.transform(self._view(Path(self.paths[m]).stem, v, self.paths[m])), m, v

    def _view(self, stem: str, v: int, path: str) -> Image.Image:
        """Vista renderizzata (dalla cache se presente, altrimenti la crea)."""
        cache_file = self.cache_dir / f"{stem}_v{v}.png" if self.cache_dir else None
        if cache_file is not None and cache_file.exists():
            return Image.open(cache_file).convert("RGB")

        rng = random.Random(f"{self.seed}:{stem}:{v}")   # deterministico per (pianta, vista)
        meta = load_metadata(path)
        frac = rng.uniform(self.frac_lo, self.frac_hi)
        img, _ = make_partial_query(path, meta, "random", {"fraction": frac}, rng, self.open_boundary)
        if self.augment:
            img = _augment(img, rng)

        if cache_file is not None:                        # scrittura atomica: safe se piu' processi renderizzano
            tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
            img.convert("RGB").save(tmp, "PNG")
            os.replace(tmp, cache_file)
        return img


def _render_cache_dir(save_dir: Path, config) -> Path:
    """Cache condivisa delle viste renderizzate: `embeddings/vision/_render_cache/<hash>`.
    L'hash copre tutto cio' che cambia l'immagine (masking, augment, seed) ma NON
    l'encoder/pooling -> tutti i (modello x pooling) con la stessa config la riusano."""
    key = "|".join([
        f"seed={config.training.seed}",
        f"frac={tuple(config.training.mask_fraction)}",
        f"aug={bool(config.training.augment)}",
        f"open={bool(config.training.get('open_boundary', True))}",
    ])
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return save_dir.parents[1] / "_render_cache" / h


def build_pairs(config) -> None:
    save_dir = Path(config.retrieval.save_dir)
    raw = np.load(save_dir / "embeddings.npy")
    with open(save_dir / "image_paths.json") as f:
        paths = json.load(f)

    # pool di training: split ufficiali train+valid (test tenuto fuori).
    # Salviamo anche lo split per riga: il training separa train (fit) da valid
    # (early stopping / selezione iperparametri) -> nessun tuning sul test.
    pool = set(config.training.split_pool)
    rows, row_splits = [], []
    for i, p in enumerate(paths):
        s = get_split(p)
        if s in pool:
            rows.append(i)
            row_splits.append(s)
    anchors = raw[rows].astype("float32")
    plan_paths = [paths[i] for i in rows]
    M, D = anchors.shape
    V = int(config.training.views_per_plan)
    print(f"[pairs] piante di training (split {sorted(pool)}): {M} | viste/pianta: {V} | D={D}")

    manager = VisionModelManager(config)
    encoder, device = manager.encoder, manager.device

    cache_dir = _render_cache_dir(save_dir, config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pairs] cache viste renderizzate (condivisa): {cache_dir}")

    dataset = _PartialViewDataset(
        plan_paths,
        manager.transform,
        views=V,
        frac_range=tuple(config.training.mask_fraction),
        augment=bool(config.training.augment),
        open_boundary=bool(config.training.get("open_boundary", True)),
        seed=int(config.training.seed),
        cache_dir=cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.retrieval.batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=(device != "cpu"),
    )

    positives = np.zeros((M, V, D), dtype="float32")
    with torch.no_grad():
        for tensors, ms, vs in tqdm(loader, desc="Viste positive"):
            embs = encoder(tensors.to(device)).cpu().numpy()
            positives[ms.numpy(), vs.numpy()] = embs

    np.savez(
        save_dir / "pairs.npz",
        anchors=anchors,
        positives=positives,
        splits=np.array(row_splits),
    )
    print(f"[pairs] salvato {save_dir/'pairs.npz'}: anchors {anchors.shape}, positives {positives.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vision_retrieval.yaml")
    args, overrides = parser.parse_known_args()
    build_pairs(load_vision_config(args.config, overrides))


if __name__ == "__main__":
    main()
