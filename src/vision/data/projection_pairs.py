"""Positive-pair cache for training the projection head.

Self-supervised: the positive of a plan is the same plan degraded (rooms removed) and
optionally augmented (flip/rot90, valid floor-plan symmetries). The signal differs from
the per-axis relevance used for evaluation, so there is no circularity.

Output: `pairs.npz` in the feature-variant folder, row-aligned:
  anchors   [M, D]    frozen embedding of the full plan (from the raw cache)
  positives [M, V, D] frozen embeddings of V degraded views per plan
M = plans of the training pool (train+valid, test excluded).

Rendering a degraded view depends only on (plan, view, masking config), not on the encoder,
so rendered images are cached in `_render_cache/` and shared by all (model x pooling).
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
from src.vision.data.vision_damage import (
    apply_patch_cells, check_head_damage, make_crop_query, make_patch_context,
    resolve_patch_size, room_damage_image, select_patch_cells,
)
from src.vision.models.vision_model_manager import VisionModelManager
from src.vision.utils.config import load_vision_config


# damages a view may get with `training.damage_mix`.
# View v gets damage_mix[v % len(damage_mix)]; None (default) = historical single damage.
MIX_DAMAGES = ("nowalls_random", "crop", "patch")
DEFAULT_PAIRS_FILE = "pairs.npz"


def check_damage_mix(mix) -> tuple[str, ...] | None:
    """Validate `training.damage_mix` (None = historical behaviour)."""
    if mix is None:
        return None
    mix = tuple(str(d) for d in mix)
    if not mix or len(set(mix)) != len(mix) or any(d not in MIX_DAMAGES for d in mix):
        raise ValueError(f"training.damage_mix={list(mix)!r}: distinct values among {MIX_DAMAGES}")
    return mix


def _augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Floor-plan symmetries: horizontal/vertical flip + rot90."""
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    k = rng.randint(0, 3)
    if k:
        img = img.rotate(90 * k, expand=True)
    return img


class _PartialViewDataset(Dataset):
    """V degraded views per plan (partial rendering + augmentation); item = (view tensor, plan row m, view index v).

    Rendering runs in DataLoader workers. Rendered views are cached in `cache_dir`; seeding is
    per (plan, view), independent of the positional index, so cache keys match determinism.
    """

    def __init__(self, plan_paths, transform, views, frac_range, augment, open_boundary, seed, cache_dir,
                 damage="random", damage_mix=None, patch_ctx=None, crop_params=None, patch_params=None):
        self.paths = plan_paths
        self.transform = transform
        self.views = views
        self.frac_lo, self.frac_hi = frac_range
        self.augment = augment
        self.open_boundary = open_boundary
        self.seed = seed
        self.cache_dir = cache_dir
        self.damage = check_head_damage(damage)
        self.damage_mix = check_damage_mix(damage_mix)
        if self.damage_mix and views % len(self.damage_mix):
            raise ValueError(f"views_per_plan={views} is not a multiple of {len(self.damage_mix)} damages")
        if self.damage_mix and "patch" in self.damage_mix and patch_ctx is None:
            raise ValueError("damage_mix with `patch` needs a patch context")
        self.patch_ctx = patch_ctx
        self.crop_params = dict(crop_params or {})
        self.patch_params = dict(patch_params or {})

    def view_damage(self, v: int) -> str:
        """Damage of view v: the single historical damage, or the mix in turn."""
        return self.damage_mix[v % len(self.damage_mix)] if self.damage_mix else self.damage

    def __len__(self) -> int:
        return len(self.paths) * self.views

    def __getitem__(self, k: int):
        m, v = divmod(k, self.views)
        img = self._view(Path(self.paths[m]).stem, v, self.paths[m])
        if self.view_damage(v) == "patch":            # already the resized canvas (as the evaluation)
            return self.patch_ctx["finish"](img), m, v
        return self.transform(img), m, v

    def _view(self, stem: str, v: int, path: str) -> Image.Image:
        """Rendered view, from cache if present."""
        cache_file = self.cache_dir / f"{stem}_v{v}.png" if self.cache_dir else None
        if cache_file is not None and cache_file.exists():
            return Image.open(cache_file).convert("RGB")

        rng = random.Random(f"{self.seed}:{stem}:{v}")   # deterministic per (plan, view)
        meta = load_metadata(path)
        frac = rng.uniform(self.frac_lo, self.frac_hi)
        damage = self.view_damage(v)
        if damage == "crop":
            img, _ = make_crop_query(path, frac, rng, **self.crop_params)
        elif damage == "patch":
            # Same steps as `vision_damage.damaged_query`: whole ViT patches on the
            # resized canvas; flip/rot90 of the square canvas keep the patch grid.
            canvas = self.patch_ctx["resize"](Image.open(path).convert("RGB"))
            cells, _ = select_patch_cells(canvas, self.patch_ctx["patch_size"], frac, rng,
                                          **self.patch_params)
            img = apply_patch_cells(canvas, cells, self.patch_ctx["patch_size"])
        else:
            img, _ = room_damage_image(path, meta, damage, {"fraction": frac}, rng, self.open_boundary)
        if self.augment:
            img = _augment(img, rng)

        if cache_file is not None:                        # atomic write: safe with concurrent renderers
            tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
            img.convert("RGB").save(tmp, "PNG")
            os.replace(tmp, cache_file)
        return img


def _patch_key(config) -> str:
    """Patch canvas depends on the encoder input: size and patch side enter the cache key."""
    return f"{config.model.kwargs.get('image_size')}/{config.model.kwargs.get('patch_size')}"


def _render_cache_dir(save_dir: Path, config) -> Path:
    """Shared render cache `embeddings/vision/_render_cache/<hash>`.

    The hash covers what changes the image (masking, augment, seed), not encoder/pooling.
    `training.damage` enters the key only if not `random`, so existing caches keep their hash."""
    parts = [
        f"seed={config.training.seed}",
        f"frac={tuple(config.training.mask_fraction)}",
        f"aug={bool(config.training.augment)}",
        f"open={bool(config.training.get('open_boundary', True))}",
    ]
    damage = check_head_damage(config.training.get("damage", "random"))
    if damage != "random":
        parts.append(f"damage={damage}")
    mix = check_damage_mix(config.training.get("damage_mix"))
    if mix:                       # absent by default, existing keys unchanged
        parts.append(f"mix={'+'.join(mix)}")
        if "patch" in mix:
            parts.append(f"patch={_patch_key(config)}")
    key = "|".join(parts)
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return save_dir.parents[1] / "_render_cache" / h


def build_pairs(config) -> None:
    save_dir = Path(config.retrieval.save_dir)
    raw = np.load(save_dir / "embeddings.npy")
    with open(save_dir / "image_paths.json") as f:
        paths = json.load(f)

    # training pool: official train+valid (test excluded); the per-row split is saved so
    # training can separate fit (train) from early stopping (valid)
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

    damage = check_head_damage(config.training.get("damage", "random"))
    mix = check_damage_mix(config.training.get("damage_mix"))
    pairs_file = str(config.training.get("pairs_file") or DEFAULT_PAIRS_FILE)
    if pairs_file != DEFAULT_PAIRS_FILE and (save_dir / pairs_file).exists():
        raise FileExistsError(f"{save_dir / pairs_file} exists already: nothing is overwritten")
    patch_ctx, crop_params, patch_params = None, {}, {}
    if mix:
        strat = config.partial.strategies       # same damage parameters as the evaluation
        crop_params = {"tolerance": float(strat.crop.tolerance), "max_tries": int(strat.crop.max_tries)}
        patch_params = {"tolerance": float(strat.patch.tolerance)}
        if "patch" in mix:
            ctx = make_patch_context(manager.transform, int(config.model.kwargs.get("patch_size")))
            p = resolve_patch_size(encoder, config.model.kwargs.get("patch_size"), ctx["image_size"])
            patch_ctx = make_patch_context(manager.transform, p)
        print(f"[pairs] head v2: viste a danni misti {list(mix)} (vista v -> mix[v % {len(mix)}]) "
              f"-> {pairs_file}")
    cache_dir = _render_cache_dir(save_dir, config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pairs] danno delle viste: {damage} | cache viste renderizzate (condivisa): {cache_dir}")

    dataset = _PartialViewDataset(
        plan_paths,
        manager.transform,
        views=V,
        frac_range=tuple(config.training.mask_fraction),
        augment=bool(config.training.augment),
        open_boundary=bool(config.training.get("open_boundary", True)),
        seed=int(config.training.seed),
        cache_dir=cache_dir,
        damage=damage,
        damage_mix=mix,
        patch_ctx=patch_ctx,
        crop_params=crop_params,
        patch_params=patch_params,
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

    saved_damage = f"mix:{'+'.join(mix)}" if mix else damage
    extra = {}
    if mix:
        extra["view_damages"] = np.array([dataset.view_damage(v) for v in range(V)])
        extra["mask_fraction"] = np.array(list(config.training.mask_fraction), dtype="float64")
    np.savez(
        save_dir / pairs_file,
        anchors=anchors,
        positives=positives,
        splits=np.array(row_splits),
        damage=np.array(saved_damage),      # compared by train_projection with config and probe
        **extra,
    )
    print(f"[pairs] salvato {save_dir/pairs_file}: anchors {anchors.shape}, positives {positives.shape} | danno {saved_damage}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vision_retrieval.yaml")
    args, overrides = parser.parse_known_args()
    build_pairs(load_vision_config(args.config, overrides))


if __name__ == "__main__":
    main()
