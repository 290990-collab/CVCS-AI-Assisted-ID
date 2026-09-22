# tests/test_vision_damage.py

"""
CPU smoke tests for the vision damage modes (`src/vision/data/vision_damage.py`):
room removal (the historical strategies), rectangular crop and ViT-patch
masking.

The first block holds GOLDEN hashes computed with the room-removal code as it
was BEFORE the damage module existed (11 Sep 2026): if any of them changes, the
historical random/semantic/topology queries are no longer bit-identical and
every partial number already reported would be invalidated.

The last block covers `src/evaluation/damage_curves.py` (curves vs removed
area) on synthetic per-query files.

Everything runs on CPU on a synthetic 4-room plan drawn in memory (no .mat, no
dataset, no downloaded weights).

Run: python -m pytest tests/test_vision_damage.py -v
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.data.rplan_metadata import RoomMeta
from src.evaluation.perquery import gallery_sha1, load_perquery
from src.evaluation.relevance import GalleryAxes
from src.vision.data.preprocess import IMAGENET_MEAN, IMAGENET_STD, compose_transform
from src.vision.data.vision_damage import (
    NOWALLS_STRATEGIES,
    apply_patch_cells,
    damaged_query,
    make_crop_query,
    make_wiped_query,
    neutral_wall_mask,
    make_patch_context,
    plan_area_removed,
    plan_mask,
    resolve_patch_size,
    select_patch_cells,
    split_query_transform,
)
from src.vision.data.vision_partial_query import make_partial_query
from src.vision.evaluation.evaluate import _label_slug, evaluate_partial, partial_runs

# ----------------------------------------------------------------------
# Synthetic plan: the 4-room chain 0-1-2-3 of `tests/test_contracts.py`
# (`_full_meta`), drawn on a NON-square canvas so that ResizeWithPad pads.
# Grid 0..40 -> pixels with scale 2, offset (40, 24): plan spans x 40..120,
# y 24..104 on a 160x128 white canvas.
# ----------------------------------------------------------------------

CANVAS_W, CANVAS_H = 160, 128
OFFSET_X, OFFSET_Y, SCALE = 40, 24, 2
WALL_RGB = (70, 70, 70)
WALL_HALF = 3
ROOM_RGB = ((236, 206, 160), (196, 228, 196), (176, 196, 236), (240, 222, 150))


def plan_meta() -> RoomMeta:
    """Same RoomMeta as `_full_meta` in tests/test_contracts.py."""
    return RoomMeta(
        name="p42", split="valid",
        room_types=(0, 1, 2, 3),
        edges=((0, 1, 1), (1, 2, 1), (2, 3, 1)),
        boxes=((0, 0, 20, 20), (20, 0, 40, 20), (0, 20, 20, 40), (20, 20, 40, 40)),
        footprint=(0, 0, 40, 40), entrance=None,
    )


def draw_plan(meta: RoomMeta | None = None) -> np.ndarray:
    """RGB uint8 [H, W, 3]: white background, dark walls, pastel rooms."""
    meta = meta or plan_meta()
    arr = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)

    def px(x, y):
        return OFFSET_X + SCALE * x, OFFSET_Y + SCALE * y

    gx0 = min(b[0] for b in meta.boxes)
    gy0 = min(b[1] for b in meta.boxes)
    gx1 = max(b[2] for b in meta.boxes)
    gy1 = max(b[3] for b in meta.boxes)
    (x0, y0), (x1, y1) = px(gx0, gy0), px(gx1, gy1)
    arr[y0:y1 + 1, x0:x1 + 1] = WALL_RGB
    for (bx0, by0, bx1, by1), rtype in zip(meta.boxes, meta.room_types):
        (x0, y0), (x1, y1) = px(bx0, by0), px(bx1, by1)
        arr[y0 + WALL_HALF:y1 - WALL_HALF + 1, x0 + WALL_HALF:x1 - WALL_HALF + 1] = \
            ROOM_RGB[rtype % len(ROOM_RGB)]
    return arr


@pytest.fixture
def plan_png(tmp_path):
    path = tmp_path / "p42.png"
    Image.fromarray(draw_plan()).save(path)
    return path


def _sha1(img: Image.Image) -> str:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return hashlib.sha1(arr.tobytes()).hexdigest()


# ======================================================================
# 1. GOLDEN: the historical room-removal queries stay bit-identical.
# ======================================================================

# (strategy, params, seed, qi, open_boundary, removed, sha1 of the RGB image),
# computed with `make_partial_query` BEFORE any damage code was written.
GOLDEN_ROOMS = [
    ('random', {'fraction': 0.0}, 42, 0, True, [], '1f77b1c8ca30a0ba73dfea2a4ace5f225801ce21'),
    ('random', {'fraction': 0.0}, 42, 7, True, [], '1f77b1c8ca30a0ba73dfea2a4ace5f225801ce21'),
    ('random', {'fraction': 0.0}, 42, 13, True, [], '1f77b1c8ca30a0ba73dfea2a4ace5f225801ce21'),
    ('random', {'fraction': 0.0}, 0, 3, True, [], '1f77b1c8ca30a0ba73dfea2a4ace5f225801ce21'),
    ('random', {'fraction': 0.25}, 42, 0, True, [0], '46721e1bb169d151d73d823581ac5d853050e9a9'),
    ('random', {'fraction': 0.25}, 42, 7, True, [0], '46721e1bb169d151d73d823581ac5d853050e9a9'),
    ('random', {'fraction': 0.25}, 42, 13, True, [0], '46721e1bb169d151d73d823581ac5d853050e9a9'),
    ('random', {'fraction': 0.25}, 0, 3, True, [1], '5cf5aa0c40a4d4c8b5d07d6f1774222981efbebd'),
    ('random', {'fraction': 0.5}, 42, 0, True, [0, 3], '92e61ebb40a003cb07ee67ed1d51d0994806f2cc'),
    ('random', {'fraction': 0.5}, 42, 7, True, [0, 1], '2b3bd6da897670c5fa38384d452c78837f3ed059'),
    ('random', {'fraction': 0.5}, 42, 13, True, [0, 3], '92e61ebb40a003cb07ee67ed1d51d0994806f2cc'),
    ('random', {'fraction': 0.5}, 0, 3, True, [1, 2], 'aced2e6199b7c815ff2ee503383b0e178159b238'),
    ('random', {'fraction': 0.75}, 42, 0, True, [0, 1, 3], '8c1d5f46d2b3d783982582400996508fe8925907'),
    ('random', {'fraction': 0.75}, 42, 7, True, [0, 1, 2], '7e343676907a9ba1a226903bc9e5b5665acf7a1d'),
    ('random', {'fraction': 0.75}, 42, 13, True, [0, 2, 3], 'f9ca0a2c32f729a70c1107eeabf0d998602446da'),
    ('random', {'fraction': 0.75}, 0, 3, True, [0, 1, 2], '7e343676907a9ba1a226903bc9e5b5665acf7a1d'),
    ('random', {'fraction': 0.5}, 42, 7, False, [0, 1], '37b4ceb92f1e76c68b37ebd50dfc571cc09db717'),
    ('semantic', {'keep_types': [0, 2, 3]}, 42, 0, True, [1], '5cf5aa0c40a4d4c8b5d07d6f1774222981efbebd'),
    ('semantic', {'keep_types': [0, 2, 3]}, 42, 0, False, [1], '44941360270a4a2585f74b9ef20eb694c816537f'),
    ('topology', {'max_degree': 1}, 42, 0, True, [0, 3], '92e61ebb40a003cb07ee67ed1d51d0994806f2cc'),
    ('topology', {'max_degree': 1}, 42, 0, False, [0, 3], 'dbf3a87f56b58a8235df8a95ebcb8a397f9dffaa'),
]


@pytest.mark.parametrize("strategy, params, seed, qi, open_boundary, removed, sha1", GOLDEN_ROOMS)
def test_room_removal_is_bit_identical_to_the_pre_damage_code(
    plan_png, strategy, params, seed, qi, open_boundary, removed, sha1
):
    img, got_removed = make_partial_query(
        plan_png, plan_meta(), strategy, params, random.Random(seed + qi), open_boundary
    )
    assert got_removed == removed
    assert _sha1(img) == sha1


def test_damaged_query_room_branch_equals_the_historical_two_calls(plan_png):
    """`damaged_query` on a room strategy = make_partial_query + transform, and
    measuring the area does not consume the query rng."""
    T = compose_transform(224, IMAGENET_MEAN, IMAGENET_STD)
    for strategy, params in (("random", {"fraction": 0.5}), ("random", {"fraction": 0.0}),
                             ("semantic", {"keep_types": [0, 2, 3]}),
                             ("topology", {"max_degree": 1})):
        rng_old, rng_new = random.Random(49), random.Random(49)
        img, removed = make_partial_query(plan_png, plan_meta(), strategy, params, rng_old, True)
        expected = T(img)
        got, info = damaged_query(plan_png, plan_meta(), strategy, params, rng_new, True, T)
        assert torch.equal(got, expected)
        assert rng_old.random() == rng_new.random()
        assert info["removed_any"] == bool(removed)
        assert info["fallback"] is False
        if removed:
            assert 0.0 < info["area_removed"] <= 1.0
        else:
            assert info["area_removed"] == 0.0


# ======================================================================
# 2. partial_runs: historical list untouched, crop/patch appended only if on.
# ======================================================================

HISTORICAL_STRATEGIES = {
    "random": {"enabled": True, "fractions": [0.0, 0.25, 0.5, 0.75]},
    "semantic": {"enabled": True, "keep_types": [0, 2, 3]},
    "topology": {"enabled": True, "max_degree": 1},
}
HISTORICAL_RUNS = [
    ("random f=0.0", "random", {"fraction": 0.0}),
    ("random f=0.25", "random", {"fraction": 0.25}),
    ("random f=0.5", "random", {"fraction": 0.5}),
    ("random f=0.75", "random", {"fraction": 0.75}),
    ("semantic", "semantic", {"keep_types": [0, 2, 3]}),
    ("topology", "topology", {"max_degree": 1}),
]


def _pcfg(crop=None, patch=None, **strategies):
    s = {**HISTORICAL_STRATEGIES, **strategies}
    if crop is not None:
        s["crop"] = crop
    if patch is not None:
        s["patch"] = patch
    return OmegaConf.create({"seed": 42, "open_boundary": True, "strategies": s})


def test_partial_runs_without_damage_keys_is_the_historical_list():
    assert partial_runs(_pcfg()) == HISTORICAL_RUNS
    off = _pcfg(crop={"enabled": False, "fractions": [0.5]},
                patch={"enabled": False, "fractions": [0.5]})
    assert partial_runs(off) == HISTORICAL_RUNS


def test_partial_runs_appends_crop_and_patch_after_topology():
    on = _pcfg(crop={"enabled": True, "fractions": [0.25, 0.5], "tolerance": 0.05,
                     "max_tries": 100},
               patch={"enabled": True, "fractions": [0.75], "tolerance": 0.02})
    runs = partial_runs(on)
    assert runs[:len(HISTORICAL_RUNS)] == HISTORICAL_RUNS
    assert runs[len(HISTORICAL_RUNS):] == [
        ("crop f=0.25", "crop", {"fraction": 0.25, "tolerance": 0.05, "max_tries": 100}),
        ("crop f=0.5", "crop", {"fraction": 0.5, "tolerance": 0.05, "max_tries": 100}),
        ("patch f=0.75", "patch", {"fraction": 0.75, "tolerance": 0.02}),
    ]
    assert [_label_slug(l) for l, _, _ in runs[len(HISTORICAL_RUNS):]] == \
        ["crop-f0.25", "crop-f0.5", "patch-f0.75"]


def test_yaml_ships_damage_modes_switched_off():
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "vision_retrieval.yaml")
    assert cfg.partial.strategies.crop.enabled is False
    assert cfg.partial.strategies.patch.enabled is False
    assert partial_runs(cfg.partial) == HISTORICAL_RUNS


# ======================================================================
# 3. Crop.
# ======================================================================

def test_crop_changes_only_the_rectangle_and_blanks_it(plan_png):
    before = np.array(Image.open(plan_png).convert("RGB"))
    img, info = make_crop_query(plan_png, 0.5, random.Random(7))
    after = np.array(img)
    assert after.shape == before.shape
    x0, y0, x1, y1 = info["rect"]
    inside = np.zeros(before.shape[:2], dtype=bool)
    inside[y0:y1, x0:x1] = True
    changed = np.any(after != before, axis=2)
    assert not (changed & ~inside).any()                   # nothing outside moves
    assert (after[inside] == 255).all()                    # the rectangle is all white
    assert np.array_equal(after[~inside], before[~inside])


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_crop_area_is_within_tolerance_and_matches_an_independent_count(plan_png, fraction):
    before = np.array(Image.open(plan_png).convert("RGB"))
    plan_before = np.any(before < 248, axis=2)
    for seed in range(20):
        img, info = make_crop_query(plan_png, fraction, random.Random(seed))
        after = np.array(img)
        assert not info["fallback"]
        assert abs(info["area_removed"] - fraction) <= 0.05
        white_after = np.all(after >= 248, axis=2)
        independent = (plan_before & white_after).sum() / plan_before.sum()
        assert info["area_removed"] == pytest.approx(independent, abs=1e-12)
        assert plan_area_removed(before, after) == pytest.approx(independent, abs=1e-12)


def test_crop_fallback_is_flagged_and_keeps_the_best_try(plan_png):
    # tolerance=0 cannot be met: the plan has an odd number of pixels
    before = np.array(Image.open(plan_png).convert("RGB"))
    assert int(plan_mask(before).sum()) % 2 == 1
    img, info = make_crop_query(plan_png, 0.5, random.Random(3), tolerance=0.0, max_tries=5)
    assert info["fallback"] is True
    assert info["tries"] == 5
    # the kept rectangle is the closest of the 5 draws
    rng = random.Random(3)
    ys, xs = np.where(plan_mask(before))
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max(), ys.max()
    errs = []
    for _ in range(5):
        w = rng.randint(1, bx1 - bx0 + 1)
        h = rng.randint(1, by1 - by0 + 1)
        x0 = rng.randint(bx0, bx1 - w + 1)
        y0 = rng.randint(by0, by1 - h + 1)
        cov = plan_mask(before)[y0:y0 + h, x0:x0 + w].sum() / plan_mask(before).sum()
        errs.append(abs(cov - 0.5))
    assert abs(info["area_removed"] - 0.5) == pytest.approx(min(errs), abs=1e-12)
    assert plan_area_removed(before, np.array(img)) == pytest.approx(info["area_removed"])


# ======================================================================
# 4. Patch.
# ======================================================================

@pytest.mark.parametrize("image_size, patch_size", [(224, 14), (224, 16), (448, 14)])
def test_patch_changes_only_whole_aligned_blocks(plan_png, image_size, patch_size):
    T = compose_transform(image_size, IMAGENET_MEAN, IMAGENET_STD)
    resize, _ = split_query_transform(T)
    canvas = np.array(resize(Image.open(plan_png).convert("RGB")))
    cells, info = select_patch_cells(canvas, patch_size, 0.5, random.Random(11))
    damaged = np.array(apply_patch_cells(canvas, cells, patch_size))

    assert damaged.shape == canvas.shape == (image_size, image_size, 3)
    assert cells and not info["fallback"]
    chosen = np.zeros(canvas.shape[:2], dtype=bool)
    for r, c in cells:
        chosen[r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size] = True
    assert (damaged[chosen] == 255).all()                     # chosen blocks all white
    assert np.array_equal(damaged[~chosen], canvas[~chosen])  # the rest identical
    ys, xs = np.where(np.any(damaged != canvas, axis=2))
    assert all((y // patch_size, x // patch_size) in set(cells) for y, x in zip(ys, xs))
    # the area is measured on the resized canvas and hits the target
    area = plan_area_removed(canvas, damaged)
    assert area == pytest.approx(info["coverage"], abs=1e-12)
    assert abs(area - 0.5) <= 0.05


def test_patch_rejects_a_grid_that_does_not_divide_the_canvas(plan_png):
    canvas = np.full((224, 224, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError):
        select_patch_cells(canvas, 15, 0.5, random.Random(0))


@pytest.mark.parametrize("mean, std", [
    (IMAGENET_MEAN, IMAGENET_STD),
    ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
])
def test_split_transform_is_the_same_computation_and_blank_cells_are_white(plan_png, mean, std):
    T = compose_transform(224, mean, std)
    img = Image.open(plan_png).convert("RGB")
    resize, finish = split_query_transform(T)
    assert torch.equal(finish(resize(img)), T(img))

    ctx = make_patch_context(T, 16)
    x, info = damaged_query(plan_png, plan_meta(), "patch", {"fraction": 0.5},
                            random.Random(5), True, T, ctx)
    canvas = resize(img)
    cells, _ = select_patch_cells(canvas, 16, 0.5, random.Random(5))
    assert x.shape == (3, 224, 224)
    white = (1.0 - torch.tensor(mean, dtype=torch.float32)) / torch.tensor(std, dtype=torch.float32)
    for r, c in cells:
        block = x[:, r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
        assert torch.allclose(block, white.view(3, 1, 1).expand_as(block), atol=1e-6)


def test_split_transform_rejects_a_transform_without_resize_with_pad():
    from torchvision import transforms
    with pytest.raises(ValueError):
        split_query_transform(transforms.Compose([transforms.ToTensor()]))
    with pytest.raises(ValueError):
        split_query_transform(lambda img: img)


@pytest.mark.parametrize("strategy", ["crop", "patch"])
def test_damage_is_deterministic_with_the_seed(plan_png, strategy):
    T = compose_transform(224, IMAGENET_MEAN, IMAGENET_STD)
    ctx = make_patch_context(T, 16)
    params = {"fraction": 0.5}
    a, ia = damaged_query(plan_png, plan_meta(), strategy, params, random.Random(42 + 7), True, T, ctx)
    b, ib = damaged_query(plan_png, plan_meta(), strategy, params, random.Random(42 + 7), True, T, ctx)
    c, _ = damaged_query(plan_png, plan_meta(), strategy, params, random.Random(42 + 8), True, T, ctx)
    assert torch.equal(a, b) and ia == ib
    assert not torch.equal(a, c)


# ======================================================================
# 5. resolve_patch_size.
# ======================================================================

def test_resolve_patch_size_checks_the_model_when_it_can():
    hf = SimpleNamespace(backbone=SimpleNamespace(config=SimpleNamespace(patch_size=14)))
    assert resolve_patch_size(hf, 14, 224) == 14
    assert resolve_patch_size(hf, None, 224) == 14
    with pytest.raises(ValueError, match="patch_size"):
        resolve_patch_size(hf, 16, 224)

    clip = SimpleNamespace(model=SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(patch_size=16))))
    assert resolve_patch_size(clip, 16, 224) == 16
    timm_like = SimpleNamespace(model=SimpleNamespace(patch_embed=SimpleNamespace(patch_size=(16, 16))))
    assert resolve_patch_size(timm_like, 16, 224) == 16
    with pytest.raises(ValueError):
        resolve_patch_size(timm_like, 14, 224)


def test_resolve_patch_size_warns_when_the_model_hides_it_and_checks_the_grid():
    opaque = SimpleNamespace(model=SimpleNamespace())
    with pytest.warns(UserWarning):
        assert resolve_patch_size(opaque, 16, 224) == 16
    with pytest.raises(ValueError):
        resolve_patch_size(opaque, None, 224)
    hf = SimpleNamespace(backbone=SimpleNamespace(config=SimpleNamespace(patch_size=16)))
    with pytest.raises(ValueError, match="multiple"):
        resolve_patch_size(hf, 16, 448 + 8)


def test_every_encoder_preset_declares_a_patch_size_that_divides_its_image_size():
    presets = Path(__file__).resolve().parents[1] / "configs" / "vision_models"
    for path in sorted(presets.glob("*.yaml")):
        cfg = OmegaConf.load(path)
        assert "patch_size" in cfg, path.name
        assert int(cfg.image_size) % int(cfg.patch_size) == 0, path.name


# ======================================================================
# 6. Smoke of evaluate_partial with a fake pipeline.
# ======================================================================

GALLERY_METAS = [
    RoomMeta(name=f"g{i}", split="valid", room_types=types, edges=edges, boxes=boxes,
             footprint=(0, 0, 40, 40), entrance=None)
    for i, (types, edges, boxes) in enumerate([
        ((0, 1, 2, 3), ((0, 1, 1), (1, 2, 1), (2, 3, 1)),
         ((0, 0, 20, 20), (20, 0, 40, 20), (0, 20, 20, 40), (20, 20, 40, 40))),
        ((0, 1, 2, 3), ((0, 1, 1), (1, 2, 1), (2, 3, 1)),
         ((0, 0, 20, 20), (20, 0, 40, 20), (0, 20, 20, 40), (20, 20, 40, 40))),
        ((0, 2, 2, 3), ((0, 1, 1), (0, 2, 1), (2, 3, 1)),
         ((0, 0, 20, 20), (20, 0, 40, 20), (0, 20, 20, 40), (20, 20, 40, 40))),
        ((1, 0, 3), ((0, 1, 1), (1, 2, 1)),
         ((0, 0, 20, 40), (20, 0, 40, 20), (20, 20, 40, 40))),
        ((1, 0, 3), ((0, 1, 1), (1, 2, 1)),
         ((0, 0, 20, 40), (20, 0, 40, 20), (20, 20, 40, 40))),
        ((3, 3, 0), ((0, 2, 1), (1, 2, 1)),
         ((0, 0, 40, 20), (0, 20, 20, 40), (20, 20, 40, 40))),
    ])
]


class _FakePipeline:
    """Nearest neighbour on the preprocessed pixels: the ranking depends on the
    damaged query, so a change in the damage would change the metrics."""

    def __init__(self, image_paths, transform):
        self.transform = transform
        self.image_paths = [str(p) for p in image_paths]
        self._gallery = torch.stack([
            transform(Image.open(p).convert("RGB")).flatten() for p in self.image_paths
        ])

    def query(self, query_image, top_k=5):
        d = ((self._gallery - query_image.flatten()) ** 2).sum(dim=1).numpy()
        order = np.argsort(d, kind="stable")[:top_k]
        return [{"path": self.image_paths[j], "score": float(-d[j])} for j in order]


def _run_smoke(tmp_path, monkeypatch, out_dir, crop_on, patch_on):
    by_name = {m.name: m for m in GALLERY_METAS}
    paths = []
    for m in GALLERY_METAS:
        p = tmp_path / f"{m.name}.png"
        if not p.exists():
            Image.fromarray(draw_plan(m)).save(p)
        paths.append(str(p))
    monkeypatch.setattr("src.vision.evaluation.evaluate.load_metadata",
                        lambda p, *a, **k: by_name[Path(str(p)).stem])
    T = compose_transform(224, IMAGENET_MEAN, IMAGENET_STD)
    pipeline = _FakePipeline(paths, T)
    axes = GalleryAxes(GALLERY_METAS)
    stem2row = {Path(p).stem: i for i, p in enumerate(paths)}
    pcfg = OmegaConf.create({
        "seed": 42, "open_boundary": True,
        "strategies": {
            "random": {"enabled": True, "fractions": [0.0, 0.5]},
            "semantic": {"enabled": False, "keep_types": [0, 2, 3]},
            "topology": {"enabled": False, "max_degree": 1},
            "crop": {"enabled": crop_on, "fractions": [0.5], "tolerance": 0.05, "max_tries": 10000},
            "patch": {"enabled": patch_on, "fractions": [0.5], "tolerance": 0.05},
        },
    })
    names = [m.name for m in GALLERY_METAS]
    ctx = {"dir": out_dir, "tag": "fake_natural_raw", "split": "valid", "seed": 42,
           "gallery": {"n": len(names), "sha1": gallery_sha1(names), "source": "synthetic"}}
    evaluate_partial(pipeline, axes, stem2row, list(range(len(paths))), paths, (1, 3),
                     pcfg, perquery=ctx, patch_size=16 if patch_on else None)


def test_evaluate_partial_damage_runs_leave_the_random_files_untouched(tmp_path, monkeypatch):
    off, on = tmp_path / "off", tmp_path / "on"
    _run_smoke(tmp_path, monkeypatch, off, crop_on=False, patch_on=False)
    _run_smoke(tmp_path, monkeypatch, on, crop_on=True, patch_on=True)

    assert sorted(p.name for p in off.glob("*.npz")) == [
        "vision_fake_natural_raw_partial-random-f0.0_valid.npz",
        "vision_fake_natural_raw_partial-random-f0.5_valid.npz",
    ]
    assert sorted(p.name for p in on.glob("*.npz")) == sorted([
        "vision_fake_natural_raw_partial-random-f0.0_valid.npz",
        "vision_fake_natural_raw_partial-random-f0.5_valid.npz",
        "vision_fake_natural_raw_partial-crop-f0.5_valid.npz",
        "vision_fake_natural_raw_partial-patch-f0.5_valid.npz",
    ])
    for p in off.glob("*.npz"):
        a, b = load_perquery(p), load_perquery(on / p.name)
        for key in ("names", "qi", "ndcg", "recall", "map", "num_relevant",
                    "ret_rows", "n_ret", "self_rr", "area_removed"):
            assert np.array_equal(getattr(a, key), getattr(b, key), equal_nan=key in
                                  ("ndcg", "recall", "map", "self_rr")), (p.name, key)
        assert a.meta["damage"] == b.meta["damage"]
        assert a.meta["damage"]["area_space"] == "native"
        assert a.meta["damage"]["patch_size"] is None

    f0 = load_perquery(off / "vision_fake_natural_raw_partial-random-f0.0_valid.npz")
    assert np.all(f0.area_removed == 0.0)
    f5 = load_perquery(off / "vision_fake_natural_raw_partial-random-f0.5_valid.npz")
    assert np.all(f5.area_removed > 0.0)

    crop = load_perquery(on / "vision_fake_natural_raw_partial-crop-f0.5_valid.npz")
    patch = load_perquery(on / "vision_fake_natural_raw_partial-patch-f0.5_valid.npz")
    for d, strategy, space in ((crop, "crop", "native"), (patch, "patch", "resized")):
        assert d.area_removed is not None and d.area_removed.shape == (len(GALLERY_METAS),)
        assert np.all(np.abs(d.area_removed - 0.5) <= 0.05)
        dmg = d.meta["damage"]
        assert dmg["strategy"] == strategy and dmg["area_space"] == space
        assert dmg["image_size"] == 224 and dmg["n_fallback"] == 0
        assert d.meta["partial_label"] == f"{strategy} f=0.5"
    assert patch.meta["damage"]["patch_size"] == 16
    assert crop.meta["damage"]["patch_size"] is None


def test_evaluate_partial_patch_without_patch_size_fails_before_any_query(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="patch_size"):
        by_name = {m.name: m for m in GALLERY_METAS}
        monkeypatch.setattr("src.vision.evaluation.evaluate.load_metadata",
                            lambda p, *a, **k: by_name[Path(str(p)).stem])
        pcfg = OmegaConf.create({"strategies": {"patch": {"enabled": True, "fractions": [0.5]}}})
        pipeline = SimpleNamespace(transform=compose_transform(224, IMAGENET_MEAN, IMAGENET_STD),
                                   query=lambda *a, **k: pytest.fail("query reached"))
        evaluate_partial(pipeline, GalleryAxes(GALLERY_METAS), {}, [0], ["g0.png"], (1,), pcfg)


# ======================================================================
# 7. damage_curves: windows on the achieved area, paired crop - patch.
# ======================================================================

def _write_damage_file(prefix, strategy, frac, names, self_rr, area, geometry=None,
                       with_area=True):
    from src.evaluation.perquery import write_npz
    from src.evaluation.relevance import AXES
    from src.evaluation.robustness_auc import fraction_path
    q = len(names)
    ndcg = np.full((len(AXES), 1, q), np.nan, dtype=np.float32)
    if geometry is not None:
        ndcg[AXES.index("geometry"), 0] = geometry
    meta = {
        "branch": "vision", "run_tag": "sys", "mode": "partial",
        "partial_label": f"{strategy} f={frac}", "split": "valid", "exclude_self": True,
        "query_seed": 42, "axes": list(AXES), "k_values": [10],
        "gallery": {"n": 10, "sha1": gallery_sha1([f"g{i}" for i in range(10)]),
                    "source": "synthetic"},
        "damage": {"strategy": strategy, "n_fallback": 0,
                   "area_space": "resized" if strategy == "patch" else "native"},
    }
    write_npz(fraction_path(prefix, frac, "valid", strategy), meta=meta, names=names,
              qi=list(range(q)), ndcg=ndcg, recall=ndcg.copy(), map_=ndcg.copy(),
              num_relevant=np.full((len(AXES), q), -1, dtype=np.int32),
              self_rr=np.asarray(self_rr, dtype=np.float32),
              area_removed=np.asarray(area, dtype=np.float32) if with_area else None)


def test_damage_curves_average_per_query_before_over_queries(tmp_path):
    from src.evaluation.damage_curves import load_runs, window_curve
    pre = tmp_path / "cfg"
    names = ["q0", "q1", "q2", "q3"]
    # q0 falls in the 0.25 window from BOTH nominal levels: it must count once
    _write_damage_file(pre, "random", 0.25, names, [1.0, 0.5, 0.5, 0.0], [0.22, 0.31, 0.26, 0.5])
    _write_damage_file(pre, "random", 0.5, names, [0.0, 0.2, 0.2, 0.2], [0.27, 0.5, 0.55, 0.48])
    runs = load_runs(pre, verbose=False)
    row = window_curve(runs, "random", "self_rr", centers=(0.25,), halfwidth=0.05)[0]
    assert row["n_queries"] == 2 and row["n_points"] == 3
    assert row["mean"] == pytest.approx(np.mean([np.mean([1.0, 0.0]), 0.5]))
    assert row["area_mean"] == pytest.approx(np.mean([np.mean([0.22, 0.27]), 0.26]), abs=1e-6)
    assert row["ci_lo"] <= row["mean"] <= row["ci_hi"]


def test_damage_curves_paired_delta_and_area_stats(tmp_path):
    from src.evaluation.damage_curves import area_stats, load_runs, paired_delta
    pre = tmp_path / "cfg"
    _write_damage_file(pre, "crop", 0.5, ["a", "b", "c"], [1.0, 0.5, 0.25], [0.5, 0.52, 0.48],
                       geometry=[0.9, 0.8, np.nan])
    # same queries in another order, plus one the crop file does not have
    _write_damage_file(pre, "patch", 0.5, ["c", "a", "b", "z"], [0.25, 0.5, 0.5, 1.0],
                       [0.49, 0.5, 0.51, 0.5], geometry=[0.7, 0.6, 0.8, 0.1])
    runs = load_runs(pre, verbose=False)
    d = paired_delta(runs, "self_rr")[0]
    assert d["n_pairs"] == 3
    assert d["delta"] == pytest.approx(np.mean([1.0 - 0.5, 0.5 - 0.5, 0.25 - 0.25]))
    g = paired_delta(runs, "ndcg10_geometry")[0]
    assert g["n_pairs"] == 2                                   # NaN drops "c" in AND
    assert g["delta"] == pytest.approx(np.mean([0.9 - 0.6, 0.8 - 0.8]), abs=1e-6)

    stats = {s["strategy"]: s for s in area_stats(runs)}
    assert stats["crop"]["mean"] == pytest.approx(0.5, abs=1e-6)
    assert stats["crop"]["min"] == pytest.approx(0.48, abs=1e-6)
    assert stats["patch"]["area_space"] == "resized"


def test_damage_curves_skip_old_files_and_write_the_csv(tmp_path, capsys):
    import csv as _csv
    from src.evaluation.damage_curves import load_runs, main
    pre = tmp_path / "cfg"
    _write_damage_file(pre, "random", 0.5, ["a", "b"], [1.0, 0.5], [0.5, 0.5], with_area=False)
    _write_damage_file(pre, "crop", 0.5, ["a", "b"], [1.0, 0.5], [0.5, 0.52])
    runs = load_runs(pre)
    assert [r.strategy for r in runs] == ["crop"]
    assert "skip" in capsys.readouterr().out

    out = tmp_path / "curves.csv"
    main(["--prefix", str(pre), "--out", str(out), "--centers", "0.5"])
    with out.open() as fh:
        rows = list(_csv.DictReader(fh))
    assert {r["metric"] for r in rows} == {"self_rr", "ndcg10_composition",
                                           "ndcg10_topology", "ndcg10_geometry"}
    self_row = next(r for r in rows if r["metric"] == "self_rr")
    assert float(self_row["mean"]) == pytest.approx(0.75)
    assert int(self_row["n_queries"]) == 2


# ======================================================================
# 6. Nowalls: la stanza tolta perde anche i propri muri.
# ======================================================================

# RPLAN disegna il perimetro e le pareti interne con DUE grigi diversi (79 e
# 128): la soglia storica `_WALL_MAX = 120` prende solo il primo. Il piano
# sintetico dei test sopra ha un solo grigio, quindi non mostrerebbe la
# differenza: qui se ne disegna uno con entrambi.
PERIMETER_RGB = (79, 79, 79)
PARTITION_RGB = (128, 128, 128)


def draw_plan_two_greys() -> np.ndarray:
    """Come `draw_plan`, ma le pareti interne sono piu' chiare del perimetro."""
    arr = draw_plan()
    wall = np.all(arr == WALL_RGB, axis=2)
    arr[wall] = PARTITION_RGB
    ys, xs = np.where(np.any(arr < 248, axis=2))
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    border = np.zeros(arr.shape[:2], dtype=bool)
    border[y0:y0 + 2 * WALL_HALF + 1, x0:x1 + 1] = True
    border[y1 - 2 * WALL_HALF:y1 + 1, x0:x1 + 1] = True
    border[y0:y1 + 1, x0:x0 + 2 * WALL_HALF + 1] = True
    border[y0:y1 + 1, x1 - 2 * WALL_HALF:x1 + 1] = True
    arr[border & np.all(arr == PARTITION_RGB, axis=2)] = PERIMETER_RGB
    return arr


@pytest.fixture
def plan_png_two_greys(tmp_path):
    path = tmp_path / "p42_two_greys.png"
    Image.fromarray(draw_plan_two_greys()).save(path)
    return path


def test_neutral_wall_mask_takes_both_greys_and_no_room_colour():
    arr = draw_plan_two_greys()
    wall = neutral_wall_mask(arr)
    assert wall[np.all(arr == PERIMETER_RGB, axis=2)].all()
    assert wall[np.all(arr == PARTITION_RGB, axis=2)].all()
    for rgb in ROOM_RGB:
        sel = np.all(arr == rgb, axis=2)
        if sel.any():
            assert not wall[sel].any()
    assert not wall[np.all(arr == 255, axis=2)].any()


def test_nowalls_erases_the_partitions_the_historical_mode_leaves(plan_png_two_greys):
    """Stesse stanze tolte, ma nessun muro residuo attorno al buco."""
    from scipy.ndimage import binary_dilation

    params = {"fraction": 0.5}
    before = np.asarray(Image.open(plan_png_two_greys).convert("RGB"))
    old_img, old_removed = make_partial_query(
        plan_png_two_greys, plan_meta(), "random", params, random.Random(42), True
    )
    new_img, new_removed = make_wiped_query(
        plan_png_two_greys, plan_meta(), "random", params, random.Random(42)
    )
    assert new_removed == old_removed and new_removed  # appaiate: stesse stanze

    # Regione svuotata dal solo riempimento (identica nelle due modalita'): il
    # bordo si misura da li', non dai pixel diventati bianchi, che nel nowalls
    # includono i muri appena cancellati e sposterebbero l'anello sui muri delle
    # stanze VICINE, che restano al loro posto per costruzione.
    filled, _ = make_partial_query(plan_png_two_greys, plan_meta(), "random",
                                   params, random.Random(42), False)
    core = np.all(np.asarray(filled) >= 248, axis=2) & ~np.all(before >= 248, axis=2)
    ring = binary_dilation(core, iterations=3) & ~core

    def residual_walls(img):
        return int((ring & neutral_wall_mask(np.asarray(img))).sum())

    assert residual_walls(old_img) > 0        # il difetto storico, documentato
    assert residual_walls(new_img) == 0       # il buco non ha piu' muri


def test_nowalls_only_removes_pixels(plan_png_two_greys):
    """Nessun muro nuovo: i pixel possono solo diventare bianchi."""
    before = np.asarray(Image.open(plan_png_two_greys).convert("RGB"))
    img, _ = make_wiped_query(plan_png_two_greys, plan_meta(), "random",
                              {"fraction": 0.5}, random.Random(42))
    after = np.asarray(img)
    changed = np.any(after != before, axis=2)
    assert np.all(after[changed] == 255)
    assert plan_area_removed(before, after) > 0.0


def test_damaged_query_nowalls_branch_matches_make_wiped_query(plan_png_two_greys):
    T = compose_transform(224, IMAGENET_MEAN, IMAGENET_STD)
    for strategy, params in (("nowalls_random", {"fraction": 0.5}),
                             ("nowalls_random", {"fraction": 0.0}),
                             ("nowalls_semantic", {"keep_types": [0, 2, 3]}),
                             ("nowalls_topology", {"max_degree": 1})):
        assert strategy in NOWALLS_STRATEGIES
        base = strategy.split("_", 1)[1]
        rng_a, rng_b = random.Random(49), random.Random(49)
        img, removed = make_wiped_query(plan_png_two_greys, plan_meta(), base,
                                        params, rng_a)
        got, info = damaged_query(plan_png_two_greys, plan_meta(), strategy, params,
                                  rng_b, True, T)
        assert torch.equal(got, T(img))
        assert rng_a.random() == rng_b.random()      # stesso consumo di rng
        assert info["removed_any"] == bool(removed)
        assert info["fallback"] is False


def test_nowalls_and_historical_remove_the_same_rooms(plan_png_two_greys):
    """L'appaiamento che serve al confronto: stesso seed -> stesse stanze."""
    for qi in (0, 3, 7, 13):
        for params in ({"fraction": 0.25}, {"fraction": 0.5}, {"fraction": 0.75}):
            _, a = make_partial_query(plan_png_two_greys, plan_meta(), "random",
                                      params, random.Random(42 + qi), True)
            _, b = make_wiped_query(plan_png_two_greys, plan_meta(), "random",
                                    params, random.Random(42 + qi))
            assert a == b


NOWALLS_RUNS = [
    ("nowalls-random f=0.25", "nowalls_random", {"fraction": 0.25}),
    ("nowalls-random f=0.5", "nowalls_random", {"fraction": 0.5}),
    ("nowalls-semantic", "nowalls_semantic", {"keep_types": [0, 2, 3]}),
    ("nowalls-topology", "nowalls_topology", {"max_degree": 1}),
]


def test_partial_runs_appends_nowalls_last():
    on = _pcfg(nowalls_random={"enabled": True, "fractions": [0.25, 0.5]},
               nowalls_semantic={"enabled": True, "keep_types": [0, 2, 3]},
               nowalls_topology={"enabled": True, "max_degree": 1})
    runs = partial_runs(on)
    assert runs[:len(HISTORICAL_RUNS)] == HISTORICAL_RUNS
    assert runs[len(HISTORICAL_RUNS):] == NOWALLS_RUNS
    assert [_label_slug(l) for l, _, _ in NOWALLS_RUNS] == [
        "nowalls-random-f0.25", "nowalls-random-f0.5",
        "nowalls-semantic", "nowalls-topology",
    ]


def test_nowalls_slug_does_not_collide_with_the_historical_random():
    """I file per-query delle due modalita' devono restare distinti."""
    import re
    from src.evaluation.robustness_auc import _file_re
    historical = "vision_x_partial-random-f0.5_valid.npz"
    nowalls = "vision_x_partial-nowalls-random-f0.5_valid.npz"
    assert _file_re("random").match(historical)
    assert _file_re("random").match(nowalls) is None
    assert _file_re("nowalls-random").match(nowalls)


def test_yaml_ships_nowalls_switched_off():
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "vision_retrieval.yaml")
    for key in ("nowalls_random", "nowalls_semantic", "nowalls_topology"):
        assert cfg.partial.strategies[key].enabled is False


# ======================================================================
# 7. Visualizzazioni: stesso danno della valutazione, tutte le famiglie.
# ======================================================================

def test_damaged_query_return_image_is_what_gets_encoded(plan_png_two_greys):
    T = compose_transform(224, IMAGENET_MEAN, IMAGENET_STD)
    ctx = make_patch_context(T, 16)
    for strategy, params in (("random", {"fraction": 0.5}), ("nowalls_random", {"fraction": 0.5}),
                             ("crop", {"fraction": 0.5}), ("patch", {"fraction": 0.5})):
        t1, i1 = damaged_query(plan_png_two_greys, plan_meta(), strategy, params,
                               random.Random(7), True, T, ctx)
        t2, i2, img = damaged_query(plan_png_two_greys, plan_meta(), strategy, params,
                                    random.Random(7), True, T, ctx, return_image=True)
        assert torch.equal(t1, t2) and i1 == i2, strategy
        again = ctx["finish"](img) if strategy == "patch" else T(img)
        assert torch.equal(t2, again), strategy
        assert ("removed" in i1) == (strategy in ("random", "nowalls_random")), strategy


def test_visualization_renders_every_damage_family(tmp_path, monkeypatch):
    from src.vision.utils import retrieval_visualization as viz

    by_name = {m.name: m for m in GALLERY_METAS}
    paths = []
    for m in GALLERY_METAS:
        p = tmp_path / f"{m.name}.png"
        Image.fromarray(draw_plan(m)).save(p)
        paths.append(p)
    monkeypatch.setattr(viz, "load_metadata", lambda p, *a, **k: by_name[Path(str(p)).stem])
    T = compose_transform(224, IMAGENET_MEAN, IMAGENET_STD)
    pipeline = _FakePipeline(paths, T)
    stem2row = {p.stem: i for i, p in enumerate(paths)}
    pcfg = OmegaConf.create({"seed": 42, "open_boundary": True, "strategies": {
        "random": {"enabled": False, "fractions": [0.5]},
        "semantic": {"enabled": False, "keep_types": [0, 2, 3]},
        "topology": {"enabled": False, "max_degree": 1},
        "crop": {"enabled": True, "fractions": [0.5], "tolerance": 0.05, "max_tries": 10000},
        "patch": {"enabled": True, "fractions": [0.5], "tolerance": 0.05},
        "nowalls_random": {"enabled": True, "fractions": [0.5]},
    }})
    viz._apply_scale(96)                       # pannelli piccoli: il test resta veloce
    out = tmp_path / "viz"
    args = SimpleNamespace(top_k=3, out_dir=str(out))
    viz._run_partial_viz(pipeline, GalleryAxes(GALLERY_METAS), stem2row, paths, pcfg, args,
                         make_patch_context(T, 16))
    for folder in ("crop_f0.5", "patch_f0.5", "nowalls-random_f0.5"):
        assert len(list((out / folder).glob("query_*.png"))) == len(paths), folder
