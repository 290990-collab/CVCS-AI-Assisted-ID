
"""
Smoke test CPU of head v2 preparation: mixed-damage views in pairs (`training.damage_mix`) and
multi-damage probe (`training.probe.damages`). Without the new keys nothing changes (incl. cache keys).

Run: python -m pytest tests/test_head_v2_prep.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.vision.data import projection_pairs as pp
from src.vision.data.preprocess import get_transform
from src.vision.data.vision_damage import make_patch_context
from src.vision.training import retrieval_probe as rp
from tests.test_vision_damage import draw_plan_two_greys, plan_meta

MIX = ["nowalls_random", "crop", "patch"]


def _cfg(**training):
    base = {"seed": 42, "mask_fraction": [0.1, 0.5], "augment": True, "open_boundary": True}
    return OmegaConf.create({"training": {**base, **training},
                             "model": {"kwargs": {"image_size": 224, "patch_size": 16}}})


@pytest.fixture
def plan_png(tmp_path):
    path = tmp_path / "p42.png"
    Image.fromarray(draw_plan_two_greys()).save(path)
    return path


def test_existing_cache_keys_are_unchanged():
    save_dir = Path("/x/vision/pespatial/gem")
    assert pp._render_cache_dir(save_dir, _cfg()).name == "b6245e020eed"
    assert pp._render_cache_dir(save_dir, _cfg(damage="nowalls_random")).name == "64da740303b8"


def test_mix_gets_its_own_cache_and_the_patch_canvas_enters_the_key():
    save_dir = Path("/x/vision/pespatial/gem")
    v2 = _cfg(damage="nowalls_random", damage_mix=MIX, mask_fraction=[0.25, 0.75])
    k = pp._render_cache_dir(save_dir, v2)
    assert k.name not in ("b6245e020eed", "64da740303b8") and k.parent.name == "_render_cache"
    other = OmegaConf.merge(v2, {"model": {"kwargs": {"patch_size": 14}}})
    assert pp._render_cache_dir(save_dir, other).name != k.name
    no_patch = _cfg(damage="nowalls_random", damage_mix=["nowalls_random", "crop"], mask_fraction=[0.25, 0.75])
    assert pp._render_cache_dir(save_dir, no_patch).name != k.name


def test_check_damage_mix():
    assert pp.check_damage_mix(None) is None
    assert pp.check_damage_mix(MIX) == tuple(MIX)
    for bad in ([], ["random"], ["crop", "crop"], ["semantic"]):
        with pytest.raises(ValueError, match="damage_mix"):
            pp.check_damage_mix(bad)


def _dataset(plan_png, tmp_path, views=6, mix=MIX):
    tf = get_transform(224)
    return pp._PartialViewDataset(
        [str(plan_png)], tf, views=views, frac_range=(0.25, 0.75), augment=True,
        open_boundary=True, seed=42, cache_dir=tmp_path / "cache", damage="nowalls_random",
        damage_mix=mix, patch_ctx=make_patch_context(tf, 16),
        crop_params={"tolerance": 0.05, "max_tries": 10000}, patch_params={"tolerance": 0.05})


def test_views_take_the_damages_in_turn(plan_png, tmp_path):
    (tmp_path / "cache").mkdir()
    ds = _dataset(plan_png, tmp_path)
    assert [ds.view_damage(v) for v in range(6)] == MIX * 2
    with pytest.raises(ValueError, match="multiple"):
        _dataset(plan_png, tmp_path, views=4)
    hist = pp._PartialViewDataset([str(plan_png)], get_transform(224), 3, (0.1, 0.5), True, True, 42,
                                  None, damage="nowalls_random")
    assert [hist.view_damage(v) for v in range(3)] == ["nowalls_random"] * 3


def test_every_damage_renders_deterministically_and_reads_back_from_cache(plan_png, tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "load_metadata", lambda path: plan_meta())
    (tmp_path / "cache").mkdir()
    ds = _dataset(plan_png, tmp_path)
    original = np.asarray(Image.open(plan_png).convert("RGB"))
    first = [ds[k][0] for k in range(len(ds))]
    for t in first:
        assert t.shape == (3, 224, 224) and torch.isfinite(t).all()
    files = sorted((tmp_path / "cache").glob("*.png"))
    assert len(files) == 6
    assert Image.open(tmp_path / "cache" / "p42_v2.png").size == (224, 224)   # patch = resized canvas
    for v in (0, 1):                                                          # rooms, crop: native image changed
        img = np.asarray(Image.open(tmp_path / "cache" / f"p42_v{v}.png").convert("RGB"))
        assert img.shape != original.shape or not np.array_equal(img, original)
    again = [ds[k][0] for k in range(len(ds))]                                # from the cache
    for a, b in zip(first, again):
        assert torch.equal(a, b)
    fresh = _dataset(plan_png, tmp_path / "other")
    (tmp_path / "other" / "cache").mkdir(parents=True)
    for k in range(len(ds)):                                                  # same seed -> same views
        assert torch.equal(fresh[k][0], first[k])


def test_mixed_probe_never_touches_the_current_probe_file(tmp_path):
    cfg = OmegaConf.create({"retrieval": {"save_dir": str(tmp_path)},
                            "training": {"probe": {"damages": MIX, "file": rp.PROBE_FILE,
                                                   "fractions": [0.25], "seed": 1}}})
    with pytest.raises(ValueError, match="probe.file"):
        rp._build_probe_mixed(cfg, None, [], [], {})
    (tmp_path / "probe_v2.npz").write_bytes(b"x")
    cfg.training.probe.file = "probe_v2.npz"
    with pytest.raises(FileExistsError):
        rp._build_probe_mixed(cfg, None, [], [], {})
