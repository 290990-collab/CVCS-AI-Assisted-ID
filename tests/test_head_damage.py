# tests/test_head_damage.py

"""
Smoke test CPU del danno configurabile della projection head (15 set 2026,
`training.damage`, status.md §46): coppie (`projection_pairs.py`), probe
(`retrieval_probe.py`) e controllo di coerenza nel training (`train_projection.py`).

Il default `random` deve lasciare tutto com'era (render, chiave della cache, file
vecchi senza l'informazione del danno); `nowalls_random` deve togliere le stesse
stanze con lo stesso rng e non mescolarsi mai con coppie o probe `random`.

Esecuzione: python -m pytest tests/test_head_damage.py -v
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from src.vision.data import projection_pairs as pp
from src.vision.data.vision_damage import (
    HEAD_DAMAGES,
    check_head_damage,
    make_wiped_query,
    neutral_wall_mask,
    room_damage_image,
)
from src.vision.data.vision_partial_query import make_partial_query
from src.vision.training import retrieval_probe as rp
from src.vision.training.train_projection import train
from tests.test_head_probe import _cfg, _write_fake_run
from tests.test_vision_damage import draw_plan_two_greys, plan_meta

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"fraction": 0.5}


@pytest.fixture
def plan_png(tmp_path):
    path = tmp_path / "p42.png"
    Image.fromarray(draw_plan_two_greys()).save(path)
    return path


def test_check_head_damage_accepts_only_the_two_room_renders():
    assert HEAD_DAMAGES == ("random", "nowalls_random")
    for ok in HEAD_DAMAGES:
        assert check_head_damage(ok) == ok
    for bad in ("crop", "patch", "semantic", "nowalls_semantic", "nowalls"):
        with pytest.raises(ValueError, match="training.damage"):
            check_head_damage(bad)


def test_yaml_ships_the_historical_damage():
    cfg = OmegaConf.load(ROOT / "configs" / "vision_retrieval.yaml")
    assert cfg.training.damage == "random"


def test_random_is_literally_the_historical_render(plan_png):
    img, removed = room_damage_image(plan_png, plan_meta(), "random", PARAMS, random.Random(7), True)
    ref, ref_removed = make_partial_query(plan_png, plan_meta(), "random", PARAMS, random.Random(7), True)
    assert removed == ref_removed
    assert np.array_equal(np.asarray(img), np.asarray(ref))


def test_nowalls_removes_the_same_rooms_and_keeps_the_rng_in_step(plan_png):
    rng_old, rng_new = random.Random(7), random.Random(7)
    old, old_removed = room_damage_image(plan_png, plan_meta(), "random", PARAMS, rng_old, True)
    new, new_removed = room_damage_image(plan_png, plan_meta(), "nowalls_random", PARAMS, rng_new, True)
    ref, _ = make_wiped_query(plan_png, plan_meta(), "random", PARAMS, random.Random(7))
    assert new_removed == old_removed and new_removed
    assert np.array_equal(np.asarray(new), np.asarray(ref))
    assert rng_old.random() == rng_new.random()           # flip/rot90 successivi identici
    walls = lambda im: int(neutral_wall_mask(np.asarray(im)).sum())
    assert walls(new) < walls(old)                         # i muri della stanza tolta spariscono


def _pairs_cfg(**training):
    base = {"seed": 42, "mask_fraction": [0.1, 0.5], "augment": True, "open_boundary": True}
    return OmegaConf.create({"training": {**base, **training}})


def test_render_cache_key_is_unchanged_by_default_and_split_by_the_damage(tmp_path):
    save_dir = tmp_path / "embeddings" / "vision" / "m" / "v"
    # la chiave scritta prima del 15 set: le cache gia' su disco restano valide
    historic = hashlib.md5(b"seed=42|frac=(0.1, 0.5)|aug=True|open=True").hexdigest()[:12]
    assert pp._render_cache_dir(save_dir, _pairs_cfg()).name == historic
    assert pp._render_cache_dir(save_dir, _pairs_cfg(damage="random")).name == historic
    nowalls = pp._render_cache_dir(save_dir, _pairs_cfg(damage="nowalls_random"))
    assert nowalls.name != historic and nowalls.parent.name == "_render_cache"


def test_pairs_view_renders_the_configured_damage(plan_png, monkeypatch):
    monkeypatch.setattr(pp, "load_metadata", lambda path: plan_meta())

    def view(damage=None):
        extra = {} if damage is None else {"damage": damage}
        ds = pp._PartialViewDataset([str(plan_png)], None, 2, (0.5, 0.5), True, True, 42, None, **extra)
        return np.asarray(ds._view(plan_png.stem, 1, str(plan_png)))

    rng = random.Random(f"42:{plan_png.stem}:1")                 # stesso seeding di `_view`
    frac = rng.uniform(0.5, 0.5)
    img, _ = room_damage_image(plan_png, plan_meta(), "nowalls_random", {"fraction": frac}, rng, True)
    assert np.array_equal(view("nowalls_random"), np.asarray(pp._augment(img, rng)))
    assert not np.array_equal(view("nowalls_random"), view("random"))
    assert np.array_equal(view(), view("random"))                 # default = render storico


@pytest.mark.parametrize("damage", [None, "random", "nowalls_random"])
def test_probe_renders_and_records_the_configured_damage(tmp_path, plan_png, monkeypatch, damage):
    import src.vision.evaluation.evaluate as ev

    tf = transforms.Compose([transforms.Resize((16, 20)), transforms.ToTensor()])
    paths = [str(plan_png), str(tmp_path / "q1.png"), str(tmp_path / "q2.png")]
    (tmp_path / "image_paths.json").write_text(json.dumps(paths))
    fake = SimpleNamespace(image_paths=paths, transform=tf, device="cpu",
                           encoder=lambda x: x.flatten(1)[:, :32])
    monkeypatch.setattr(ev, "load_pipeline", lambda cfg: fake)
    monkeypatch.setattr(rp, "probe_rows", lambda paths, cfg: [0])
    monkeypatch.setattr(rp, "load_metadata", lambda path: plan_meta())
    training = {"probe": {"num_queries": 1, "seed": 1042, "fractions": [0.5]}}
    if damage is not None:
        training["damage"] = damage
    cfg = OmegaConf.create({
        "retrieval": {"save_dir": str(tmp_path)},
        "eval": {"num_queries": 1, "seed": 42, "split": "valid", "gallery_names": None,
                 "k_values": [1, 10]},
        "training": training, "partial": {"open_boundary": True},
        "head": {"enabled": False}, "whitening": {"enabled": False},
    })

    with np.load(rp.build_probe(cfg)) as z:
        meta = json.loads(str(z["meta"].item()))
        q_raw = z["q_raw"]
    expected = damage or "random"
    assert meta["strategy"] == expected
    img, _ = room_damage_image(plan_png, plan_meta(), expected, {"fraction": 0.5},
                               random.Random(1042 + 0), True)
    np.testing.assert_allclose(q_raw[0, 0], tf(img).flatten()[:32].numpy(), rtol=1e-6)


def test_training_refuses_pairs_or_probe_with_another_damage(tmp_path):
    _write_fake_run(tmp_path)          # coppie e probe «vecchie»: nessun danno salvato = random
    over = dict(training={"selection": "probe_partial", "damage": "nowalls_random"},
                head={"file": "head_nowalls.pt"})
    with pytest.raises(ValueError, match="training.damage"):
        train(_cfg(tmp_path, **over))

    with np.load(tmp_path / "pairs.npz") as z:
        pairs = {k: z[k] for k in z.files}
    np.savez(tmp_path / "pairs.npz", **pairs, damage=np.array("nowalls_random"))
    with pytest.raises(ValueError, match="training.damage"):
        train(_cfg(tmp_path, **over))                # coppie giuste, probe ancora random

    with np.load(tmp_path / rp.PROBE_FILE) as z:
        probe = {k: z[k] for k in z.files}
    meta = json.loads(str(probe["meta"].item()))
    probe["meta"] = np.array(json.dumps({**meta, "strategy": "nowalls_random"}))
    np.savez(tmp_path / rp.PROBE_FILE, **probe)
    train(_cfg(tmp_path, **over))
    assert (tmp_path / "head_nowalls.pt").exists()
    assert not (tmp_path / "head.pt").exists()


def test_training_with_nowalls_pairs_refuses_the_default_damage(tmp_path):
    _write_fake_run(tmp_path)
    with np.load(tmp_path / "pairs.npz") as z:
        pairs = {k: z[k] for k in z.files}
    np.savez(tmp_path / "pairs.npz", **pairs, damage=np.array("nowalls_random"))
    with pytest.raises(ValueError, match="training.damage"):
        train(_cfg(tmp_path))                        # val_loss, config senza danno = random
