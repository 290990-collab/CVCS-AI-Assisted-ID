# tests/test_head_probe.py

"""
Smoke test CPU della fase B.6: probe di retrieval partial per la selezione della
head (`src/vision/training/retrieval_probe.py`, `train_projection.py`).
Niente dataset reale ne' encoder: vettori sintetici in una cartella temporanea.

Esecuzione: python -m pytest tests/test_head_probe.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.evaluation.perquery import gallery_sha1
from src.vision.models.projection_head import ProjectionHead, load_head
from src.vision.training import retrieval_probe as rp
from src.vision.training.train_projection import train
from src.vision.utils.config import transform_tag


def _brute_rr(q, g, self_rows, max_rank):
    rr = []
    for i in range(len(q)):
        s = g @ q[i]
        order = np.argsort(-s, kind="stable")
        rank = int(np.where(order == self_rows[i])[0][0]) + 1
        rr.append(1.0 / rank if rank <= max_rank else 0.0)
    return np.asarray(rr)


def test_self_reciprocal_ranks_match_brute_force_and_cutoff():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(300, 16)).astype("float32")
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    self_rows = rng.choice(300, size=40, replace=False)
    q = g[self_rows] + 0.8 * rng.normal(size=(40, 16)).astype("float32")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for max_rank in (5, 100):
        got = rp.self_reciprocal_ranks(torch.from_numpy(q), torch.from_numpy(g),
                                       torch.from_numpy(self_rows), max_rank, chunk=7)
        np.testing.assert_allclose(got.numpy(), _brute_rr(q, g, self_rows, max_rank), rtol=1e-6)


def test_transform_tag_separates_head_checkpoints():
    base = {"whitening": {"enabled": False, "fit_split": "train"}}
    tag = lambda head: transform_tag(OmegaConf.create({**base, "head": head}))
    assert tag({"enabled": True}) == "head"                        # nomi storici invariati
    assert tag({"enabled": True, "file": "head.pt"}) == "head"
    assert tag({"enabled": True, "file": "head_probe.pt"}) == "head-probe"
    assert tag({"enabled": False, "file": "head_probe.pt"}) == "raw"


def test_probe_rows_are_valid_and_disjoint_from_eval_queries(monkeypatch):
    paths = [f"/x/{i}.png" for i in range(400)]
    split_of = lambda p: "valid" if int(p.split("/")[-1][:-4]) % 2 else "train"
    monkeypatch.setattr(rp, "get_split", split_of)
    import src.vision.evaluation.evaluate as ev
    monkeypatch.setattr(ev, "get_split", split_of)
    cfg = OmegaConf.create({"eval": {"num_queries": 50, "seed": 42, "split": "valid"},
                            "training": {"probe": {"num_queries": 60, "seed": 1042}}})
    eval_rows = set(ev.sample_query_rows(paths, 50, 42, "valid"))
    rows = rp.probe_rows(paths, cfg)
    assert len(rows) == 60
    assert not (set(rows) & eval_rows)
    assert all(split_of(paths[r]) == "valid" for r in rows)


def _write_fake_run(save_dir, n=120, d=12, views=2, n_probe=20, fractions=(0.25, 0.5, 0.75)):
    rng = np.random.default_rng(1)
    gallery = rng.normal(size=(n, d)).astype("float32")
    np.save(save_dir / "embeddings.npy", gallery)
    paths = [f"/x/{i}.png" for i in range(n)]
    (save_dir / "image_paths.json").write_text(json.dumps(paths))
    splits = np.asarray(["train"] * 80 + ["valid"] * 40)
    anchors = gallery[:n]
    positives = anchors[:, None, :] + 0.3 * rng.normal(size=(n, views, d)).astype("float32")
    np.savez(save_dir / "pairs.npz", anchors=anchors, positives=positives, splits=splits)
    q_rows = np.arange(80, 80 + n_probe)
    q_raw = np.stack([gallery[q_rows] + f * rng.normal(size=(n_probe, d)).astype("float32")
                      for f in fractions])
    names = [str(i) for i in range(n)]
    meta = {"fractions": list(fractions), "max_k": 100, "gallery_sha1": gallery_sha1(names)}
    np.savez(save_dir / rp.PROBE_FILE, q_raw=q_raw, q_rows=q_rows,
             gallery_rows=np.arange(n), meta=np.array(json.dumps(meta)))
    return d


def _cfg(save_dir, **over):
    cfg = {
        "retrieval": {"save_dir": str(save_dir)},
        "model": {"device": "cpu", "name": "fake", "variant": "v"},
        "head": {"hidden_dim": 16, "out_dim": 8, "file": "head.pt"},
        "training": {"epochs": 3, "lr": 1e-3, "batch_size": 32, "weight_decay": 0.0,
                     "temperature": 0.1, "patience": 0, "seed": 0, "selection": "val_loss"},
        "wandb": {"enabled": False},
    }
    return OmegaConf.merge(OmegaConf.create(cfg), OmegaConf.create(over))


def test_probe_selection_writes_separate_checkpoint_and_history(tmp_path):
    d = _write_fake_run(tmp_path)
    train(_cfg(tmp_path, training={"selection": "probe_partial"}, head={"file": "head_probe.pt"}))
    assert (tmp_path / "head_probe.pt").exists()
    assert not (tmp_path / "head.pt").exists()                    # lo storico non si tocca
    hist = json.loads((tmp_path / "head_probe_history.json").read_text())
    aucs = [h["probe_auc"] for h in hist["history"]]
    # regola: migliora solo chi supera il migliore di > 1e-4
    assert hist["best_probe_auc"] == pytest.approx(aucs[hist["best_epoch"] - 1])
    assert hist["best_probe_auc"] >= max(aucs) - 1e-4
    # stessa run, checkpoint scelto dalla val-loss: salvato a parte e coerente con la history
    assert (tmp_path / "head_probe_valloss.pt").exists()
    vls = [h["val_loss"] for h in hist["history"]]
    assert hist["valloss_epoch"] == 1 + int(np.argmin(vls)) or \
        vls[hist["valloss_epoch"] - 1] <= min(vls) + 1e-4
    # il checkpoint salvato riproduce l'AUC dichiarata
    head = load_head(str(tmp_path), d, 16, 8, "cpu", filename="head_probe.pt")
    auc, _ = rp.probe_auc(head, rp.load_probe(tmp_path, "cpu"))
    assert auc == pytest.approx(hist["best_probe_auc"], abs=1e-6)


def test_probe_selection_refuses_to_overwrite_historic_head(tmp_path):
    _write_fake_run(tmp_path)
    with pytest.raises(ValueError, match="head.file"):
        train(_cfg(tmp_path, training={"selection": "probe_partial"}))


def test_default_selection_unchanged_writes_head_pt(tmp_path):
    _write_fake_run(tmp_path)
    train(_cfg(tmp_path))
    assert (tmp_path / "head.pt").exists()
    assert not (tmp_path / "head_probe_history.json").exists()


def test_training_is_reproducible_with_fixed_seed(tmp_path):
    _write_fake_run(tmp_path)
    cfg = _cfg(tmp_path, training={"selection": "probe_partial"}, head={"file": "head_probe.pt"})
    train(cfg)
    first = json.loads((tmp_path / "head_probe_history.json").read_text())["history"]
    train(cfg)
    second = json.loads((tmp_path / "head_probe_history.json").read_text())["history"]
    assert [h["probe_auc"] for h in first] == [h["probe_auc"] for h in second]


def test_load_probe_rejects_mismatched_gallery(tmp_path):
    _write_fake_run(tmp_path)
    (tmp_path / "image_paths.json").write_text(json.dumps([f"/x/{i}.png" for i in range(1, 121)]))
    with pytest.raises(ValueError, match="gallery"):
        rp.load_probe(tmp_path, "cpu")
