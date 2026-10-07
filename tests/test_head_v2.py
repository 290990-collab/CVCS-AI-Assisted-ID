
"""
Smoke test CPU of head v2: residual head on the query only, after the frozen whitening.

- at init the head is the identity (z = v): epoch 0 = frozen vision;
- checkpoint round-trips and carries the whitening it was trained on;
- run tag gets `+qhead-v2` only when the query head is on;
- in the pipeline the head touches the query only, the gallery stays the frozen whitened vector, the
  checkpoint whitening is imposed; option off = historical query path;
- plain fusion refuses a head-v2 vision file; inside `head_v2_patch` the head applies to query vectors, not gallery.

Run: python -m pytest tests/test_head_v2.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.vision.models.projection_head import ResidualQueryHead, load_query_head
from src.vision.training import train_head_v2 as t

D = 16


def _whitening(rng):
    mean = rng.normal(size=(1, D)).astype(np.float32)
    matrix = rng.normal(size=(D, D)).astype(np.float32)
    return mean, matrix


def _ckpt(tmp_path, head, mean, matrix, name="head_v2.pt"):
    torch.save({"state_dict": head.state_dict(), "dim": D, "hidden_dim": 8, "best_epoch": 0,
                "whiten_mean": torch.from_numpy(mean), "whiten_matrix": torch.from_numpy(matrix),
                "whitening_sha1": t.whitening_sha1(mean, matrix)}, tmp_path / name)
    return tmp_path / name


def test_identity_at_init():
    torch.manual_seed(0)
    v = torch.nn.functional.normalize(torch.randn(5, D), dim=-1)
    assert torch.allclose(ResidualQueryHead(D, 8)(v), v, atol=1e-6)


def test_checkpoint_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    mean, matrix = _whitening(rng)
    head = ResidualQueryHead(D, 8)
    torch.nn.init.normal_(head.net[-1].weight)
    p = _ckpt(tmp_path, head, mean, matrix)
    h2, ck = load_query_head(p)
    x = torch.randn(3, D)
    assert torch.allclose(h2(x), head.eval()(x))
    assert t.whitening_sha1(ck["whiten_mean"].numpy(), ck["whiten_matrix"].numpy()) == ck["whitening_sha1"]


def test_whiten_l2_is_the_job_whitening():
    from src.vision.models.retrieval_model import VisionRetrievalPipeline
    rng = np.random.default_rng(1)
    mean, matrix = _whitening(rng)
    raw = rng.normal(size=(7, D)).astype(np.float32)
    p = VisionRetrievalPipeline.__new__(VisionRetrievalPipeline)
    p.whiten_mean, p.whiten_matrix = mean, matrix
    assert np.allclose(t.whiten_l2(raw, mean, matrix), p._apply_whitening(raw), atol=1e-6)


def test_tag():
    from src.vision.utils.config import transform_tag
    base = {"head": {"enabled": False}, "whitening": {"enabled": True, "fit_split": "train"}}
    assert transform_tag(OmegaConf.create(base)) == "whiten-train"
    on = {**base, "query_head": {"enabled": True, "file": "head_v2.pt"}}
    assert transform_tag(OmegaConf.create(on)) == "whiten-train+qhead-v2"
    off = {**base, "query_head": {"enabled": False, "file": "head_v2.pt"}}
    assert transform_tag(OmegaConf.create(off)) == "whiten-train"


def _pipeline(raw, mean, matrix):
    from src.vision.models.retrieval_model import VisionRetrievalPipeline
    p = VisionRetrievalPipeline(torch.nn.Identity(), transform=lambda x: x, device="cpu")
    p.raw_embeddings = raw
    p.image_paths = [f"{i}.png" for i in range(len(raw))]
    p.whiten_mean, p.whiten_matrix = mean, matrix
    p.embeddings = p._apply_whitening(raw)
    p.build_index()
    return p


def test_pipeline_query_only_and_imposed_whitening(tmp_path):
    from src.vision.evaluation.evaluate import attach_query_head
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(30, D)).astype(np.float32)
    mean, matrix = _whitening(rng)
    refit_mean, refit_matrix = mean + 0.1, matrix * 1.01            # refit differs on another node
    head = ResidualQueryHead(D, 8)
    torch.nn.init.normal_(head.net[-1].weight, std=0.5)
    _ckpt(tmp_path, head, mean, matrix)
    cfg = OmegaConf.create({"head": {"enabled": False}, "whitening": {"enabled": True, "fit_split": "train"},
                            "retrieval": {"save_dir": str(tmp_path)}, "query_head": {"enabled": True,
                                                                                    "file": "head_v2.pt"}})
    p = _pipeline(raw, refit_mean, refit_matrix)
    q = torch.from_numpy(raw[:1])
    before = p.query(q, top_k=3, return_embeddings=True)[2]          # option off: historical path
    assert np.allclose(before, p._apply_whitening(raw[:1]))
    attach_query_head(p, cfg, "cpu")
    assert np.array_equal(p.whiten_matrix, matrix) and np.array_equal(p.whiten_mean, mean)
    assert np.allclose(p.embeddings, t.whiten_l2(raw, mean, matrix), atol=1e-6)   # gallery: no head
    final = p.query(q, top_k=3, return_embeddings=True)[2]
    expect = head.eval()(torch.from_numpy(t.whiten_l2(raw[:1], mean, matrix))).detach().numpy()
    assert np.allclose(final, expect, atol=1e-5)


def test_attach_refuses_head_or_transductive_whitening(tmp_path):
    from src.vision.evaluation.evaluate import attach_query_head
    base = {"retrieval": {"save_dir": str(tmp_path)}, "query_head": {"enabled": True}}
    with pytest.raises(ValueError):
        attach_query_head(None, OmegaConf.create({**base, "head": {"enabled": True},
                                                  "whitening": {"enabled": True, "fit_split": "train"}}), "cpu")
    with pytest.raises(ValueError):
        attach_query_head(None, OmegaConf.create({**base, "head": {"enabled": False},
                                                  "whitening": {"enabled": True, "fit_split": "all"}}), "cpu")


def test_plain_fusion_refuses_head_v2_files():
    from src.evaluation import late_fusion as lf
    meta = {"branch": "vision", "split": "valid", "mode": "partial", "gallery": {"sha1": "s", "n": 3},
            "head": None, "query_head": {"enabled": True, "file": "head_v2.pt"}}
    with pytest.raises(ValueError, match="query head"):
        lf.check_branch({0.0: SimpleNamespace(meta=meta)}, "vision", "valid", "s", 3)


def test_patch_applies_head_to_queries_not_gallery(monkeypatch):
    from src.evaluation import fusion_head_v2 as fv
    from src.evaluation import late_fusion as lf
    rng = np.random.default_rng(3)
    mean, matrix = _whitening(rng)
    head = ResidualQueryHead(D, 8)
    torch.nn.init.normal_(head.net[-1].weight, std=0.5)
    monkeypatch.setattr(fv, "_HEAD", {"head": head.eval(), "whitening_sha1": "x"})
    raw_g = rng.normal(size=(9, D)).astype(np.float32)
    monkeypatch.setattr(lf, "load_vision_gallery", lambda *a, **k: raw_g)
    orig_whiten = lf.whiten
    with fv.head_v2_patch():
        g = lf.whiten(lf.load_vision_gallery("x", [], "s"), mean, matrix)
        q = lf.whiten(raw_g[[1, 2]], mean, matrix)
    assert lf.whiten is orig_whiten                                    # restored
    assert np.allclose(g, t.whiten_l2(raw_g, mean, matrix), atol=1e-6)
    expect = head(torch.from_numpy(t.whiten_l2(raw_g[[1, 2]], mean, matrix))).detach().numpy()
    assert np.allclose(q, expect, atol=1e-5) and not np.allclose(q, g[[1, 2]], atol=1e-3)
