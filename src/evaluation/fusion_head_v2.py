"""Fusion W + vision with the query head v2, one replica.

Head v2 acts on the query only, after the frozen whitening; the vision gallery stays frozen.
`late_fusion.run` and `fusion_select.cmd_check` run unchanged inside `head_v2_patch()`, which swaps three module globals:
- `late_fusion.load_vision_gallery`: the raw gallery comes back marked (`_Gallery`), so that
- `late_fusion.whiten` applies head v2 after the whitening only to arrays that are not the gallery (the queries);
- `check_branch` (late_fusion and fusion_select): vision files must carry exactly this query head
  (`{"enabled": True, "file": "head_v2.pt"}`) and the checkpoint's whitening (sha1).
Before fusing, `check_query_final` verifies head(whiten(raw)) = the final vectors the vision job searched with.

Graph = W of the same replica, from its second-round evaluation (`round2.graph_source`, own gallery).

Commands (CPU):
    python -m src.evaluation.fusion_head_v2 paths   --seed S --split valid
    python -m src.evaluation.fusion_head_v2 run     <late_fusion run args>
    python -m src.evaluation.fusion_head_v2 check   <fusion_select check args>
    python -m src.evaluation.fusion_head_v2 select  <fusion_select select args>
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.evaluation import fusion_select as fs
from src.evaluation import late_fusion as lf
from src.evaluation import multiseed_runs as r2
from src.graph import final_graph_configs as rc

HEAD_PATH = Path("embeddings/vision/pespatial/gem/head_v2.pt")
QUERY_HEAD_META = {"enabled": True, "file": "head_v2.pt"}
VISION_TAG = "vision_pespatial_gem_whiten-train+qhead-v2"
QUERY_TOL = 1e-4                       # head on GPU in the vision job, CPU here

_HEAD: dict = {}


def head_paths(seed: int, split: str, smoke: bool = False) -> dict:
    r = rc.root(smoke)
    vh = r / "vision_head_v2"
    gq, gp = r2.graph_source(split, *r2.W, seed, seed)
    fdir = r / "fusion_head_v2" / f"s{int(seed)}"
    return {
        "W_ENC": r2.W[0], "W_CFG": r2.W[1], "GRAPH_QVEC": gq, "GRAPH_PQ": gp,
        "VH_PQ_DIR": str(vh / "perquery" / f"s{int(seed)}" / split),
        "VH_QV_DIR": str(vh / "queryvec" / f"s{int(seed)}" / split),
        "VISION_QVEC": str(vh / "queryvec" / f"s{int(seed)}" / split / VISION_TAG),
        "VISION_PQ": str(vh / "perquery" / f"s{int(seed)}" / split / VISION_TAG),
        "FUSION_DIR": str(fdir / f"fusion_{split}"),
        "HEAD_JSON": str(fdir / f"head_v2_fusion_{split}.json"),
        "CHECK_JSON": str(fdir / f"check_{split}.json"),
        "CHECK_RESET_JSON": str(fdir / f"check_{split}_reset.json"),
        "WAIVER_JSON": str(fdir / f"c3_waiver_{split}.json"),
        "SELECT_JSON": str(fdir / f"select_{split}.json"),
        "SELECT_VALID_JSON": str(fdir / "select_valid.json"),
    }


def load_head():
    """Head v2 on CPU and the sha1 of the whitening it was trained on (cached)."""
    if "head" not in _HEAD:
        from src.vision.models.projection_head import load_query_head
        _HEAD["head"], ckpt = load_query_head(HEAD_PATH, "cpu")
        _HEAD["whitening_sha1"] = ckpt["whitening_sha1"]
    return _HEAD["head"], _HEAD["whitening_sha1"]


def apply_head(vectors: np.ndarray) -> np.ndarray:
    head, _ = load_head()
    with torch.no_grad():
        return head(torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32))).numpy().astype("float32")


class _Gallery(np.ndarray):
    """Marker: the raw vision gallery (no head v2 after whitening)."""


def _check_meta(datas: dict) -> dict:
    from src.vision.training.train_head_v2 import whitening_sha1
    _, wsha = load_head()
    for f, d in datas.items():
        if d.meta.get("query_head") != QUERY_HEAD_META:
            raise ValueError(f"vision f={f}: query_head {d.meta.get('query_head')!r}, expected {QUERY_HEAD_META!r}")
        if whitening_sha1(d.whiten_mean, d.whiten_matrix) != wsha:
            raise ValueError(f"vision f={f}: whitening of the file is not the head v2 checkpoint's")
    return {f: replace(d, meta={k: v for k, v in d.meta.items() if k != "query_head"}) for f, d in datas.items()}


@contextmanager
def head_v2_patch():
    orig = (lf.whiten, lf.load_vision_gallery, lf.check_branch, fs.check_branch)

    def load_gallery_marked(*a, **k):
        return orig[1](*a, **k).view(_Gallery)

    def whiten_v2(vectors, whiten_mean, whiten_matrix):
        out = orig[0](np.asarray(vectors), whiten_mean, whiten_matrix)
        return out if isinstance(vectors, _Gallery) else apply_head(out)

    def check_branch_v2(datas, branch, split, shared_sha1, n_shared):
        if branch == "vision":
            datas = _check_meta(datas)
        return orig[2](datas, branch, split, shared_sha1, n_shared)

    lf.whiten, lf.load_vision_gallery = whiten_v2, load_gallery_marked
    lf.check_branch = fs.check_branch = check_branch_v2
    try:
        yield
    finally:
        lf.whiten, lf.load_vision_gallery, lf.check_branch, fs.check_branch = orig


def check_query_final(vision_qvec, split: str, fractions, tol: float = QUERY_TOL) -> dict:
    """max |head(whiten(raw)) - final| per fraction; raises above `tol`."""
    datas = lf.load_branch_qvecs(vision_qvec, split, [float(f) for f in fractions], lf.VISION_STRATEGY)
    _check_meta(datas)
    out = {}
    for f, d in datas.items():
        if d.vectors_final is None:
            raise ValueError(f"vision f={f}: no final vectors in the qvec file")
        v = apply_head(lf.whiten(d.vectors, d.whiten_mean, d.whiten_matrix))
        out[str(f)] = float(np.abs(v - d.vectors_final).max()) if len(v) else 0.0
        print(f"[fusion_head_v2] f={f}: max |head(whiten(raw)) - job final| = {out[str(f)]:.2e}")
        if out[str(f)] > tol:
            raise ValueError(f"vision f={f}: head v2 here differs from the job's final vectors by {out[str(f)]:.2e}")
    return out


def _write_once(path: Path, obj) -> None:
    if path.exists():
        raise SystemExit(f"!! {path} esiste gia': niente sovrascritture")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True))
    print(f"[fusion_head_v2] scritto {path}")


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("uso: fusion_head_v2 {paths|run|check|select} ...")
    cmd, rest = argv[0], argv[1:]
    if cmd == "paths":
        ap = argparse.ArgumentParser(prog="fusion_head_v2 paths")
        ap.add_argument("--seed", type=int, required=True)
        ap.add_argument("--split", default="valid")
        x = ap.parse_args(rest)
        for k, v in head_paths(x.seed, x.split).items():
            print(f"{k}={shlex.quote(str(v))}")
    elif cmd == "run":
        args = lf.parse_args(["run", *rest])
        diffs = check_query_final(args.vision_qvec, args.split, args.fractions)
        with head_v2_patch():
            lf.run(args)
        _, wsha = load_head()
        from src.vision.evaluation.evaluate import _file_sha1
        _write_once(Path(args.out).parent / f"head_v2_fusion_{args.split}.json",
                    {"rule": "status.md §64.2", "head_file": str(HEAD_PATH), "head_sha1": _file_sha1(HEAD_PATH),
                     "whitening_sha1": wsha, "query_final_max_abs_diff": diffs, "query_tol": QUERY_TOL,
                     "argv": sys.argv, "written": datetime.now().isoformat(timespec="seconds")})
    elif cmd == "check":
        with head_v2_patch():
            fs.main(["check", *rest])
    elif cmd == "select":
        fs.main(["select", *rest])
    else:
        raise SystemExit(f"!! comando sconosciuto: {cmd}")


if __name__ == "__main__":
    main()
