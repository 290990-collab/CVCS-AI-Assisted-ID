"""Secondary fusion: W + vision with the frozen head.

Tests whether the fusion gain depends on the vision branch not being trained on the damage.
Vision = `pespatial/gem` raw -> head `head_nowalls_conv.pt` -> whitening fit on train after the head
(`head+whiten-train`); graph = W of the same replica.

`late_fusion.run` and `fusion_select.cmd_check` run unchanged inside `head_patch()`, which swaps two module globals:
- `late_fusion.whiten`: raw vision vectors (gallery and queries) go through the head before the job's whitening;
- `check_branch` (late_fusion and fusion_select): vision files must carry exactly this head
  (`{"enabled": True, "file": HEAD_FILE}`) instead of none.
Before fusing, `check_query_final` verifies that head + whitening applied to the stored raw query vectors
give the final vectors the vision job searched with.

Commands (CPU):
    python -m src.evaluation.fusion_head paths   --seed S --split valid      # shell variables
    python -m src.evaluation.fusion_head run     <late_fusion run args>
    python -m src.evaluation.fusion_head check   <fusion_select check args>
    python -m src.evaluation.fusion_head select  <fusion_select select args>
    python -m src.evaluation.fusion_head compare --seed S                    # measures 2-3
On the test: same commands with --split test; alpha fixed from the replica's select_valid.json,
verdict decided by Δ_H (`test_verdict_b`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from src.evaluation import fusion_select as fs
from src.evaluation import late_fusion as lf
from src.evaluation import graph_config_select as rs
from src.graph import final_graph_configs as rc
from src.vision.models.retrieval_model import VisionRetrievalPipeline

HEAD_DIR = Path("embeddings/vision/pespatial/gem")
HEAD_FILE = "head_nowalls_conv.pt"
HEAD_SHA1 = "49a7405949ac9fe9abf1feb49c2dbbce6a3de60c"
HEAD_HIDDEN, HEAD_OUT = 1024, 256                 # configs/vision_retrieval.yaml `head:`
HEAD_META = {"enabled": True, "file": HEAD_FILE}  # as written by the vision job in qvec meta
VISION_HEAD_TAG = "vision_pespatial_gem_head-nowalls-conv+whiten-train"
QUERY_TOL = 1e-4                                  # head on GPU in the vision job, CPU here: float32 rounding up to ~1.5e-5


def file_sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- paths (one replica): vision_head/ and fusion_head/ under results/final_pipeline/, never the main fusion folders ---

def vision_head_dirs(seed: int, split: str, smoke: bool = False) -> dict:
    r = rc.root(smoke) / "vision_head"
    return {"PQ_DIR": str(r / "perquery" / f"s{int(seed)}" / split),
            "QV_DIR": str(r / "queryvec" / f"s{int(seed)}" / split)}


def head_paths(seed: int, split: str, smoke: bool = False) -> dict:
    main = rs.fusion_paths(seed, split, smoke)
    vh = vision_head_dirs(seed, split, smoke)
    fdir = rc.root(smoke) / "fusion_head" / f"s{int(seed)}"
    return {
        "W_ENC": main["W_ENC"], "W_CFG": main["W_CFG"],
        "GRAPH_QVEC": main["GRAPH_QVEC"], "GRAPH_PQ": main["GRAPH_PQ"],
        "VISION_QVEC": f"{vh['QV_DIR']}/{VISION_HEAD_TAG}", "VISION_PQ": f"{vh['PQ_DIR']}/{VISION_HEAD_TAG}",
        "VH_PQ_DIR": vh["PQ_DIR"], "VH_QV_DIR": vh["QV_DIR"],
        "FUSION_DIR": str(fdir / f"fusion_{split}"),
        "HEAD_JSON": str(fdir / f"head_fusion_{split}.json"),
        "CHECK_JSON": str(fdir / f"check_{split}.json"),
        "CHECK_RESET_JSON": str(fdir / f"check_{split}_reset.json"),
        "WAIVER_JSON": str(fdir / f"c3_waiver_{split}.json"),
        "SELECT_JSON": str(fdir / f"select_{split}.json"),
        "SELECT_VALID_JSON": str(fdir / "select_valid.json"),
        "COMPARE_JSON": str(fdir / f"compare_{split}.json"),
        "MAIN_FUSION_DIR": main["FUSION_DIR"], "MAIN_SELECT_JSON": main["SELECT_JSON"],
    }


# --- head and patch ---

class _HeadStub(VisionRetrievalPipeline):
    """Pipeline without encoder: only `_apply_head` is used."""

    def __init__(self, head):
        self.device = "cpu"
        self.head = head
        self.whiten_mean = self.whiten_matrix = None


_HEADS: dict = {}


def load_fixed_head(in_dim: int):
    """The sha1-pinned head on CPU (cached per input dim)."""
    if in_dim not in _HEADS:
        sha1 = file_sha1(HEAD_DIR / HEAD_FILE)
        if sha1 != HEAD_SHA1:
            raise ValueError(f"{HEAD_DIR / HEAD_FILE}: sha1 {sha1[:12]}, expected {HEAD_SHA1[:12]}")
        from src.vision.models.projection_head import load_head
        _HEADS[in_dim] = load_head(str(HEAD_DIR), in_dim, HEAD_HIDDEN, HEAD_OUT, "cpu", filename=HEAD_FILE)
    return _HEADS[in_dim]


def apply_head(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    return _HeadStub(load_fixed_head(vectors.shape[1]))._apply_head(vectors)


def _check_vision_head_meta(datas: dict) -> dict:
    for f, d in datas.items():
        if d.meta.get("head") != HEAD_META:
            raise ValueError(f"vision f={f}: head {d.meta.get('head')!r}, expected {HEAD_META!r} "
                             "(fusion with head, status.md §59.H)")
    return {f: replace(d, meta={**d.meta, "head": None}) for f, d in datas.items()}


@contextmanager
def head_patch():
    """`late_fusion.whiten` = whiten(head(raw)); `check_branch` accepts only HEAD_META."""
    orig_whiten, orig_lf_cb, orig_fs_cb = lf.whiten, lf.check_branch, fs.check_branch

    def whiten_with_head(vectors, whiten_mean, whiten_matrix):
        return orig_whiten(apply_head(vectors), whiten_mean, whiten_matrix)

    def check_branch_head(datas, branch, split, shared_sha1, n_shared):
        if branch == "vision":
            datas = _check_vision_head_meta(datas)
        return orig_lf_cb(datas, branch, split, shared_sha1, n_shared)

    lf.whiten, lf.check_branch, fs.check_branch = whiten_with_head, check_branch_head, check_branch_head
    try:
        yield
    finally:
        lf.whiten, lf.check_branch, fs.check_branch = orig_whiten, orig_lf_cb, orig_fs_cb


def check_query_final(vision_qvec, split: str, fractions, tol: float = QUERY_TOL) -> dict:
    """max |whiten(head(raw)) - final| per fraction; raises above `tol`."""
    datas = lf.load_branch_qvecs(vision_qvec, split, [float(f) for f in fractions], lf.VISION_STRATEGY)
    _check_vision_head_meta(datas)
    out = {}
    for f, d in datas.items():
        if d.vectors_final is None:
            raise ValueError(f"vision f={f}: no final vectors in the qvec file")
        v = lf.whiten(apply_head(d.vectors), d.whiten_mean, d.whiten_matrix)
        out[str(f)] = float(np.abs(v - d.vectors_final).max()) if len(v) else 0.0
        print(f"[fusion_head] f={f}: max |whiten(head(raw)) - job final| = {out[str(f)]:.2e}")
        if out[str(f)] > tol:
            raise ValueError(f"vision f={f}: head + whitening here differ from the job's final vectors "
                             f"by {out[str(f)]:.2e} > {tol:g}")
    return out


# --- measures 2-3 (measure 1 = verdict in select_valid.json) ---

def outcome(helps: bool, ci_lo: float, ci_hi: float) -> str:
    """One of four outcomes, from measure 1 and the CI of Δ_G."""
    if not helps:
        return "il guadagno dipendeva dal vision non allenato sul danno"
    if ci_hi < 0:
        return "il guadagno resta ma si riduce"
    if ci_lo > 0:
        return "il guadagno cresce con la head"
    return "il guadagno non dipende dall'allenamento del vision"


def test_verdict_b(delta_h: dict) -> str:
    """Test verdict decided by Δ_H, paired CI."""
    if delta_h["ci_lo"] > 0:
        return "«il guadagno cresce con la head» confermato"
    if delta_h["ci_hi"] < 0:
        return "«il guadagno cresce con la head» smentito"
    return "«il guadagno cresce con la head» non confermato"


def compare(seed: int, split: str = "valid", smoke: bool = False) -> dict:
    p = head_paths(seed, split, smoke)
    head_sel = json.loads(Path(p["SELECT_JSON"]).read_text())
    main_sel = json.loads(Path(p["MAIN_SELECT_JSON"]).read_text())
    for what, sel, fdir in (("head", head_sel, p["FUSION_DIR"]), ("main", main_sel, p["MAIN_FUSION_DIR"])):
        if sel.get("split") != split or Path(str(sel.get("fusion_dir", ""))).resolve() != Path(fdir).resolve():
            raise ValueError(f"{what} select json does not describe {fdir} on {split}")
    a_h, a_m = float(head_sel["alpha_star"]), float(main_sel["alpha_star"])
    H, comp_h = fs._fusion_endpoints(p["FUSION_DIR"], split, a_h, fs.FUSED_STRATEGY, "vision-graph")
    M, comp_m = fs._fusion_endpoints(p["MAIN_FUSION_DIR"], split, a_m, fs.FUSED_STRATEGY, "vision-graph")
    if len({d.meta["gallery"]["sha1"] for d in list(H.values()) + list(M.values())}) != 1:
        raise ValueError("the two fusions use different galleries")
    names = H[a_h].names
    al = lambda d: fs.align_by_name(names, d.names, d.auc)  # noqa: E731
    st = fs.complementarity_stats(al(H[a_h]), al(H[comp_h]), al(M[a_m]), al(M[comp_m]), rule="four_way")
    helps = bool(head_sel["verdict"]["helps"])
    res = {
        "seed": int(seed), "split": split, "written": datetime.now().isoformat(timespec="seconds"),
        "rule": "status.md §59.H", "n": st["n"],
        "measure1_head_fusion_helps": head_sel["verdict"],
        "alpha_star": {"head": a_h, "main": a_m},
        "best_component": {"head": comp_h, "main": comp_m},
        "measure2_delta_H": st["AUC_F_minus_AUC_C"],
        "measure3_delta_G": st["D"],
        "gain_head": st["G_F"], "gain_main": st["G_C"],
        "outcome": outcome(helps, st["D"]["ci_lo"], st["D"]["ci_hi"]),
    }
    if split == "test":
        # weights must be fixed from the valid
        for what, sel in (("head", head_sel), ("main", main_sel)):
            if sel.get("alpha_star_rule") != "fixed":
                raise ValueError(f"{what} select json on the test: alpha not fixed from the valid")
        res["test_verdict_B"] = {"rule": "status.md §62 B (Δ_H, CI 95%)",
                                 "text": test_verdict_b(st["AUC_F_minus_AUC_C"])}
    return res


# --- CLI ---

def _write_once(path: Path, obj) -> None:
    if path.exists():
        raise SystemExit(f"!! {path} exists already: nothing is overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("uso: fusion_head {paths|run|check|select|compare} ...")
    cmd, rest = argv[0], argv[1:]
    if cmd == "paths":
        ap = argparse.ArgumentParser(prog="fusion_head paths")
        ap.add_argument("--seed", type=int, required=True)
        ap.add_argument("--split", default="valid")
        ap.add_argument("--smoke", action="store_true")
        x = ap.parse_args(rest)
        for k, v in head_paths(x.seed, x.split, x.smoke).items():
            print(f"{k}={shlex.quote(str(v))}")
    elif cmd == "run":
        args = lf.parse_args(["run", *rest])
        diffs = check_query_final(args.vision_qvec, args.split, args.fractions)
        with head_patch():
            lf.run(args)
        side = Path(args.out).parent / f"head_fusion_{args.split}.json"
        _write_once(side, {"rule": "status.md §59.H", "head_file": str(HEAD_DIR / HEAD_FILE),
                           "head_sha1": HEAD_SHA1, "query_final_max_abs_diff": diffs,
                           "query_tol": QUERY_TOL, "argv": sys.argv,
                           "written": datetime.now().isoformat(timespec="seconds")})
    elif cmd == "check":
        with head_patch():
            fs.main(["check", *rest])
    elif cmd == "select":
        fs.main(["select", *rest])
    elif cmd == "compare":
        ap = argparse.ArgumentParser(prog="fusion_head compare")
        ap.add_argument("--seed", type=int, required=True)
        ap.add_argument("--split", default="valid")
        ap.add_argument("--smoke", action="store_true")
        x = ap.parse_args(rest)
        res = compare(x.seed, x.split, x.smoke)
        _write_once(Path(head_paths(x.seed, x.split, x.smoke)["COMPARE_JSON"]), res)
        d2, d3 = res["measure2_delta_H"], res["measure3_delta_G"]
        print(f"[fusion_head] s{x.seed}: misura 1 {res['measure1_head_fusion_helps']['text']} · "
              f"Δ_H {d2['mean']:+.4f} [{d2['ci_lo']:+.4f}, {d2['ci_hi']:+.4f}] · "
              f"Δ_G {d3['mean']:+.4f} [{d3['ci_lo']:+.4f}, {d3['ci_hi']:+.4f}] → {res['outcome']}")
        if "test_verdict_B" in res:
            print(f"[fusion_head] test (§62 B): {res['test_verdict_B']['text']}")
    else:
        raise SystemExit(f"!! comando sconosciuto: {cmd}")


if __name__ == "__main__":
    main()
