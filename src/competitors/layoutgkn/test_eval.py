"""LayoutGKN evaluated on the test split with the sha1-pinned checkpoint of the full comparison.

No training (best epoch 297 on the valid probe). Same code as `full.py` (`evaluate_view`): full gallery
(67,405), the 2000 test queries of W seed 42 (`qi` from W's test query-vector files), removed rooms
`Random(42 + qi)` checked per query against W's test evaluation, f in {0, .25, .5, .75} + whole plan.

Paired delta = AUC(LayoutGKN) - AUC(W seed 42) via `compare_auc` (B=10000, seed 0); also whole-plan nDCG@10 per axis.

Usage (GPU job, scripts/competitors/03_layoutgkn_test.sh):
    python -m src.competitors.layoutgkn.test_eval
    python -m src.competitors.layoutgkn.test_eval --smoke --root <scratch>     # 20 queries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.competitors.layoutgkn import UPSTREAM_COMMIT
from src.competitors.layoutgkn import graphs as lg
from src.competitors.layoutgkn import score as ls
from src.competitors.layoutgkn import train as lt
from src.competitors.layoutgkn.full import FRACTIONS, K_VALUES, MAX_K, SELF_SIM_TOL, evaluate_view
from src.competitors.layoutgkn.pilot import GALLERY_JSON, GALLERY_SHA1, PARTIAL_SEED
from src.data.rplan_metadata import load_metadata
from src.evaluation.perquery import PerQueryRecorder, gallery_sha1, load_perquery
from src.evaluation.relevance import AXES, GalleryAxes
from src.evaluation.robustness_auc import compare_auc, load_auc

SPLIT = "test"
RULE = "status.md §62 D"
CKPT = Path("embeddings/competitors/layoutgkn/asym/best.pt")
CKPT_SHA1 = "28fa4031e4b7bb8bbcd8696b28b161bff40c2790"
BEST_EPOCH = 297
W_TAG = "graph_gat_rg_comb_s42"
W_QVEC = "results/final_pipeline/queryvec/test/s42/" + W_TAG + "_partial-random-f{f}_test.npz"
W_PQ_PREFIX = "results/final_pipeline/perquery/test/s42/" + W_TAG
GATE = Path("results/final_pipeline/TEST_PREREGISTERED")
TAG = "lgkn_asym_s42"
FULL_K = 10


def log(msg: str) -> None:
    print(f"[lgkn-test {datetime.now():%H:%M:%S}] {msg}", flush=True)


def paths(root: Path) -> dict:
    return {"pq": root / "results/perquery/competitor_layoutgkn/asym_test",
            "out": root / "results/competitors/layoutgkn/test"}


def file_sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_w_test_queries() -> dict:
    """f -> (names, qi, removed rooms per query) of W's TEST evaluation; same queries in every f."""
    out = {}
    for f in FRACTIONS:
        z = np.load(W_QVEC.format(f=f), allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        assert meta["gallery"]["sha1"] == GALLERY_SHA1 and meta["partial_seed"] == PARTIAL_SEED
        assert meta["split"] == SPLIT, f"W qvec f={f}: split {meta['split']!r}"
        ptr, idx = z["removed_ptr"], z["removed_idx"]
        removed = [idx[ptr[k]:ptr[k + 1]].tolist() for k in range(len(z["names"]))]
        out[f] = (z["names"], z["qi"].astype(int), removed)
    for f in FRACTIONS[1:]:
        assert np.array_equal(out[f][1], out[0.0][1]), "W query files disagree on the queries"
    return out


def check_removed(metas, wq, rows) -> int:
    """Rooms removed here = rooms removed by W's test evaluation, query by query, every f."""
    n = 0
    for f in FRACTIONS:
        _, qis, removed = wq[f]
        pos = {int(q): k for k, q in enumerate(qis)}
        for qi in rows:
            _, ours = lg.damaged_lgkn_graph(metas[qi], f, random.Random(PARTIAL_SEED + qi))
            if sorted(ours) != sorted(removed[pos[qi]]):
                raise AssertionError(f"f={f} qi={qi}: removed {ours} here, {removed[pos[qi]]} in W's test eval")
            n += 1
    return n


def write_view(recorder, pq_dir: Path, frac, gallery: dict) -> Path:
    label = None if frac is None else f"random f={frac}"
    slug = "full" if frac is None else f"partial-random-f{frac}"
    path = pq_dir / f"graph_{TAG}_{slug}_{SPLIT}.npz"
    recorder.write(path, meta={"branch": "graph", "run_tag": TAG, "mode": "full" if frac is None else "partial",
                               "partial_label": label, "split": SPLIT, "exclude_self": True,
                               "query_seed": 42, "gallery": gallery, "partial_seed": PARTIAL_SEED,
                               "competitor": {"name": "LayoutGKN (GKN-asym)", "upstream_commit": UPSTREAM_COMMIT,
                                              "checkpoint": str(CKPT), "checkpoint_sha1": CKPT_SHA1,
                                              "epoch": BEST_EPOCH}})
    return path


def full_ndcg(prefix) -> dict:
    """Whole plan: mean nDCG@10 per axis and the paired query names."""
    d = load_perquery(f"{prefix}_full_{SPLIT}.npz")
    ki = d.k_index(FULL_K)
    return {ax: float(np.nanmean(d.ndcg[d.axis_index(ax), ki])) for ax in AXES} | {"n": int(len(d.names))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="LayoutGKN on the test (status.md §62 D)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", default=".", help="base folder of the outputs (smoke: a scratch folder)")
    args = ap.parse_args(argv)
    t_start = time.time()
    if not GATE.exists():
        sys.exit(f"!! {GATE} mancante: test non pre-registrato")
    out = paths(Path(args.root))
    for d in out.values():
        if d.exists():
            sys.exit(f"!! {d} exists already: nothing is overwritten")
    sha = file_sha1(CKPT)
    if sha != CKPT_SHA1:
        sys.exit(f"!! {CKPT}: sha1 {sha[:12]}, pinned {CKPT_SHA1[:12]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
        + (" · SMOKE" if args.smoke else ""))

    names = json.loads(GALLERY_JSON.read_text())["names"]
    assert gallery_sha1(names) == GALLERY_SHA1 and len(names) == 67405
    metas = [load_metadata(n) for n in names]
    assert all(m is not None for m in metas)
    gallery = {"n": len(names), "sha1": GALLERY_SHA1, "source": str(GALLERY_JSON)}
    wq = load_w_test_queries()
    rows = [int(r) for r in wq[0.0][1]]
    assert all(metas[r].split == SPLIT for r in rows), "W's test queries are not all test plans"
    if args.smoke:
        rows = rows[:20]
    n_checked = check_removed(metas, wq, rows)
    log(f"gallery {len(names)} · test queries {len(rows)} · removed rooms = W's test eval ({n_checked} views)")

    cfg = lt.upstream_cfg()
    ck = torch.load(CKPT, map_location=device)
    assert int(ck["epoch"]) == BEST_EPOCH, f"checkpoint epoch {ck['epoch']}, expected {BEST_EPOCH}"
    model = lt.build_model(cfg, 42).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    gal_graphs = [lg.to_lgkn_graph(m) for m in metas]
    axes = GalleryAxes(metas)
    out["pq"].mkdir(parents=True)
    E, M, own = ls.embed_rooms(model, gal_graphs, device)
    sc = ls.GalleryScorer(E, M, own, len(gal_graphs), cfg.mu, device)
    views = {}
    for frac in (*FRACTIONS, None):
        t = time.time()
        rec = PerQueryRecorder(K_VALUES, MAX_K, with_self_rr=frac is not None)
        info = evaluate_view(model, sc, metas, names, axes, rows, frac, device, rec)
        path = write_view(rec, out["pq"], frac, gallery)
        info["file"], info["sec"] = str(path), time.time() - t
        views["full" if frac is None else f"f{frac}"] = info
        log(f"{'full' if frac is None else f'f={frac}'}: n {info['n']} · {info['sec']:.0f}s -> {path}")

    a = load_auc(out["pq"] / f"graph_{TAG}", SPLIT)
    w = load_auc(W_PQ_PREFIX, SPLIT)
    vs_w = compare_auc(a, w)
    verdict = ("LayoutGKN più robusto" if vs_w["ci_lo"] > 0 else
               "W più robusto" if vs_w["ci_hi"] < 0 else "pareggio")
    self_min = views["f0.0"]["self_sim_min"]
    report = {
        "schema": "lgkn_test/1", "written": datetime.now().isoformat(timespec="seconds"), "smoke": args.smoke,
        "argv": sys.argv, "upstream_commit": UPSTREAM_COMMIT, "rule": RULE, "split": SPLIT,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "checkpoint": {"path": str(CKPT), "sha1": CKPT_SHA1, "epoch": BEST_EPOCH},
        "gallery": gallery, "views": views,
        "checks": {"self_sim_f0_min": self_min,
                   "self_sim_ok": bool(self_min is not None and self_min >= 1 - SELF_SIM_TOL),
                   "removed_rooms_equal_W_views": n_checked,
                   "paired_with_W": vs_w["n_pairs"], "expected_pairs": len(rows)},
        "auc": {"lgkn_asym": a.mean, "w_gat_comb_s42": w.mean},
        "vs_W": vs_w, "verdict": verdict,
        "full_plan_ndcg10_descriptive": {"lgkn_asym": full_ndcg(out["pq"] / f"graph_{TAG}"),
                                         "w_gat_comb_s42": full_ndcg(W_PQ_PREFIX)},
        "elapsed_hours": (time.time() - t_start) / 3600,
    }
    out["out"].mkdir(parents=True)
    (out["out"] / "test.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"written {out['out'] / 'test.json'} · checks: self-sim ok {report['checks']['self_sim_ok']} · "
        f"appaiate {vs_w['n_pairs']}/{len(rows)} · {report['elapsed_hours']:.2f} h")


if __name__ == "__main__":
    main()
