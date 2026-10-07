"""Full LayoutGKN comparison: training and evaluation in one GPU job (<= 24 h).

- training: GKN-asym (train.py recipe) on the whole `train` split, seed 42; 300 epochs or 20 h;
  checkpoint = best epoch on the robustness probe (2000 valid queries disjoint from eval, f in {.25, .5, .75});
- evaluation: full gallery (67,405), the 2000 valid queries of W's evaluation, removed rooms
  `Random(42 + qi)`, f in {0, .25, .5, .75} + whole plan; per-query files in `perquery/1` format;
- same network with initial random weights, damaged views only;
- paired AUC(LayoutGKN) - AUC(W seed 42) via `robustness_auc.compare_auc` (bootstrap B=10000, seed 0).

Ranking: self-recovery RR with ties broken at random in expectation (`score.expected_rr`);
per-axis rows = plans sorted by similarity (stable), self removed.

Usage:
    python -m src.competitors.layoutgkn.full                 # the real job
    python -m src.competitors.layoutgkn.full --smoke --root <scratch>
"""

from __future__ import annotations

import argparse
import copy
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
from src.competitors.layoutgkn.pilot import GALLERY_JSON, GALLERY_SHA1, PARTIAL_SEED, load_w_queries
from src.data.rplan_metadata import load_metadata
from src.evaluation.perquery import PerQueryRecorder, gallery_sha1
from src.evaluation.relevance import DISCRETE_AXES, GalleryAxes
from src.evaluation.robustness_auc import AUC_FRACTIONS, compare_auc, load_auc
from src.graph.evaluation.axis_metrics import accumulate_axes, new_metrics
from src.graph.evaluation.robustness_probe import probe_query_rows

FRACTIONS = (0.0, 0.25, 0.5, 0.75)
K_VALUES = (1, 5, 10, 100)
MAX_K = max(K_VALUES)
W_PQ_PREFIX = "results/final_pipeline/perquery/valid/s42/graph_gat_rg_comb_s42"
SELF_SIM_TOL = 1e-5                        # self-similarity check at f=0

REAL = dict(max_epochs=300, train_hours=20.0, n_probe=2000, n_eval=2000, n_train=None)
SMOKE = dict(max_epochs=1, train_hours=0.2, n_probe=20, n_eval=20, n_train=300)


def log(msg: str) -> None:
    print(f"[lgkn-full {datetime.now():%H:%M:%S}] {msg}", flush=True)


def paths(root: Path) -> dict:
    return {"ckpt": root / "embeddings/competitors/layoutgkn/asym",
            "pq": root / "results/perquery/competitor_layoutgkn/asym",
            "out": root / "results/competitors/layoutgkn/full"}


# --- scoring ---

def ranked_rows(scorer, model, graphs, device, block: int = 64):
    """Yields (index in `graphs`, similarity to every gallery plan [n])."""
    for lo in range(0, len(graphs), block):
        qE, qM, qown = ls.embed_rooms(model, graphs[lo:lo + block], device)
        sim = scorer.similarity(qE, qM, qown, int(qown.max()) + 1).cpu().numpy()
        for k in range(sim.shape[0]):
            yield lo + k, sim[k]


def top_rows(sim: np.ndarray, n: int) -> list[int]:
    """Indices of the n highest scores, by decreasing score (stable on ties)."""
    part = np.argpartition(-sim, n)[:n + 1] if n < len(sim) else np.arange(len(sim))
    order = part[np.lexsort((part, -sim[part]))]
    return [int(r) for r in order[:n]]


def evaluate_view(model, scorer, metas, names, axes, rows, frac, device, recorder):
    """Damaged view f (or whole plan if frac is None) of every query row; fills `recorder`."""
    metrics = new_metrics(K_VALUES)
    skipped = {ax: 0 for ax in DISCRETE_AXES}
    graphs, kept, n_empty = [], [], 0
    for qi in rows:
        if frac is None:
            g = lg.to_lgkn_graph(metas[qi])
        else:
            g, _ = lg.damaged_lgkn_graph(metas[qi], frac, random.Random(PARTIAL_SEED + qi))
        if g is None:
            n_empty += 1
            continue
        graphs.append(g)
        kept.append(qi)
    self_sims, rrs = [], []
    for k, sim in ranked_rows(scorer, model, graphs, device):
        qi = kept[k]
        top = top_rows(sim, MAX_K + 1)
        axis_rows = [r for r in top if r != qi][:MAX_K]
        before = dict(skipped)
        accumulate_axes(metrics, skipped, axes, qi, axis_rows, K_VALUES, exclude_self=True)
        rr = ls.expected_rr(sim, qi, MAX_K) if frac is not None else None
        rrs.append(rr)
        self_sims.append(float(sim[qi]))
        recorder.add(name=names[qi], qi=qi, ret_rows=axis_rows, metrics=metrics,
                     skipped_before=before, skipped=skipped, axes=axes, exclude_self=True,
                     **({"self_rr": rr} if frac is not None else {}))
    return {"n": len(kept), "n_emptied": n_empty, "self_sim_min": float(min(self_sims)) if self_sims else None,
            "mrr": float(np.mean(rrs)) if frac is not None and rrs else None}


def write_view(recorder, pq_dir: Path, tag: str, frac, gallery: dict) -> Path:
    label = None if frac is None else f"random f={frac}"
    slug = "full" if frac is None else f"partial-random-f{frac}"
    path = pq_dir / f"graph_{tag}_{slug}_valid.npz"
    recorder.write(path, meta={"branch": "graph", "run_tag": tag, "mode": "full" if frac is None else "partial",
                               "partial_label": label, "split": "valid", "exclude_self": True,
                               "query_seed": 42, "gallery": gallery, "partial_seed": PARTIAL_SEED,
                               "competitor": {"name": "LayoutGKN (GKN-asym)", "upstream_commit": UPSTREAM_COMMIT}})
    return path


# --- main ---

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="LayoutGKN full comparison (status.md §60.2)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", default=".", help="base folder of the outputs (smoke: a scratch folder)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    P = SMOKE if args.smoke else REAL
    t_start = time.time()
    out = paths(Path(args.root))
    for d in out.values():
        if d.exists():
            sys.exit(f"!! {d} exists already: nothing is overwritten")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
        + (" · SMOKE" if args.smoke else ""))

    # --- data ---
    names = json.loads(GALLERY_JSON.read_text())["names"]
    assert gallery_sha1(names) == GALLERY_SHA1 and len(names) == 67405
    metas = [load_metadata(n) for n in names]
    assert all(m is not None for m in metas)
    gallery = {"n": len(names), "sha1": GALLERY_SHA1, "source": str(GALLERY_JSON)}
    wq = load_w_queries()
    all_eval = [int(r) for r in wq[0.0][1]]
    eval_rows = all_eval[:P["n_eval"]]
    valid_rows = [r for r, m in enumerate(metas) if m.split == "valid"]
    probe_rows = probe_query_rows(valid_rows, all_eval, P["n_probe"], args.seed)
    train_rows = [r for r, m in enumerate(metas) if m.split == "train"]
    if P["n_train"]:
        train_rows = random.Random(args.seed).sample(train_rows, P["n_train"])
    leak = sum(metas[r].split != "train" for r in train_rows)
    assert leak == 0 and not (set(train_rows) & set(all_eval)) and not (set(train_rows) & set(probe_rows))
    log(f"gallery {len(names)} · train {len(train_rows)} (valid/test in training: {leak}) · "
        f"eval {len(eval_rows)} · probe {len(probe_rows)}")

    gal_graphs = [lg.to_lgkn_graph(m) for m in metas]
    probe_graphs, probe_self = {f: [] for f in AUC_FRACTIONS}, []
    for r in probe_rows:
        built = [lg.damaged_lgkn_graph(metas[r], f, random.Random(PARTIAL_SEED + r))[0] for f in AUC_FRACTIONS]
        if any(g is None for g in built):
            continue
        for f, g in zip(AUC_FRACTIONS, built):
            probe_graphs[f].append(g)
        probe_self.append(r)
    log(f"probe queries kept in every f: {len(probe_self)}/{len(probe_rows)}")

    cfg = lt.upstream_cfg()
    model = lt.build_model(cfg, args.seed).to(device)
    random_state = copy.deepcopy(model.state_dict())

    def probe(m):
        E, M, own = ls.embed_rooms(m, gal_graphs, device)
        sc = ls.GalleryScorer(E, M, own, len(gal_graphs), cfg.mu, device)
        res = {}
        for f in AUC_FRACTIONS:
            rr = [ls.expected_rr(sim, probe_self[k], MAX_K)
                  for k, sim in ranked_rows(sc, m, probe_graphs[f], device)]
            res[f"mrr_f{f}"] = float(np.mean(rr))
        res["auc"] = float(np.mean([res[f"mrr_f{f}"] for f in AUC_FRACTIONS]))
        return res

    # --- training ---
    out["ckpt"].mkdir(parents=True)
    deadline = t_start + P["train_hours"] * 3600
    tlog = lt.fit(model, cfg, [metas[r] for r in train_rows], probe, device, P["max_epochs"], deadline,
                  out["ckpt"] / "best.pt", args.seed, log_path=out["ckpt"] / "train_log.json")
    log(f"training stopped: {tlog.stop_reason} · best epoch {tlog.best_epoch} (probe AUC {tlog.best_auc:.4f})")
    model.load_state_dict(torch.load(out["ckpt"] / "best.pt", map_location=device)["state_dict"])

    # --- evaluation: trained and random weights ---
    axes = GalleryAxes(metas)
    out["pq"].mkdir(parents=True)
    rnd = lt.build_model(cfg, args.seed).to(device)
    rnd.load_state_dict(random_state)
    views = {}
    for tag, net, fracs in (("lgkn_asym_s42", model, (*FRACTIONS, None)),
                            ("lgkn_random_s42", rnd, FRACTIONS)):
        E, M, own = ls.embed_rooms(net, gal_graphs, device)
        sc = ls.GalleryScorer(E, M, own, len(gal_graphs), cfg.mu, device)
        for frac in fracs:
            t = time.time()
            rec = PerQueryRecorder(K_VALUES, MAX_K, with_self_rr=frac is not None)
            info = evaluate_view(net, sc, metas, names, axes, eval_rows, frac, device, rec)
            path = write_view(rec, out["pq"], tag, frac, gallery)
            info["file"], info["sec"] = str(path), time.time() - t
            views[f"{tag}/{'full' if frac is None else f'f{frac}'}"] = info
            log(f"{tag} {'full' if frac is None else f'f={frac}'}: n {info['n']} · MRR {info['mrr']} · "
                f"{info['sec']:.0f}s -> {path}")

    # --- comparison with W seed 42, paired by query name ---
    a = load_auc(out["pq"] / "graph_lgkn_asym_s42", "valid")
    w = load_auc(W_PQ_PREFIX, "valid")
    r = load_auc(out["pq"] / "graph_lgkn_random_s42", "valid")
    vs_w = compare_auc(a, w)
    vs_rnd = compare_auc(a, r)
    verdict = ("LayoutGKN più robusto" if vs_w["ci_lo"] > 0 else
               "il nostro ramo graph (W) più robusto" if vs_w["ci_hi"] < 0 else "pareggio")
    self_min = views["lgkn_asym_s42/f0.0"]["self_sim_min"]
    report = {
        "schema": "lgkn_full/1", "written": datetime.now().isoformat(timespec="seconds"), "smoke": args.smoke,
        "argv": sys.argv, "upstream_commit": UPSTREAM_COMMIT, "rule": "status.md §60.2",
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "gallery": gallery, "train_plans": len(train_rows), "valid_or_test_in_training": leak,
        "training": tlog.to_json(), "views": views,
        "checks": {"self_sim_f0_min": self_min, "self_sim_ok": bool(self_min is not None and self_min >= 1 - SELF_SIM_TOL),
                   "paired_with_W": vs_w["n_pairs"], "expected_pairs": len(eval_rows)},
        "auc": {"lgkn_asym": a.mean, "w_gat_comb_s42": w.mean, "lgkn_random": r.mean},
        "vs_W": vs_w, "vs_random_weights": vs_rnd, "verdict": verdict,
        "elapsed_hours": (time.time() - t_start) / 3600,
    }
    out["out"].mkdir(parents=True)
    (out["out"] / "full.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"AUC LayoutGKN {a.mean:.4f} · W {w.mean:.4f} · Δ {vs_w['delta']:+.4f} "
        f"[{vs_w['ci_lo']:+.4f}, {vs_w['ci_hi']:+.4f}] → {verdict}")
    log(f"written {out['out'] / 'full.json'} · {report['elapsed_hours']:.2f} h")


if __name__ == "__main__":
    main()
