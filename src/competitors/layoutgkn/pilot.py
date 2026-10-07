"""
LayoutGKN feasibility pilot on a sub-problem (one GPU job, <= 8 h, seed 42).

- training: 10,000 plans of the `train` split;
- gallery: 5,000 plans = originals of evaluation and probe queries + random plans;
- evaluation queries: first 500 of the 2,000 valid queries (W's sampling order, `qi` from W's
  query-vector files), same removed rooms (`random.Random(42 + qi)`), f in {0, .25, .5, .75};
- probe queries: 500 valid plans disjoint from the 2,000 evaluation ones (`robustness_probe.probe_query_rows`).

Outputs (`pilot.json` + per-query `.npz`):
- self-recovery RR per query and f, AUC = mean over f in {.25, .5, .75}, for GKN-asym trained,
  the same network at initial random weights, `hist` (type histogram), W seed 42 (gat/comb) on the sub-gallery;
- AUC(trained) - AUC(random weights), paired bootstrap CI (B 10000);
- costs: time per epoch, per-query scoring time on the full gallery (67,405, measured), GPU memory,
  estimates of full evaluation (<= 16 h) and full training (>= 100 epochs, <= 48 GPU h).

Usage:
    python -m src.competitors.layoutgkn.pilot                  # the real pilot (GPU job)
    python -m src.competitors.layoutgkn.pilot --smoke --out-dir <scratch> --ckpt-dir <scratch>
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
from src.data.rplan_metadata import NUM_ROOM_TYPES, load_metadata
from src.evaluation.perquery import gallery_sha1
from src.evaluation.significance import BOOTSTRAP_B, BOOTSTRAP_SEED, bootstrap_ci
from src.graph.evaluation.robustness_probe import probe_query_rows
from src.graph.graph_partial_query import filter_meta

GALLERY_JSON = Path("results/shared_gallery.json")
GALLERY_SHA1 = "0c24cfc05e188bb9072205c83437feb215470019"
W_EMB = Path("embeddings/graph/gat/rg_comb_s42/embeddings.npy")
W_QVEC = "results/final_pipeline/queryvec/valid/s42/graph_gat_rg_comb_s42_partial-random-f{f}_valid.npz"
FRACTIONS = (0.0, 0.25, 0.5, 0.75)
AUC_FRACTIONS = (0.25, 0.5, 0.75)
PARTIAL_SEED = 42
MAX_K = 100                                   # RR = 0 beyond

REAL = dict(n_train=10_000, n_gallery=5_000, n_eval=500, n_probe=500, max_epochs=100,
            job_hours=8.0, train_hours=6.5, f4_queries=200)
SMOKE = dict(n_train=200, n_gallery=400, n_eval=30, n_probe=30, max_epochs=2,
             job_hours=1.0, train_hours=0.3, f4_queries=5)
FULL_EVAL_QUERIES = 2000 * len(FRACTIONS) + 2000      # 4 damage levels + whole-plan view
FULL_EPOCHS = 100
FULL_PROBE_QUERIES = 2000 * len(AUC_FRACTIONS)


def log(msg: str) -> None:
    print(f"[lgkn-pilot {datetime.now():%H:%M:%S}] {msg}", flush=True)


# --- data ---

def load_w_queries():
    """names, qi (sampling order) and vectors of W's evaluation, per f; same queries in every f."""
    out = {}
    for f in FRACTIONS:
        z = np.load(W_QVEC.format(f=f), allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        assert meta["gallery"]["sha1"] == GALLERY_SHA1 and meta["partial_seed"] == PARTIAL_SEED
        out[f] = (z["names"], z["qi"].astype(int), z["vectors"])
    for f in FRACTIONS[1:]:
        assert np.array_equal(out[f][1], out[0.0][1]), "W query files disagree on the queries"
    return out


def damaged_queries(metas, rows):
    """{f: graphs} and the histogram of the damaged plan, for queries kept in every f."""
    graphs = {f: [] for f in FRACTIONS}
    hists = {f: [] for f in FRACTIONS}
    kept, dropped = [], 0
    for r in rows:
        built = {}
        for f in FRACTIONS:
            g, removed = lg.damaged_lgkn_graph(metas[r], f, random.Random(PARTIAL_SEED + r))
            if g is None:
                break
            built[f] = (g, filter_meta(metas[r], removed).type_histogram)
        if len(built) != len(FRACTIONS):
            dropped += 1
            continue
        for f in FRACTIONS:
            graphs[f].append(built[f][0])
            hists[f].append(built[f][1])
        kept.append(r)
    return graphs, hists, kept, dropped


# --- scoring ---

def lgkn_rr(model, gal_graphs, q_graphs, self_idx, mu, device, block: int = 64):
    """Expected self-recovery RR of each query against `gal_graphs`; also min self-sim."""
    E, M, own = ls.embed_rooms(model, gal_graphs, device)
    scorer = ls.GalleryScorer(E, M, own, len(gal_graphs), mu, device)
    rr, self_sim = [], []
    for lo in range(0, len(q_graphs), block):
        qE, qM, qown = ls.embed_rooms(model, q_graphs[lo:lo + block], device)
        sim = scorer.similarity(qE, qM, qown, int(qown.max()) + 1).cpu().numpy()
        for k in range(sim.shape[0]):
            s = self_idx[lo + k]
            rr.append(ls.expected_rr(sim[k], s, MAX_K))
            self_sim.append(float(sim[k, s]))
    return np.array(rr), np.array(self_sim)


def cosine_rr(gal, qry, self_idx):
    gal = gal / np.maximum(np.linalg.norm(gal, axis=1, keepdims=True), 1e-12)
    qry = qry / np.maximum(np.linalg.norm(qry, axis=1, keepdims=True), 1e-12)
    sims = qry @ gal.T
    return np.array([ls.expected_rr(sims[k], self_idx[k], MAX_K) for k in range(len(qry))])


def auc_of(rr_by_f):
    return np.mean(np.stack([rr_by_f[f] for f in AUC_FRACTIONS]), axis=0)


def summary(rr_by_f):
    return {"auc": float(auc_of(rr_by_f).mean()),
            **{f"mrr_f{f}": float(rr_by_f[f].mean()) for f in FRACTIONS}}


# --- main ---

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--out-dir", default="results/competitors/layoutgkn/pilot")
    p.add_argument("--ckpt-dir", default="embeddings/competitors/layoutgkn/pilot")
    p.add_argument("--smoke", action="store_true", help="tiny sizes, to check the pipeline")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    P = SMOKE if args.smoke else REAL
    t_start = time.time()
    out_dir, ckpt_dir = Path(args.out_dir), Path(args.ckpt_dir)
    for d in (out_dir, ckpt_dir):
        if d.exists():
            sys.exit(f"!! {d} exists already: nothing is overwritten")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
        + (" · SMOKE" if args.smoke else ""))

    # --- gallery, queries, sub-gallery, training plans ---
    names = json.loads(GALLERY_JSON.read_text())["names"]
    assert gallery_sha1(names) == GALLERY_SHA1 and len(names) == 67405
    metas = [load_metadata(n) for n in names]
    assert all(m is not None for m in metas)
    wq = load_w_queries()
    all_eval = [int(r) for r in wq[0.0][1]]
    eval_rows = all_eval[:P["n_eval"]]
    valid_rows = [r for r, m in enumerate(metas) if m.split == "valid"]
    probe_rows = probe_query_rows(valid_rows, all_eval, P["n_probe"], args.seed)
    fixed = set(eval_rows) | set(probe_rows)
    others = sorted(set(range(len(names))) - fixed)
    sub = sorted(fixed | set(random.Random(args.seed).sample(others, P["n_gallery"] - len(fixed))))
    pos = {r: i for i, r in enumerate(sub)}
    train_pool = [r for r, m in enumerate(metas) if m.split == "train"]
    train_rows = random.Random(args.seed).sample(train_pool, P["n_train"])
    log(f"gallery {len(names)} · sub-gallery {len(sub)} · eval {len(eval_rows)} · probe "
        f"{len(probe_rows)} · train {len(train_rows)} of {len(train_pool)}")

    gal_graphs = [lg.to_lgkn_graph(metas[r]) for r in sub]
    q_graphs, q_hist, q_kept, q_drop = damaged_queries(metas, eval_rows)
    p_graphs, _, p_kept, p_drop = damaged_queries(metas, probe_rows)
    q_self = [pos[r] for r in q_kept]
    p_self = [pos[r] for r in p_kept]
    log(f"queries dropped (emptied in some f): eval {q_drop} · probe {p_drop}")

    # --- model (initial random weights kept) ---
    cfg = lt.upstream_cfg()
    model = lt.build_model(cfg, args.seed).to(device)
    random_state = copy.deepcopy(model.state_dict())

    def probe(m):
        rr = {f: lgkn_rr(m, gal_graphs, p_graphs[f], p_self, cfg.mu, device)[0] for f in AUC_FRACTIONS}
        return {"auc": float(auc_of(rr).mean()), **{f"mrr_f{f}": float(rr[f].mean()) for f in AUC_FRACTIONS}}

    ckpt_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    deadline = t_start + P["train_hours"] * 3600
    train_metas = [metas[r] for r in train_rows]
    tlog = lt.fit(model, cfg, train_metas, probe, device, P["max_epochs"], deadline,
                  ckpt_dir / "best.pt", args.seed, log_path=ckpt_dir / "train_log.json")
    log(f"training stopped: {tlog.stop_reason} · best epoch {tlog.best_epoch} (probe AUC {tlog.best_auc:.4f})")
    mem_train = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None

    # --- evaluation on the sub-gallery ---
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device)["state_dict"])
    rnd = lt.build_model(cfg, args.seed).to(device)
    rnd.load_state_dict(random_state)
    res, self_sims = {}, {}
    for tag, net in (("gkn_asym", model), ("gkn_random", rnd)):
        res[tag] = {}
        for f in FRACTIONS:
            res[tag][f], ss = lgkn_rr(net, gal_graphs, q_graphs[f], q_self, cfg.mu, device)
            if f == 0.0:
                self_sims[tag] = float(ss.min())
    gal_hist = np.array([metas[r].type_histogram for r in sub], dtype=np.float64)
    res["hist"] = {f: cosine_rr(gal_hist, np.array(q_hist[f], dtype=np.float64), q_self) for f in FRACTIONS}
    w_gal = np.load(W_EMB, mmap_mode="r")[sub].astype(np.float64)
    k_of = {int(r): k for k, r in enumerate(wq[0.0][1])}
    res["w_gat_comb_s42"] = {f: cosine_rr(w_gal, wq[f][2][[k_of[r] for r in q_kept]].astype(np.float64), q_self)
                             for f in FRACTIONS}
    diff = auc_of(res["gkn_asym"]) - auc_of(res["gkn_random"])
    lo, hi = bootstrap_ci(diff, BOOTSTRAP_B, BOOTSTRAP_SEED)
    log(f"AUC trained {auc_of(res['gkn_asym']).mean():.4f} · random {auc_of(res['gkn_random']).mean():.4f} "
        f"· delta {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # --- costs on the full gallery ---
    t0 = time.time()
    full_graphs = [lg.to_lgkn_graph(m) for m in metas]
    t_build = time.time() - t0
    t0 = time.time()
    E, M, own = ls.embed_rooms(model, full_graphs, device)
    t_embed = time.time() - t0
    t0 = time.time()
    scorer = ls.GalleryScorer(E, M, own, len(full_graphs), cfg.mu, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_self = time.time() - t0
    qs = q_graphs[0.5][:P["f4_queries"]]
    t0 = time.time()
    for lo_ in range(0, len(qs), 64):
        qE, qM, qown = ls.embed_rooms(model, qs[lo_:lo_ + 64], device)
        sim = scorer.similarity(qE, qM, qown, int(qown.max()) + 1).cpu().numpy()
        for k in range(sim.shape[0]):
            ls.expected_rr(sim[k], q_kept[lo_ + k], MAX_K)
    t_query = (time.time() - t0) / len(qs)
    mem_peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    gallery_prep = t_build + t_embed + t_self
    f4_hours = (gallery_prep + t_query * FULL_EVAL_QUERIES) / 3600
    ep = [e["sec_train"] for e in tlog.epochs]
    sec_per_plan = float(np.mean(ep)) / len(train_rows) if ep else float("nan")
    probe_full = t_embed + t_self + t_query * FULL_PROBE_QUERIES
    f5_hours = FULL_EPOCHS * (sec_per_plan * len(train_pool) + probe_full) / 3600
    log(f"full gallery: build {t_build:.0f}s embed {t_embed:.0f}s self-kernels {t_self:.0f}s · "
        f"{t_query * 1000:.1f} ms/query → full evaluation ≈ {f4_hours:.2f} h · full training ≈ {f5_hours:.1f} h")

    # --- write ---
    np.savez_compressed(
        out_dir / "perquery_pilot_valid.npz",
        names=np.array([names[r] for r in q_kept]), qi=np.array(q_kept),
        **{f"rr_{tag}_f{f}": res[tag][f] for tag in res for f in FRACTIONS})
    report = {
        "schema": "lgkn_pilot/1", "written": datetime.now().isoformat(timespec="seconds"),
        "smoke": args.smoke, "argv": sys.argv, "upstream_commit": UPSTREAM_COMMIT,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "sizes": {"gallery_full": len(names), "sub_gallery": len(sub), "eval_queries": len(q_kept),
                  "eval_dropped": q_drop, "probe_queries": len(p_kept), "probe_dropped": p_drop,
                  "train_plans": len(train_rows), "train_split_full": len(train_pool)},
        "gallery_sha1": GALLERY_SHA1, "sub_gallery_sha1": gallery_sha1([names[r] for r in sub]),
        "partial_seed": PARTIAL_SEED, "max_k": MAX_K, "hid_dim": int(cfg.hid_dim), "mu": float(cfg.mu),
        "training": tlog.to_json(),
        "results_sub_gallery": {tag: summary(res[tag]) for tag in res},
        "third_criterion": {"delta_auc": float(diff.mean()), "ci95": [lo, hi], "n": int(len(diff)),
                            "bootstrap": {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED}, "pass": bool(lo > 0)},
        "self_similarity_f0_min": self_sims,
        "costs": {"sec_per_epoch_train": ep, "sec_per_train_plan": sec_per_plan,
                  "full_gallery_sec": {"build": t_build, "embed": t_embed, "self_kernels": t_self},
                  "sec_per_query_full_gallery": t_query, "queries_timed": len(qs),
                  "gpu_mem_gb": {"training": mem_train, "peak": mem_peak}},
        "fourth_criterion": {"full_eval_queries": FULL_EVAL_QUERIES, "est_hours": f4_hours,
                             "limit_hours": 16, "pass": bool(f4_hours <= 16)},
        "fifth_criterion": {"epochs": FULL_EPOCHS, "probe_queries_per_epoch": FULL_PROBE_QUERIES,
                            "est_gpu_hours": f5_hours, "limit_hours": 48, "pass": bool(f5_hours <= 48)},
        "elapsed_hours": (time.time() - t_start) / 3600,
    }
    (out_dir / "pilot.json").write_text(json.dumps(report, indent=1))
    log(f"written {out_dir / 'pilot.json'} · {report['elapsed_hours']:.2f} h")


if __name__ == "__main__":
    main()
