"""GKN-asym: the LayoutGKN network re-trained with our `asym_partial` training signal.

Unchanged upstream (pinned commit): network `GraphSiameseNetwork`, kernel loss `ghopper_loss`
(triplet margin on -log of the normalised GraphHopper similarity), `conf/default.yaml` hyper-parameters
with `kernel_loss=true` (mu = 1/64), AdamW lr 1e-4, batch 64, graph-vector regularisation 0.5 * 5e-4 * mean(g^2).

Changed, the training signal:
- query    = plan with round(f * n) rooms removed, f ~ U[0.25, 0.75], at least one room kept
             (as `augment.remove_rooms_view`), shortest paths recomputed;
- positive = same plan, whole, with flip / 90-degree rotation (p = 0.5 each);
- negative = another plan of the batch, whole, same symmetries;
- triplet fed as upstream: graphs [q, pos, q, neg], pairs (0,1), (2,3); no Drive triplets, no mining;
- checkpoint = best epoch on the robustness probe (self-recovery AUC on valid queries disjoint from eval).
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch_geometric.data import Batch, Data

from src.competitors.layoutgkn import UPSTREAM_CONF, upstream
from src.competitors.layoutgkn import graphs as lg
from src.data.rplan_metadata import RoomMeta
from src.graph.graph_partial_query import filter_meta

upstream()
from LayoutGKN.data import prep_data  # noqa: E402
from LayoutGKN.loss import ghopper_loss  # noqa: E402
from LayoutGKN.model import GraphSiameseNetwork  # noqa: E402

FRAC_MIN, FRAC_MAX = 0.25, 0.75     # training damage
FLIP_PROB = ROT_PROB = 0.5          # symmetries of the whole view


def upstream_cfg():
    """conf/default.yaml with kernel_loss=true and mu = 1/(2 * hid_dim/2) (upstream finalize_cfg)."""
    cfg = OmegaConf.load(UPSTREAM_CONF)
    cfg.kernel_loss = True
    cfg.mu = 1.0 / (2.0 * (cfg.hid_dim / 2))
    return cfg


def build_model(cfg, seed: int) -> GraphSiameseNetwork:
    torch.manual_seed(seed)
    return GraphSiameseNetwork(cfg)


# --- views ---

def symmetric_view(g: Data, rng: random.Random) -> Data:
    """Flip (cx -> -cx) and/or k*90-degree rotation, (cx, cy) -> (-cy, cx) with w <-> h (coordinates centred)."""
    flip = rng.random() < FLIP_PROB
    k = rng.randint(1, 3) if rng.random() < ROT_PROB else 0
    if not flip and k == 0:
        return g
    geo = g.geometry.clone()
    if flip:
        geo[:, 0] = -geo[:, 0]
    for _ in range(k):
        cx, cy, w, h = geo[:, 0].clone(), geo[:, 1].clone(), geo[:, 2].clone(), geo[:, 3].clone()
        geo[:, 0], geo[:, 1], geo[:, 2], geo[:, 3] = -cy, cx, h, w
    out = g.clone()
    out.geometry = geo
    return out


def damaged_view(meta: RoomMeta, rng: random.Random) -> Data:
    """round(f * n) rooms removed at random, f ~ U[0.25, 0.75], at least one room left."""
    n = meta.num_rooms
    f = FRAC_MIN + (FRAC_MAX - FRAC_MIN) * rng.random()
    k = min(round(f * n), n - 1)
    removed = sorted(rng.sample(range(n), k)) if k > 0 else []
    return lg.to_lgkn_graph(filter_meta(meta, removed))


def triplet_batch(metas, whole, idx, rng: random.Random) -> Batch:
    """[q_i, pos_i, q_i, neg_i] for every plan i of the batch (upstream order)."""
    out = []
    b = len(idx)
    for p, i in enumerate(idx):
        q = damaged_view(metas[i], rng)
        j = idx[(p + rng.randint(1, b - 1)) % b]
        out += [q, symmetric_view(whole[i], rng), q, symmetric_view(whole[j], rng)]
    return Batch.from_data_list(out)


# --- loop ---

@dataclass
class TrainLog:
    epochs: list = field(default_factory=list)
    best_epoch: int = 0
    best_auc: float = float("-inf")
    stop_reason: str = ""

    def to_json(self) -> dict:
        return {"epochs": self.epochs, "best_epoch": self.best_epoch,
                "best_auc": self.best_auc, "stop_reason": self.stop_reason}


def train_epoch(model, opt, cfg, metas, whole, rng: random.Random, device) -> dict:
    model.train()
    order = list(range(len(metas)))
    rng.shuffle(order)
    losses, accs, t_data, t_step = [], [], 0.0, 0.0
    for lo in range(0, len(order) - 1, cfg.bs):
        idx = order[lo:lo + cfg.bs]
        if len(idx) < 2:
            break
        t0 = time.time()
        batch = triplet_batch(metas, whole, idx, rng)
        t1 = time.time()
        ei, xg, xc, shp, xe, bvec = prep_data(batch, device)
        feats, gfeats = model(ei, xg, xc, xe, bvec)
        loss = 0.5 * cfg.graph_vec_weight * torch.mean(gfeats ** 2) if cfg.graph_reg else 0.0
        d_rel, trip = ghopper_loss(feats, shp, bvec, margin=cfg.margin, mu=cfg.mu)
        loss = loss + trip.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_data += t1 - t0
        t_step += time.time() - t1
        losses.append(float(loss))
        accs.append(float((d_rel < 0).float().mean()))
    return {"loss": float(np.mean(losses)), "triplet_acc": float(np.mean(accs)),
            "sec_data": t_data, "sec_step": t_step}


def fit(model, cfg, metas, probe, device, max_epochs: int, deadline: float,
        ckpt_path, seed: int, log_path=None) -> TrainLog:
    """Train until `max_epochs` or `deadline` (time.time() value); best-probe-AUC state goes to `ckpt_path`."""
    model.to(device)
    opt = AdamW(model.parameters(), lr=cfg.lr)
    whole = [lg.to_lgkn_graph(m) for m in metas]
    rng = random.Random(seed)
    log = TrainLog()
    last = 0.0
    for epoch in range(1, max_epochs + 1):
        if time.time() + 1.2 * last > deadline:
            log.stop_reason = f"time budget (before epoch {epoch})"
            break
        t0 = time.time()
        row = train_epoch(model, opt, cfg, metas, whole, rng, device)
        t1 = time.time()
        row.update(probe(model))
        row["epoch"], row["sec_train"], row["sec_probe"] = epoch, t1 - t0, time.time() - t1
        last = time.time() - t0
        log.epochs.append(row)
        if row["auc"] > log.best_auc:
            log.best_auc, log.best_epoch = row["auc"], epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "cfg": OmegaConf.to_container(cfg)}, ckpt_path)
        print(f"[lgkn] epoch {epoch}: loss {row['loss']:.4f} acc {row['triplet_acc']:.3f} "
              f"probe AUC {row['auc']:.4f} ({row['sec_train']:.0f}s + {row['sec_probe']:.0f}s)", flush=True)
        if log_path is not None:
            log_path.write_text(json.dumps(log.to_json(), indent=1))
    else:
        log.stop_reason = f"max epochs ({max_epochs})"
    return log
