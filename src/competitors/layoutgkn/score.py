"""Vectorised LayoutGKN similarity of queries against a whole gallery.

Plan similarity is the normalised GraphHopper kernel on room embeddings (upstream `loss.ghopper_sim`):

    k(i, j)   = sum_{u in i, w in j} (m_u . m_w) * exp(-mu * ||e_u - e_w||^2)
    sim(i, j) = clamp(k(i, j) / sqrt(k(i, i) k(j, j) + 1e-8), 1e-6, 1)

with e = room embeddings of the network, m = flattened shortest-path matrices
(`graphs.shortest_path_matrix`) and mu = 1 / hid_dim (upstream `finalize_cfg`).

The gallery is stored as a flat list of rooms (embeddings, shp rows, owner plan) with precomputed
self-kernels; a query block is scored with one room-by-room product and two scatter-sums.

`expected_rr`: self-recovery reciprocal rank, 0 beyond `max_k`; exact ties broken at random in
expectation (mean RR over the self's possible positions among tied plans).
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.loader import DataLoader

EPS_NORM = 1e-8          # as upstream ghopper_sim
CLAMP = (1e-6, 1.0)


def kernel_mu(hid_dim: int) -> float:
    """Upstream `finalize_cfg`: sigma^2 = hid_dim / 2, mu = 1 / (2 sigma^2)."""
    sigma2 = hid_dim / 2
    return 1.0 / (2.0 * sigma2)


@torch.no_grad()
def embed_rooms(model, graphs, device, batch_size: int = 512):
    """(E [N_rooms, D], M [N_rooms, 16], owner [N_rooms]) in graph order; `owner[r]` = graph index (eval mode)."""
    from src.competitors.layoutgkn import upstream
    upstream()
    from LayoutGKN.data import prep_data

    model.eval()
    E, M, owner = [], [], []
    start = 0
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        ei, xg, xc, shp, xe, b = prep_data(batch, device)
        feats, _ = model(ei, xg, xc, xe, b)
        E.append(feats.float().cpu())
        M.append(shp.float().cpu())
        owner.append(b.cpu() + start)
        start += batch.num_graphs
    return torch.cat(E), torch.cat(M), torch.cat(owner)


def _cross(qE, qM, qown, nq, gE, gM, gown, ng, mu):
    """Kernel k(query, plan) for every pair: [nq, ng]."""
    w = qM @ gM.T                                               # [rq, rg]
    lin = qE @ gE.T
    d2 = (-2 * lin.T + (qE ** 2).sum(1)).T + (gE ** 2).sum(1)
    x = w * torch.exp(-mu * d2)
    per_q = torch.zeros(nq, x.shape[1], dtype=x.dtype, device=x.device).index_add_(0, qown, x)
    return torch.zeros(nq, ng, dtype=x.dtype, device=x.device).index_add_(1, gown, per_q)


def self_kernels(E, M, owner, n: int, mu: float, block: int = 256) -> torch.Tensor:
    """k(i, i) for each of the `n` graphs, blockwise."""
    out = torch.empty(n, dtype=E.dtype, device=E.device)
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        sel = (owner >= lo) & (owner < hi)
        e, m, o = E[sel], M[sel], owner[sel] - lo
        w = m @ m.T
        d2 = (-2 * (e @ e.T).T + (e ** 2).sum(1)).T + (e ** 2).sum(1)
        x = w * torch.exp(-mu * d2) * (o[:, None] == o[None, :])
        out[lo:hi] = torch.zeros(hi - lo, dtype=x.dtype, device=x.device).index_add_(0, o, x.sum(1))
    return out


class GalleryScorer:
    """Gallery of LayoutGKN plans, scored against blocks of queries."""

    def __init__(self, E, M, owner, n: int, mu: float, device="cpu"):
        self.E, self.M = E.to(device), M.to(device)
        self.owner, self.n, self.mu, self.device = owner.to(device), n, mu, device
        self.self_k = self_kernels(self.E, self.M, self.owner, n, mu)

    @torch.no_grad()
    def similarity(self, qE, qM, qown, nq: int, room_block: int = 200_000) -> torch.Tensor:
        """sim(query, plan) [nq, n] for `nq` queries given as flat rooms (owner in 0..nq-1)."""
        qE, qM, qown = qE.to(self.device), qM.to(self.device), qown.to(self.device)
        q_self = self_kernels(qE, qM, qown, nq, self.mu)
        cross = torch.zeros(nq, self.n, dtype=qE.dtype, device=self.device)
        for lo in range(0, self.E.shape[0], room_block):
            hi = min(lo + room_block, self.E.shape[0])
            cross += _cross(qE, qM, qown, nq, self.E[lo:hi], self.M[lo:hi],
                            self.owner[lo:hi], self.n, self.mu)
        sim = cross / torch.sqrt(q_self[:, None] * self.self_k[None, :] + EPS_NORM)
        return torch.clamp(sim, *CLAMP)


def expected_rr(scores: np.ndarray, self_idx: int, max_k: int = 100) -> float:
    """Self-recovery RR (0 beyond max_k), exact ties broken at random in expectation."""
    s = scores[self_idx]
    greater = int((scores > s).sum())
    tied = int((scores == s).sum()) - 1
    ranks = 1 + greater + np.arange(tied + 1)
    return float(np.where(ranks <= max_k, 1.0 / ranks, 0.0).mean())
