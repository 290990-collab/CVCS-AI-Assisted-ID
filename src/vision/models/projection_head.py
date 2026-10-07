from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Light MLP trained on top of a frozen encoder embedding; L2-normalised output (FAISS IndexFlatIP)."""

    def __init__(self, in_dim: int, hidden_dim: int = 1024, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def info_nce(za: torch.Tensor, zp: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE/NT-Xent with in-batch negatives on L2-normalised anchor/positive projections (positive of row i is zp[i])."""
    logits = (za @ zp.t()) / temperature
    labels = torch.arange(za.size(0), device=za.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def load_head(
    save_dir: str, in_dim: int, hidden_dim: int, out_dim: int, device: str,
    filename: str = "head.pt",
) -> ProjectionHead:
    """Rebuild the head from config dims and load `filename` (`head.file`: `head.pt` = val-loss selection, `head_probe.pt` = partial-probe selection)."""
    head = ProjectionHead(in_dim, hidden_dim, out_dim)
    state = torch.load(Path(save_dir) / filename, map_location=device)
    head.load_state_dict(state)
    return head.to(device).eval()


class ResidualQueryHead(nn.Module):
    """Query-only residual head on the whitened + L2 frozen vector: L2(v + correction(v)).

    The last layer starts at zero (identity at init). The gallery stays the frozen whitened vector.
    """

    def __init__(self, dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(v + self.net(v), dim=-1)


def load_query_head(path, device: str = "cpu") -> tuple[ResidualQueryHead, dict]:
    """Head v2 checkpoint -> (head in eval mode, checkpoint dict with the whitening it was trained on)."""
    ckpt = torch.load(Path(path), map_location=device)
    head = ResidualQueryHead(int(ckpt["dim"]), int(ckpt["hidden_dim"]))
    head.load_state_dict(ckpt["state_dict"])
    return head.to(device).eval(), ckpt
