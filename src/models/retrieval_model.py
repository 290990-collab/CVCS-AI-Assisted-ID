import torch
import torch.nn as nn


class RetrievalEncoder(nn.Module):
    """Minimal encoder placeholder for graph-level embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Placeholder: x should become graph-level features from a real GNN.
        return self.net(x)
