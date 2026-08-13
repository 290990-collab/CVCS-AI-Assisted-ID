# src/vision/models/projection_head.py

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    MLP leggero allenato SOPRA l'embedding di un encoder frozen (Fase 3).

    Rimodella lo spazio dell'encoder verso il nostro compito (similarità
    architettonica) senza toccare il backbone. L'uscita è L2-normalizzata, così
    si indicizza con FAISS IndexFlatIP esattamente come gli embedding frozen.

    Args:
        in_dim:     dimensione dell'embedding dell'encoder (768/1280/2304...).
        hidden_dim: larghezza dello strato nascosto.
        out_dim:    dimensione dello spazio proiettato.
    """

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
    """
    InfoNCE/NT-Xent simmetrica con negativi in-batch.

    `za`, `zp` sono le proiezioni L2-normalizzate di àncora e positivo (stessa
    pianta sotto degrado/augmentation). Per ogni riga i il positivo è zp[i]; i
    negativi sono tutte le altre righe del batch. La temperatura `temperature`
    "affila" la distribuzione (più bassa = penalizza di più i negativi vicini).
    """
    logits = (za @ zp.t()) / temperature
    labels = torch.arange(za.size(0), device=za.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def load_head(
    save_dir: str, in_dim: int, hidden_dim: int, out_dim: int, device: str
) -> ProjectionHead:
    """Ricostruisce la head dai dim del config e ne carica i pesi da `head.pt`."""
    head = ProjectionHead(in_dim, hidden_dim, out_dim)
    state = torch.load(Path(save_dir) / "head.pt", map_location=device)
    head.load_state_dict(state)
    return head.to(device).eval()
