# src/vision/models/vision_encoders/base.py

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn

# pooling supportati per ottenere l'embedding globale dai token dell'encoder.
POOLINGS = ("natural", "mean", "gem")


def gem_pool(tokens: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """
    GeM (Generalized Mean) pooling sui patch token: media generalizzata con
    esponente `p`. p=1 equivale alla media, p→∞ tende al max pooling.

    Args:
        tokens: [B, N, D] (N patch token per immagine).

    Returns:
        [B, D] embedding globale.
    """
    return tokens.clamp(min=eps).pow(p).mean(dim=1).pow(1.0 / p)


def pool_global(
    natural: torch.Tensor | None,
    patches: torch.Tensor | None,
    pooling: str,
    gem_p: float = 3.0,
) -> torch.Tensor:
    """
    Sceglie l'embedding globale [B, D] in base alla strategia di pooling.

    Args:
        natural: pooling NATIVO dell'encoder (CLS per DINO, attention-pool per
                 SigLIP2, summary per RADIO, mean per I-JEPA); usato se pooling="natural".
        patches: [B, N, D] patch token, usati per "mean"/"gem".
        pooling: "natural" | "mean" | "gem".
        gem_p:   esponente del GeM.
    """
    if pooling == "natural":
        return natural
    if pooling == "mean":
        return patches.mean(dim=1)
    if pooling == "gem":
        return gem_pool(patches, gem_p)
    raise ValueError(f"pooling '{pooling}' non valido (scegli tra {POOLINGS})")


class BaseVisionEncoder(nn.Module, ABC):
    """
    Contratto comune a tutti gli encoder vision usati per il retrieval.

    Ogni encoder concreto e' responsabile di tre cose:
      1. forward(images) -> [B, D] embedding L2-normalizzato. 
        Il pooling dipende dal modello;
      2. build_transform() -> il preprocessing SPECIFICO del modello.
      3. embedding_dim -> la dimensione D dell'embedding prodotto.

    Tutti gli encoder sono usati FROZEN (feature extraction pura).
    """

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: batch di immagini gia' preprocessate, shape [B, 3, H, W].

        Returns:
            embeddings: shape [B, D], L2-normalizzati (norma 1 per riga).
        """
        raise NotImplementedError

    @abstractmethod
    def build_transform(self) -> Callable:
        """
        Returns:
            transform torchvision (callable: PIL.Image -> Tensor [3, H, W])
            con normalizzazione e risoluzione corrette per QUESTO modello.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensione D dell'embedding prodotto da forward()."""
        raise NotImplementedError

    def _freeze(self) -> None:
        """
        Congela tutti i parametri del modulo: feature extraction senza
        training (i gradienti non vengono calcolati ne' aggiornati).
        """
        for param in self.parameters():
            param.requires_grad = False
