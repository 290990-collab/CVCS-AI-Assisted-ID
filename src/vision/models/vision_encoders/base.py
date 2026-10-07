from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn

POOLINGS = ("natural", "mean", "gem")


def gem_pool(tokens: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """GeM pooling (generalised mean, exponent `p`; p=1 is the mean, p->inf the max) of patch tokens [B, N, D] -> [B, D]."""
    return tokens.clamp(min=eps).pow(p).mean(dim=1).pow(1.0 / p)


def pool_global(
    natural: torch.Tensor | None,
    patches: torch.Tensor | None,
    pooling: str,
    gem_p: float = 3.0,
) -> torch.Tensor:
    """Global embedding [B, D] by pooling strategy.

    Args:
        natural: encoder-native pooling (CLS for DINO, attention-pool for SigLIP2, summary for
            RADIO, mean for I-JEPA), used if pooling="natural".
        patches: [B, N, D] patch tokens, used by "mean"/"gem".
    """
    if pooling == "natural":
        return natural
    if pooling == "mean":
        return patches.mean(dim=1)
    if pooling == "gem":
        return gem_pool(patches, gem_p)
    raise ValueError(f"pooling '{pooling}' non valido (scegli tra {POOLINGS})")


class BaseVisionEncoder(nn.Module, ABC):
    """Contract of the frozen vision encoders: forward(images) -> L2-normalised [B, D] (model-specific pooling), build_transform() (model-specific preprocessing), embedding_dim."""

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocessed images [B, 3, H, W] -> unit-norm embeddings [B, D]."""
        raise NotImplementedError

    @abstractmethod
    def build_transform(self) -> Callable:
        """Torchvision transform (PIL.Image -> Tensor [3, H, W]) with this model's normalisation and resolution."""
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Embedding size D of forward()."""
        raise NotImplementedError

    def _freeze(self) -> None:
        """Freeze all parameters (pure feature extraction)."""
        for param in self.parameters():
            param.requires_grad = False
