from typing import Callable

import torch
import torch.nn as nn
from transformers import AutoModel

from src.vision.data.preprocess import compose_transform
from src.vision.models.vision_encoders.base import BaseVisionEncoder, pool_global


class RADIOEncoder(BaseVisionEncoder):
    """RADIO encoder (NVIDIA, agglomerative): one backbone distilled from several teachers (DINOv2, CLIP, SAM).

    Differences from the other encoders: loaded with `trust_remote_code=True`; forward returns a
    pair (summary, spatial_features); RADIO normalises internally. `embedding_dim` is inferred with a
    dummy forward (depends on the variant); `extraction_layer` does not apply (only the final output).

    Args:
        hf_name: HuggingFace checkpoint.
        image_size: input side (multiple of the patch size, e.g. 224).
        pooling: "natural" (summary token) | "mean" | "gem" over spatial features.
        gem_p: GeM exponent (only for "gem").
    """

    def __init__(
        self,
        hf_name: str = "nvidia/C-RADIOv2-B",
        image_size: int = 224,
        pooling: str = "natural",
        gem_p: float = 3.0,
        **_,
    ):
        super().__init__()
        print(f"[RADIOEncoder] Caricamento {hf_name}...")

        self.model = AutoModel.from_pretrained(hf_name, trust_remote_code=True)
        self._freeze()

        self.image_size = image_size
        self.mean = [0.0, 0.0, 0.0]
        self.std  = [1.0, 1.0, 1.0]
        self.pooling = pooling
        self.gem_p   = gem_p

        # embedding size inferred with a dummy forward through the chosen pooling (summary and spatial may differ)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            summary, spatial = self.model(dummy)
            pooled = pool_global(summary, spatial, self.pooling, self.gem_p)
        self._embedding_dim = pooled.shape[-1]
        print(f"[RADIOEncoder] Pronto: {self._embedding_dim}d, pooling={pooling}, "
              f"input {self.image_size}px")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images in [0, 1]: RADIO normalises internally
        summary, spatial = self.model(images)
        pooled = pool_global(summary, spatial, self.pooling, self.gem_p)
        return nn.functional.normalize(pooled, p=2, dim=1)

    def build_transform(self) -> Callable:
        return compose_transform(self.image_size, self.mean, self.std)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
