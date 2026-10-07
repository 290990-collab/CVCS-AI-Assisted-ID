from typing import Callable

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

from src.vision.data.preprocess import compose_transform
from src.vision.models.vision_encoders.base import BaseVisionEncoder, pool_global


class SigLIP2Encoder(BaseVisionEncoder):
    """
    SigLIP2 image encoder (sigmoid-loss vision-language).

    pooling: "natural" (native attention-pool) | "mean" | "gem" over patch tokens.
    Resolution is tied to the checkpoint (-224/-256/-384): change `hf_name`, not just
    `image_size`. No `extraction_layer`: the attention-pool head reads the last layer only.
    """

    def __init__(
        self,
        hf_name: str = "google/siglip2-base-patch16-224",
        image_size: int = 224,
        pooling: str = "natural",
        gem_p: float = 3.0,
        **_,
    ):
        super().__init__()
        print(f"[SigLIP2Encoder] Caricamento {hf_name}...")

        self.model     = AutoModel.from_pretrained(hf_name)
        self.processor = AutoImageProcessor.from_pretrained(hf_name)
        self._freeze()

        # mean/std from the processor
        self.image_size = image_size
        self.mean       = list(self.processor.image_mean)
        self.std        = list(self.processor.image_std)

        self.pooling = pooling
        self.gem_p   = gem_p

        self._embedding_dim = self.model.config.vision_config.hidden_size
        print(f"[SigLIP2Encoder] Pronto: {self._embedding_dim}d, pooling={pooling}, "
              f"input {self.image_size}px")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W]
        if self.pooling == "natural":
            features = self.model.get_image_features(pixel_values=images)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output                       # [B, D]
        else:
            # no [CLS] token: pool over all patch tokens
            patches  = self.model.vision_model(pixel_values=images).last_hidden_state
            features = pool_global(None, patches, self.pooling, self.gem_p)
        return nn.functional.normalize(features, p=2, dim=1)

    def build_transform(self) -> Callable:
        return compose_transform(self.image_size, self.mean, self.std)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
