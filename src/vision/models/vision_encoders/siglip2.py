# src/vision/models/vision_encoders/siglip2.py

from typing import Callable

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

from src.vision.data.preprocess import compose_transform
from src.vision.models.vision_encoders.base import BaseVisionEncoder, pool_global


class SigLIP2Encoder(BaseVisionEncoder):
    """
    Encoder basato su SigLIP2 (Google, vision-language con sigmoid loss).

    Args:
        hf_name:    checkpoint HuggingFace
        image_size: lato input in pixel. Default 224 (variante -224).
        pooling:    "natural" (attention-pool nativo) | "mean" | "gem" sui patch token.
        gem_p:      esponente del GeM (usato solo se pooling="gem").

    Note:
        La risoluzione è legata al checkpoint (-224/-256/-384): per cambiarla si
        cambia `hf_name`, non basta `image_size`. `extraction_layer` non è
        applicabile (la testa di attention-pool opera solo sull'ultimo layer).
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

        # image_size esplicito (default 224); mean/std letti dal processor.
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
            # SigLIP2 non ha un token [CLS]: pooling su tutti i patch token.
            patches  = self.model.vision_model(pixel_values=images).last_hidden_state
            features = pool_global(None, patches, self.pooling, self.gem_p)
        return nn.functional.normalize(features, p=2, dim=1)

    def build_transform(self) -> Callable:
        return compose_transform(self.image_size, self.mean, self.std)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
