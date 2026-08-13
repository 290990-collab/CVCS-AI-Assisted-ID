# src/vision/models/vision_encoders/dinov2.py

from typing import Callable

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

from src.vision.data.preprocess import compose_transform
from src.vision.models.vision_encoders.base import BaseVisionEncoder, pool_global


class DINOv2Encoder(BaseVisionEncoder):
    """
    Encoder basato su DINOv2 (self-supervised, Meta).
    Estrae un embedding globale per immagine; il pooling è configurabile.

    Args:
        hf_name:    checkpoint HuggingFace
        image_size: lato (quadrato) dell'immagine in input, in pixel.
        pooling:    "natural" (CLS) | "mean" | "gem" sui patch token.
        gem_p:      esponente del GeM (usato solo se pooling="gem").
        extraction_layer: indice del layer da cui leggere i token (-1 = ultimo,
                    -2 = penultimo, ...); i layer intermedi sono spesso più
                    trasferibili per il retrieval.

    Note:
        DINOv2 usa la normalizzazione ImageNet.
    """

    def __init__(
        self,
        hf_name: str = "facebook/dinov2-base",
        image_size: int = 224,
        pooling: str = "natural",
        gem_p: float = 3.0,
        extraction_layer: int = -1,
        **_,
    ):
        super().__init__()
        print(f"[DINOv2Encoder] Caricamento {hf_name}...")

        # backbone + processor
        self.backbone  = AutoModel.from_pretrained(hf_name)
        self.processor = AutoImageProcessor.from_pretrained(hf_name)
        self._freeze()

        # preprocessing del modello
        self.image_size = image_size
        self.mean       = list(self.processor.image_mean)
        self.std        = list(self.processor.image_std)

        # pooling + layer di estrazione
        self.pooling          = pooling
        self.gem_p            = gem_p
        self.extraction_layer = extraction_layer

        # D = dimensione nascosta del backbone (invariante al pooling)
        self._embedding_dim = self.backbone.config.hidden_size
        print(f"[DINOv2Encoder] Pronto: {self._embedding_dim}d, pooling={pooling}, "
              f"layer={extraction_layer}, input {self.image_size}px")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W]
        # extraction_layer=-1 = last_hidden_state (post-layernorm, baseline); per
        # layer intermedi si usa hidden_states[layer] (pre-layernorm).
        last_only = self.extraction_layer == -1
        outputs = self.backbone(pixel_values=images, output_hidden_states=not last_only)
        hidden  = (outputs.last_hidden_state if last_only
                   else outputs.hidden_states[self.extraction_layer])   # [B, 1+N, D]
        natural = hidden[:, 0, :]                                # token [CLS]
        patches = hidden[:, 1:, :]                               # token di patch
        pooled  = pool_global(natural, patches, self.pooling, self.gem_p)
        return nn.functional.normalize(pooled, p=2, dim=1)

    def build_transform(self) -> Callable:
        return compose_transform(self.image_size, self.mean, self.std)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
