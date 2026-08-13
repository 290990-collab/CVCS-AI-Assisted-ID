# src/vision/models/vision_encoders/ijepa.py

from typing import Callable

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

from src.vision.data.preprocess import compose_transform
from src.vision.models.vision_encoders.base import BaseVisionEncoder, pool_global


class IJepaEncoder(BaseVisionEncoder):
    """
    Encoder basato su I-JEPA (Image Joint-Embedding Predictive Architecture, Meta).
    I-JEPA e' self-supervised ma "predittivo", NON ha un token [CLS], produce solo token di patch.

    Pooling nativo ("natural") = mean dei patch token; selezionabile anche "gem".

    Args:
        hf_name:    checkpoint HuggingFace
        image_size: lato input in pixel.
        pooling:    "natural"/"mean" (coincidono) | "gem" sui patch token.
        gem_p:      esponente del GeM (usato solo se pooling="gem").
        extraction_layer: indice del layer da cui leggere i patch token (-1 = ultimo).

    Note:
        I-JEPA usa la normalizzazione ImageNet, letta dal processor del modello.
        I checkpoint ufficiali esistono solo in taglie grandi (ViT-H / ViT-g):
    """

    def __init__(
        self,
        hf_name: str = "facebook/ijepa_vith14_1k",
        image_size: int = 224,
        pooling: str = "natural",
        gem_p: float = 3.0,
        extraction_layer: int = -1,
        **_,
    ):
        super().__init__()
        print(f"[IJepaEncoder] Caricamento {hf_name}...")

        self.backbone  = AutoModel.from_pretrained(hf_name)
        self.processor = AutoImageProcessor.from_pretrained(hf_name)
        self._freeze()

        # image_size esplicito (default 224); mean/std letti dal processor.
        self.image_size = image_size
        self.mean       = list(self.processor.image_mean)
        self.std        = list(self.processor.image_std)

        # pooling + layer di estrazione
        self.pooling          = pooling
        self.gem_p            = gem_p
        self.extraction_layer = extraction_layer

        # D = dimensione nascosta del backbone (dim dei token di patch)
        self._embedding_dim = self.backbone.config.hidden_size
        print(f"[IJepaEncoder] Pronto: {self._embedding_dim}d, pooling={pooling}, "
              f"layer={extraction_layer}, input {self.image_size}px")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W]
        # extraction_layer=-1 = last_hidden_state (post-layernorm, baseline); per
        # layer intermedi si usa hidden_states[layer] (pre-layernorm).
        last_only = self.extraction_layer == -1
        outputs = self.backbone(pixel_values=images, output_hidden_states=not last_only)
        patches = (outputs.last_hidden_state if last_only
                   else outputs.hidden_states[self.extraction_layer])   # [B, N, D] (no CLS)
        natural = patches.mean(dim=1)                            # pooling nativo I-JEPA
        pooled  = pool_global(natural, patches, self.pooling, self.gem_p)
        return nn.functional.normalize(pooled, p=2, dim=1)

    def build_transform(self) -> Callable:
        return compose_transform(self.image_size, self.mean, self.std)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
