# src/vision/models/vision_encoders/radio.py

from typing import Callable

import torch
import torch.nn as nn
from transformers import AutoModel

from src.vision.data.preprocess import compose_transform
from src.vision.models.vision_encoders.base import BaseVisionEncoder, pool_global


class RADIOEncoder(BaseVisionEncoder):
    """
    Encoder basato su RADIO (NVIDIA, modello "agglomerativo").
    Backbone unico distillato da piu' "teacher" (DINOv2, CLIP, SAM):
    eredita le proprieta' di tutti.

    Particolarita' rispetto agli altri encoder:
      1. Si carica con `trust_remote_code=True`.
      2. Il forward ritorna una COPPIA (summary, spatial_features).
      3. RADIO applica la PROPRIA normalizzazione internamente.

    Args:
        hf_name:    checkpoint HuggingFace
        image_size: lato input (multiplo del patch size, es. 224).
        pooling:    "natural" (summary token) | "mean" | "gem" sulle spatial features.
        gem_p:      esponente del GeM (usato solo se pooling="gem").

    Note:
        La dimensione dell'embedding viene dedotta dinamicamente con un forward
        di prova, perche' dipende dalla variante RADIO scelta. `extraction_layer`
        non è applicabile (l'API espone solo l'output finale).
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

        # dimensione dell'embedding dedotta con un forward di prova (su CPU),
        # passando per il pooling scelto (summary e spatial possono differire)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            summary, spatial = self.model(dummy)
            pooled = pool_global(summary, spatial, self.pooling, self.gem_p)
        self._embedding_dim = pooled.shape[-1]
        print(f"[RADIOEncoder] Pronto: {self._embedding_dim}d, pooling={pooling}, "
              f"input {self.image_size}px")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W] in [0, 1] (RADIO normalizza internamente)
        summary, spatial = self.model(images)
        pooled = pool_global(summary, spatial, self.pooling, self.gem_p)
        return nn.functional.normalize(pooled, p=2, dim=1)

    def build_transform(self) -> Callable:
        return compose_transform(self.image_size, self.mean, self.std)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
