from typing import Callable

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

from src.vision.data.preprocess import compose_transform
from src.vision.models.vision_encoders.base import BaseVisionEncoder, pool_global


class IJepaEncoder(BaseVisionEncoder):
    """I-JEPA encoder (Meta, predictive self-supervised): no [CLS] token, only patch tokens.

    Native pooling ("natural") is the patch-token mean; "gem" is also available.
    Normalisation read from the model processor; official checkpoints exist only as ViT-H / ViT-g.

    Args:
        hf_name: HuggingFace checkpoint.
        image_size: input side in pixels.
        pooling: "natural"/"mean" (identical) | "gem".
        extraction_layer: layer to read patch tokens from (-1 = last).
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

        # mean/std from the processor
        self.image_size = image_size
        self.mean       = list(self.processor.image_mean)
        self.std        = list(self.processor.image_std)

        self.pooling          = pooling
        self.gem_p            = gem_p
        self.extraction_layer = extraction_layer

        # D = backbone hidden size
        self._embedding_dim = self.backbone.config.hidden_size
        print(f"[IJepaEncoder] Pronto: {self._embedding_dim}d, pooling={pooling}, "
              f"layer={extraction_layer}, input {self.image_size}px")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # extraction_layer=-1: last_hidden_state (post-layernorm); otherwise hidden_states[layer] (pre-layernorm)
        last_only = self.extraction_layer == -1
        outputs = self.backbone(pixel_values=images, output_hidden_states=not last_only)
        patches = (outputs.last_hidden_state if last_only
                   else outputs.hidden_states[self.extraction_layer])   # [B, N, D] (no CLS)
        natural = patches.mean(dim=1)                            # native I-JEPA pooling
        pooled  = pool_global(natural, patches, self.pooling, self.gem_p)
        return nn.functional.normalize(pooled, p=2, dim=1)

    def build_transform(self) -> Callable:
        return compose_transform(self.image_size, self.mean, self.std)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
