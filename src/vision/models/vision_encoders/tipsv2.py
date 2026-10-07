from typing import Callable

import torch
import torch.nn as nn
from transformers import AutoModel

from src.vision.data.preprocess import compose_transform
from src.vision.models.vision_encoders.base import BaseVisionEncoder, pool_global


class TIPSv2Encoder(BaseVisionEncoder):
    """
    TIPSv2 image encoder (B/14: ViT-B, patch 14, embed_dim 768).

    Differences from the other encoders:
      - loaded with `trust_remote_code=True` (custom modeling code on the HF repo);
      - input in [0, 1] without ImageNet normalisation (mean=0, std=1), as RADIO;
      - backbone returns (cls [B,1,D], registers [B,R,D], patches [B,N,D]); registers are dropped;
      - native resolution 448 (32x32 patches); other sizes interpolate positional embeddings.

    pooling: "natural" (CLS) | "mean" | "gem" over patch tokens. extraction_layer: -1 = last.
    The text encoder is unused but loading it needs `sentencepiece`.
    """

    def __init__(
        self,
        hf_name: str = "google/tipsv2-b14",
        image_size: int = 448,
        pooling: str = "natural",
        gem_p: float = 3.0,
        extraction_layer: int = -1,
        **_,
    ):
        super().__init__()
        print(f"[TIPSv2Encoder] Caricamento {hf_name}...")

        self.model = AutoModel.from_pretrained(hf_name, trust_remote_code=True)
        self._freeze()

        # side must be a multiple of the patch size (14)
        patch_size = self.model.config.patch_size
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} non e' multiplo del patch size "
                f"({patch_size}) di {hf_name}."
            )
        native_size = self.model.config.img_size
        if image_size != native_size:
            print(f"[TIPSv2Encoder] ATTENZIONE: input {image_size}px != risoluzione "
                  f"di pretrain {native_size}px -> positional embedding interpolate.")

        # pixels in [0, 1], no ImageNet normalisation
        self.image_size = image_size
        self.mean = [0.0, 0.0, 0.0]
        self.std  = [1.0, 1.0, 1.0]

        self.pooling          = pooling
        self.gem_p            = gem_p
        self.extraction_layer = extraction_layer

        # D = ViT width, independent of pooling
        self._embedding_dim = self.model.config.embed_dim
        print(f"[TIPSv2Encoder] Pronto: {self._embedding_dim}d, pooling={pooling}, "
              f"layer={extraction_layer}, input {self.image_size}px")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W] in [0, 1]
        # backbone instead of encode_image(): that one is @torch.no_grad() and hides intermediate layers
        backbone = self.model.vision_encoder

        if self.extraction_layer == -1:
            # last layer, post final layernorm
            cls_token, _registers, patches = backbone(images)   # [B,1,D] [B,R,D] [B,N,D]
            natural = cls_token[:, 0, :]                        # [B, D]
        else:
            # intermediate layer, pre-layernorm (as HF `hidden_states[layer]`);
            # negative index converted to absolute (API takes absolute only)
            n_blocks = len(backbone.blocks)
            layer    = (n_blocks + self.extraction_layer
                        if self.extraction_layer < 0 else self.extraction_layer)
            (patches, natural), = backbone.get_intermediate_layers(
                images, n=[layer], return_class_token=True, norm=False,
            )

        pooled = pool_global(natural, patches, self.pooling, self.gem_p)
        return nn.functional.normalize(pooled, p=2, dim=1)

    def build_transform(self) -> Callable:
        return compose_transform(self.image_size, self.mean, self.std)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
