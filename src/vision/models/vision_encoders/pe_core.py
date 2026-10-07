from typing import Callable

import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config

from src.vision.data.preprocess import compose_transform
from src.vision.models.vision_encoders.base import BaseVisionEncoder, pool_global


class PECoreEncoder(BaseVisionEncoder):
    """PE-Core encoder (Perception Encoder, Meta, arXiv 2504.13181): CLIP-style contrastive vision-language.

    The paper's thesis is that the best visual features sit in intermediate layers, the last one
    being specialised on text alignment, so `extraction_layer` is a key hyperparameter.
    B/16 at 224px: ViT-B, patch 16, width 768, 12 layers, 196 patch tokens.

    Differences from the other encoders:
      1. Loaded from timm (`vit_pe_core_base_patch16_224.fb`, apache-2.0, not gated), not
         transformers: the HF repo ships a raw `.pt` needing Meta's `perception_models`.
      2. Normalisation mean=std=0.5 (not ImageNet), read from the pretrained_cfg.
      3. Native pooling is an 8-head attention pool + CLIP projection, not CLS: native output is
         1024-d, patch tokens 768-d. `embedding_dim` depends on pooling (1024 "natural",
         768 "mean"/"gem") and is inferred with a dummy forward, as in RADIO.
      4. Resolution fixed at 224: the checkpoint declares `fixed_input_size=True` and RoPE is tied
         to a 14x14 grid, so any other side fails a timm assert.

    Args:
        timm_name: timm model name (weights come from HuggingFace).
        image_size: input side in pixels; must be 224 for this checkpoint.
        pooling: "natural" (attention pool + CLIP projection, 1024d) | "mean" | "gem" over patch tokens (768d).
        gem_p: GeM exponent (only for "gem").
        extraction_layer: layer to read tokens from (-1 = last). With `!= -1` the attention pool
            does not apply (it works on the last block only): "natural" falls back to that
            layer's CLS token, as in DINOv2, and the embedding drops to 768d (same asymmetry as SigLIP2).
    """

    def __init__(
        self,
        timm_name: str = "vit_pe_core_base_patch16_224.fb",
        image_size: int = 224,
        pooling: str = "natural",
        gem_p: float = 3.0,
        extraction_layer: int = -1,
        **_,
    ):
        super().__init__()
        print(f"[PECoreEncoder] Caricamento {timm_name}...")

        # keep num_classes: the head (Linear 768->1024) is the CLIP contrastive projection, not an ImageNet classifier
        self.model = timm.create_model(timm_name, pretrained=True)
        self._freeze()

        # fixed resolution: fail here rather than after loading the dataset
        native_size = self.model.pretrained_cfg["input_size"][-1]
        if image_size != native_size:
            raise ValueError(
                f"image_size={image_size} non supportato da {timm_name}: il "
                f"checkpoint e' a risoluzione fissa {native_size}px (RoPE su "
                f"griglia fissa). Per cambiare risoluzione serve un'altra "
                f"variante PE-Core (es. large_patch14_336)."
            )
        self.image_size = image_size

        # mean/std from the checkpoint pretrained_cfg: 0.5/0.5, not ImageNet
        data_cfg  = resolve_data_config({}, model=self.model)
        self.mean = list(data_cfg["mean"])
        self.std  = list(data_cfg["std"])

        self.pooling          = pooling
        self.gem_p            = gem_p
        self.extraction_layer = extraction_layer

        # D inferred with a dummy forward since it depends on pooling (attention pool: 1024, patch tokens: 768)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            self._embedding_dim = self.forward(dummy).shape[-1]

        print(f"[PECoreEncoder] Pronto: {self._embedding_dim}d, pooling={pooling}, "
              f"layer={extraction_layer}, input {self.image_size}px")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.extraction_layer == -1:
            if self.pooling == "natural":
                # native pooling: attention pool + CLIP projection = full timm forward -> [B, 1024]
                pooled = self.model(images)
            else:
                # last layer, post final layernorm: [B, 1+N, D] (1 prefix token)
                tokens  = self.model.forward_features(images)
                patches = tokens[:, self.model.num_prefix_tokens:, :]   # [B, N, D]
                pooled  = pool_global(None, patches, self.pooling, self.gem_p)
        else:
            # intermediate layer, pre-layernorm (norm=False), as HF hidden_states[layer];
            # negative index converted to an absolute block position, as in TIPSv2
            n_blocks = len(self.model.blocks)
            layer    = (n_blocks + self.extraction_layer
                        if self.extraction_layer < 0 else self.extraction_layer)
            (patches, prefix), = self.model.forward_intermediates(
                images, indices=[layer], return_prefix_tokens=True,
                norm=False, output_fmt="NLC", intermediates_only=True,
            )                                        # [B, N, D], [B, 1, D]
            natural = prefix[:, 0, :]                # CLS token of the chosen layer
            pooled  = pool_global(natural, patches, self.pooling, self.gem_p)

        return nn.functional.normalize(pooled, p=2, dim=1)

    def build_transform(self) -> Callable:
        return compose_transform(self.image_size, self.mean, self.std)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
