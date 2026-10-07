from typing import Callable

import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config

from src.vision.data.preprocess import compose_transform
from src.vision.models.vision_encoders.base import BaseVisionEncoder, pool_global


class PESpatialEncoder(BaseVisionEncoder):
    """PE-Spatial encoder (Perception Encoder, Meta, arXiv 2504.13181), B/16-512.

    The "spatial" branch of the PE-Core family: same contrastive backbone re-aligned with a dense
    objective (spatial distillation + self-distillation), so every patch token matters, which suits
    floor-plan retrieval (room layout). Controlled pair with `pecore`: same ViT-B/16, normalisation
    (mean=std=0.5) and timm loading; only the training differs.

    Differences from PE-Core:
      1. No contrastive head: the checkpoint has `num_classes=0`, `head=Identity`, so there is no
         attention pool + CLIP projection; the embedding is 768d for all three poolings.
      2. `model(images)` is not the natural pooling: the checkpoint `global_pool` is "avg", so the
         full forward returns the patch-token mean (cos = 1.0000 with the explicit mean). "natural"
         reads the CLS token (`num_prefix_tokens = 1`) as in DINOv2/DINOv3.
      3. Resolution is not locked despite `fixed_input_size=True`: timm resamples `pos_embed` at load
         (1025 -> 197 tokens from 512 to 224). Default 224 aligns with the rest of the grid; 512 is
         the native resolution.

    Weights: the HF repo `facebook/PE-Spatial-B16-512` ships a raw `.pt` needing Meta's
    `perception_models`; the same weights are on `timm/vit_pe_spatial_base_patch16_512.fb`
    (apache-2.0, not gated), used here as in PE-Core.

    Args:
        timm_name: timm model name (weights come from HuggingFace).
        image_size: input side in pixels, multiple of the patch size (16); 224 = grid, 512 = native.
        pooling: "natural" (CLS) | "mean" | "gem" over patch tokens; all 768d.
        gem_p: GeM exponent (only for "gem").
        extraction_layer: layer to read tokens from (-1 = last). PE's "best features are not in the
            last layer" thesis arises on Core; on Spatial the dense fine-tuning works on the output.
    """

    def __init__(
        self,
        timm_name: str = "vit_pe_spatial_base_patch16_512.fb",
        image_size: int = 224,
        pooling: str = "natural",
        gem_p: float = 3.0,
        extraction_layer: int = -1,
        **_,
    ):
        super().__init__()
        print(f"[PESpatialEncoder] Caricamento {timm_name} a {image_size}px...")

        # resolution goes to create_model (timm builds the patch grid and resamples pos_embed);
        # it must be a multiple of the patch size, else patch embedding fails with an obscure assert
        patch = timm.get_pretrained_cfg(timm_name).to_dict().get("input_size")[-1]
        if image_size % 16 != 0:
            raise ValueError(
                f"image_size={image_size} non valido per {timm_name}: deve "
                f"essere un multiplo del patch size (16). Nativa del "
                f"checkpoint: {patch}px; la griglia del progetto usa 224."
            )
        self.image_size = image_size

        # num_classes=0 in the checkpoint: head=Identity, no projection to preserve (unlike PE-Core)
        self.model = timm.create_model(
            timm_name, pretrained=True, img_size=image_size
        )
        self._freeze()

        # mean/std from the checkpoint pretrained_cfg: 0.5/0.5, not ImageNet
        data_cfg  = resolve_data_config({}, model=self.model)
        self.mean = list(data_cfg["mean"])
        self.std  = list(data_cfg["std"])

        self.pooling          = pooling
        self.gem_p            = gem_p
        self.extraction_layer = extraction_layer

        # D inferred with a dummy forward (768 here) to avoid a hard-coded constant
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            self._embedding_dim = self.forward(dummy).shape[-1]

        print(f"[PESpatialEncoder] Pronto: {self._embedding_dim}d, pooling={pooling}, "
              f"layer={extraction_layer}, input {self.image_size}px")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        n_prefix = self.model.num_prefix_tokens

        if self.extraction_layer == -1:
            # last layer, post final layernorm: [B, 1+N, D]; not self.model(images), which is
            # global_pool="avg" (see class docstring, point 2)
            tokens  = self.model.forward_features(images)
            natural = tokens[:, 0, :]                      # token CLS
            patches = tokens[:, n_prefix:, :]              # [B, N, D]
        else:
            # intermediate layer, pre-layernorm (norm=False), as HF hidden_states[layer];
            # negative index converted to an absolute block position, as in PE-Core/TIPSv2
            n_blocks = len(self.model.blocks)
            layer    = (n_blocks + self.extraction_layer
                        if self.extraction_layer < 0 else self.extraction_layer)
            (patches, prefix), = self.model.forward_intermediates(
                images, indices=[layer], return_prefix_tokens=True,
                norm=False, output_fmt="NLC", intermediates_only=True,
            )                                              # [B, N, D], [B, 1, D]
            natural = prefix[:, 0, :]                      # CLS token of the chosen layer

        pooled = pool_global(natural, patches, self.pooling, self.gem_p)
        return nn.functional.normalize(pooled, p=2, dim=1)

    def build_transform(self) -> Callable:
        return compose_transform(self.image_size, self.mean, self.std)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
