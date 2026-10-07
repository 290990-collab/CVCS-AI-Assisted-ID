"""Interchangeable vision encoders for retrieval.

All follow the BaseVisionEncoder contract and are selected by name via ENCODER_REGISTRY
(see VisionModelManager / YAML config).
  - dinov2 / dinov3: self-supervised (Meta)
  - siglip2: vision-language (Google)
  - radio: agglomerative (NVIDIA, distils DINOv2+CLIP+SAM)
  - ijepa: predictive self-supervised (Meta)
  - tipsv2: vision-language with patch-text alignment (Google)
  - pecore: Perception Encoder (Meta); better features in intermediate layers
  - pespatial: same Perception Encoder re-aligned with a dense objective (same ViT-B/16 as pecore)
"""

from src.vision.models.vision_encoders.base import BaseVisionEncoder
from src.vision.models.vision_encoders.dinov2 import DINOv2Encoder
from src.vision.models.vision_encoders.dinov3 import DINOv3Encoder
from src.vision.models.vision_encoders.siglip2 import SigLIP2Encoder
from src.vision.models.vision_encoders.radio import RADIOEncoder
from src.vision.models.vision_encoders.ijepa import IJepaEncoder
from src.vision.models.vision_encoders.tipsv2 import TIPSv2Encoder
from src.vision.models.vision_encoders.pe_core import PECoreEncoder
from src.vision.models.vision_encoders.pe_spatial import PESpatialEncoder

ENCODER_REGISTRY = {
    "dinov2":  DINOv2Encoder,
    "dinov3":  DINOv3Encoder,
    "siglip2": SigLIP2Encoder,
    "radio":   RADIOEncoder,
    "ijepa":   IJepaEncoder,
    "tipsv2":  TIPSv2Encoder,
    "pecore":  PECoreEncoder,
    "pespatial": PESpatialEncoder,
}


def build_encoder(name: str, **kwargs) -> BaseVisionEncoder:
    """Instantiate the encoder registered under `name` with `kwargs`; raises ValueError if unknown."""
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Encoder '{name}' non supportato. "
            f"Disponibili: {sorted(ENCODER_REGISTRY)}"
        )
    return ENCODER_REGISTRY[name](**kwargs)


__all__ = [
    "BaseVisionEncoder",
    "DINOv2Encoder",
    "DINOv3Encoder",
    "SigLIP2Encoder",
    "RADIOEncoder",
    "IJepaEncoder",
    "TIPSv2Encoder",
    "PECoreEncoder",
    "PESpatialEncoder",
    "ENCODER_REGISTRY",
    "build_encoder",
]
