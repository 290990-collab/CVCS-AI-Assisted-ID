# src/vision/models/vision_encoders/__init__.py
"""
Encoder vision intercambiabili per il retrieval.

Tutti gli encoder rispettano il contratto BaseVisionEncoder, quindi sono
interscambiabili senza toccare la pipeline FAISS. La selezione avviene per
nome tramite ENCODER_REGISTRY (vedi VisionModelManager / config YAML).

Set di encoder del benchmark (mid-paper):
  - dinov2 / dinov3 -> self-supervised (Meta)
  - siglip2         -> vision-language (Google)
  - radio           -> agglomerativo/"misto" (NVIDIA, distilla DINOv2+CLIP+SAM)
  - ijepa           -> self-supervised predittivo (Meta), vicino al partial retrieval
  - tipsv2          -> vision-language con allineamento patch-testo (Google)
  - pecore          -> vision-language "Perception Encoder" (Meta), feature
                       migliori nei layer intermedi che all'uscita
  - pespatial       -> stesso "Perception Encoder" ri-allineato con obiettivo
                       DENSO: coppia controllata di pecore (stesso ViT-B/16)
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

# nome (usato nei config) -> classe encoder
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
    """
    Istanzia l'encoder registrato sotto 'name', passandogli 'kwargs'.

    Args:
        name:   chiave del registry (dinov2 | dinov3 | siglip2 | radio | ijepa |
                tipsv2 | pecore | pespatial).
        kwargs: argomenti del costruttore dell'encoder (es. hf_name, image_size).

    Raises:
        ValueError: se `name` non e' un encoder registrato.
    """
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
