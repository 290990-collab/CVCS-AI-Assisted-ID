# src/vision/models/vision_encoders/dinov3.py

from src.vision.models.vision_encoders.dinov2 import DINOv2Encoder


class DINOv3Encoder(DINOv2Encoder):
    """
    Encoder basato su DINOv3 (Meta, successore di DINOv2).
    DINOv3 espone in HuggingFace la STESSA interfaccia di DINOv2.

    Args:
        hf_name:    checkpoint HuggingFace DINOv3.
        image_size: come in DINOv2Encoder.

    Note:
        I checkpoint DINOv3 ufficiali di Meta su HuggingFace sono "gated":
        la prima volta serve accettare la licenza sulla pagina del modello
        ed essere autenticati (`huggingface-cli login`).
    """

    def __init__(
        self,
        hf_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        image_size: int = 224,
        pooling: str = "natural",
        gem_p: float = 3.0,
        extraction_layer: int = -1,
        **_,
    ):
        super().__init__(
            hf_name=hf_name,
            image_size=image_size,
            pooling=pooling,
            gem_p=gem_p,
            extraction_layer=extraction_layer,
        )
