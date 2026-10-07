from src.vision.models.vision_encoders.dinov2 import DINOv2Encoder


class DINOv3Encoder(DINOv2Encoder):
    """DINOv3 encoder (Meta); same HuggingFace interface as DINOv2.

    Official checkpoints are gated: accept the licence on the model page and authenticate
    (`huggingface-cli login`).
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
