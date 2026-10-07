from omegaconf import DictConfig, OmegaConf

from src.vision.models.vision_encoders import BaseVisionEncoder, build_encoder


class VisionModelManager:
    """
    Builds the vision encoder from a config and exposes it with its transform.

    config.model: name (registry key), device ("cuda" | "cpu"), kwargs (encoder constructor args).
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.device = config.model.device

        self.encoder: BaseVisionEncoder = self.__build__()
        self.encoder.to(self.device)

        self.encoder.eval()

        # encoder-specific preprocessing
        self.transform = self.encoder.build_transform()


    def __build__(self) -> BaseVisionEncoder:
        """Instantiate the encoder named in config.model.name."""
        name   = self.config.model.name
        kwargs = self._kwargs_to_dict(self.config.model.get("kwargs"))
        print(f"[VisionModelManager] Encoder '{name}' su {self.device} con kwargs={kwargs}")
        return build_encoder(name, **kwargs)


    @staticmethod
    def _kwargs_to_dict(kwargs) -> dict:
        """OmegaConf node -> plain dict."""
        if kwargs is None:
            return {}
        if isinstance(kwargs, DictConfig):
            return OmegaConf.to_container(kwargs, resolve=True)
        return dict(kwargs)
