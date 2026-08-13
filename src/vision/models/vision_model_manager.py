# src/vision/models/vision_model_manager.py

from omegaconf import DictConfig, OmegaConf

from src.vision.models.vision_encoders import BaseVisionEncoder, build_encoder


class VisionModelManager:
    """
    Costruisce l'encoder vision a partire da una config

    Responsabilita' del manager:
      - leggere config.model.name e config.model.kwargs;
      - istanziare l'encoder giusto via registry (build_encoder);
      - spostarlo sul device e metterlo in eval();
      - esporre l'encoder e il suo preprocessing (transform)

    Args:
        config: OmegaConf con almeno:
                model.name   -> chiave del registry (dinov2|dinov3|siglip2|radio|ijepa|tipsv2)
                model.device -> "cuda" | "cpu"
                model.kwargs -> argomenti del costruttore dell'encoder
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.device = config.model.device

        self.encoder: BaseVisionEncoder = self.__build__()
        self.encoder.to(self.device)
        
        # imposta il modello in modalità evaluation --> dropout disattivato e BatchNorm con specifiche fisse
        self.encoder.eval()

        # il preprocessing e' quello SPECIFICO dell'encoder scelto
        self.transform = self.encoder.build_transform()


    def __build__(self) -> BaseVisionEncoder:
        """Istanzia l'encoder selezionato in config.model.name."""
        name   = self.config.model.name
        kwargs = self._kwargs_to_dict(self.config.model.get("kwargs"))
        print(f"[VisionModelManager] Encoder '{name}' su {self.device} con kwargs={kwargs}")
        return build_encoder(name, **kwargs)


    @staticmethod
    def _kwargs_to_dict(kwargs) -> dict:
        """
        Converte i kwargs del modello (eventuale nodo OmegaConf) in dict puro,
        cosi' possono essere passati come **kwargs al costruttore dell'encoder.
        """
        if kwargs is None:
            return {}
        if isinstance(kwargs, DictConfig):
            return OmegaConf.to_container(kwargs, resolve=True)
        return dict(kwargs)
