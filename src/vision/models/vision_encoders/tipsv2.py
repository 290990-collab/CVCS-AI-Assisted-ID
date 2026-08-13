# src/vision/models/vision_encoders/tipsv2.py

from typing import Callable

import torch
import torch.nn as nn
from transformers import AutoModel

from src.vision.data.preprocess import compose_transform
from src.vision.models.vision_encoders.base import BaseVisionEncoder, pool_global


class TIPSv2Encoder(BaseVisionEncoder):
    """
    Encoder basato su TIPSv2 (Google, "Text-Image Pre-training with Spatial
    awareness"): vision-language contrastivo come SigLIP2, ma con un obiettivo
    aggiuntivo di allineamento PATCH-testo che rende i token spaziali molto piu'
    informativi (il modello e' pensato anche per la segmentazione zero-shot).
    Variante B/14: ViT-B, patch 14, embed_dim 768.

    Particolarita' rispetto agli altri encoder:
      1. Si carica con `trust_remote_code=True` (codice di modeling custom sul
         repo HF: `modeling_tips.py` + `image_encoder.py` + `text_encoder.py`).
      2. L'input va dato in [0, 1] SENZA normalizzazione ImageNet (come RADIO):
         mean=0, std=1 nel transform.
      3. Il forward del backbone ritorna una TERNA
         (cls_token [B,1,D], register_tokens [B,R,D], patch_tokens [B,N,D]);
         i register token (R=1) sono scartati, non sono feature d'immagine.
      4. Risoluzione nativa 448 (32x32 = 1024 patch): 4x i token di un 224px.

    Args:
        hf_name:    checkpoint HuggingFace (b14 | l14 | so400m14 | g14).
        image_size: lato input in pixel, multiplo di patch_size (14). Con un
                    valore != 448 le positional embedding vengono interpolate
                    dal modello: legale, ma e' un regime diverso dal pretrain.
        pooling:    "natural" (token CLS) | "mean" | "gem" sui patch token.
        gem_p:      esponente del GeM (usato solo se pooling="gem").
        extraction_layer: indice del layer da cui leggere i token (-1 = ultimo).

    Note:
        Il testo non viene mai usato qui (retrieval immagine-immagine), ma il
        checkpoint contiene anche il text encoder: caricarlo richiede comunque
        il pacchetto `sentencepiece` (importato da `text_encoder.py`).
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

        # il patch embedding richiede un lato multiplo del patch size (14)
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

        # TIPSv2 lavora su pixel in [0, 1]: nessuna normalizzazione ImageNet
        # (`do_normalize: false` nel processor_config del checkpoint).
        self.image_size = image_size
        self.mean = [0.0, 0.0, 0.0]
        self.std  = [1.0, 1.0, 1.0]

        # pooling + layer di estrazione
        self.pooling          = pooling
        self.gem_p            = gem_p
        self.extraction_layer = extraction_layer

        # D = larghezza del ViT (768 per la B/14), invariante al pooling
        self._embedding_dim = self.model.config.embed_dim
        print(f"[TIPSv2Encoder] Pronto: {self._embedding_dim}d, pooling={pooling}, "
              f"layer={extraction_layer}, input {self.image_size}px")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W] in [0, 1] (nessuna normalizzazione ImageNet)
        #
        # Si chiama il backbone `vision_encoder` invece del wrapper pubblico
        # `model.encode_image()` per due motivi: quest'ultimo e' decorato
        # @torch.no_grad() (impedirebbe di allenare qualsiasi cosa a monte) e non
        # espone i layer intermedi. La terna ritornata e' la stessa.
        backbone = self.model.vision_encoder

        if self.extraction_layer == -1:
            # ultimo layer, POST-layernorm finale: la baseline degli altri encoder
            cls_token, _registers, patches = backbone(images)   # [B,1,D] [B,R,D] [B,N,D]
            natural = cls_token[:, 0, :]                        # [B, D]
        else:
            # layer intermedio, PRE-layernorm (norm=False): stessa semantica di
            # `hidden_states[layer]` in HF. L'indice negativo va convertito in
            # posizione del blocco perche' l'API accetta solo indici assoluti.
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
