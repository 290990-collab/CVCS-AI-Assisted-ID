# src/vision/models/vision_encoders/pe_core.py

from typing import Callable

import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config

from src.vision.data.preprocess import compose_transform
from src.vision.models.vision_encoders.base import BaseVisionEncoder, pool_global


class PECoreEncoder(BaseVisionEncoder):
    """
    Encoder basato su PE-Core (Perception Encoder, Meta — arXiv 2504.13181):
    vision-language contrastivo in stile CLIP, ma con la tesi esplicita che
    "le migliori feature visive NON stanno all'uscita della rete" — sono nei
    layer INTERMEDI, mentre l'ultimo layer e' specializzato sull'allineamento
    col testo. Per questo `extraction_layer` qui non e' un dettaglio: e'
    l'iperparametro che il paper stesso indica come decisivo.
    Variante B/16 a 224px: ViT-B, patch 16, width 768, 12 layer, 196 patch token.

    Particolarita' rispetto agli altri encoder:
      1. Si carica da **timm** (`vit_pe_core_base_patch16_224.fb`), non da
         `transformers`: il repo HF `facebook/PE-Core-B16-224` distribuisce un
         `.pt` grezzo che richiederebbe il pacchetto `perception_models` di
         Meta. timm ospita gli stessi pesi (`timm/vit_pe_core_base_patch16_224.fb`,
         apache-2.0, non gated) dietro una classe `Eva` gia' installata.
      2. Normalizzazione **mean=std=0.5** (non ImageNet), letta dal pretrained_cfg.
      3. Il pooling nativo NON e' il token CLS ma un blocco di **attention
         pooling** (8 teste) seguito dalla proiezione contrastiva CLIP: l'output
         nativo ha percio' **1024** dimensioni, mentre i patch token ne hanno 768.
         Di conseguenza `embedding_dim` DIPENDE dal pooling (1024 con "natural",
         768 con "mean"/"gem") e viene dedotto con un forward di prova, come in
         RADIO. Le tre varianti restano confrontabili perche' il whitening a
         valle normalizza la scala, ma non hanno la stessa larghezza.
      4. Risoluzione **bloccata a 224**: il checkpoint dichiara
         `fixed_input_size=True` e le RoPE sono agganciate a una griglia 14x14,
         quindi timm fallisce con un assert su qualunque altro lato.

    Args:
        timm_name:  nome del modello timm (i pesi arrivano da HuggingFace).
        image_size: lato input in pixel. DEVE valere 224 per questo checkpoint
                    (vedi punto 4): un valore diverso e' un errore, non uno sweep.
        pooling:    "natural" (attention pool + proiezione CLIP, 1024d) |
                    "mean" | "gem" sui patch token (768d).
        gem_p:      esponente del GeM (usato solo se pooling="gem").
        extraction_layer: indice del layer da cui leggere i token (-1 = ultimo).

    Note:
        Con `extraction_layer != -1` la testa di attention pooling non e'
        applicabile (opera solo sull'uscita dell'ultimo blocco): "natural"
        ripiega allora sul token CLS di quel layer, come in DINOv2, e
        l'embedding scende a 768d. E' la stessa asimmetria di SigLIP2.
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

        # num_classes NON va azzerato: la "testa" del modello (Linear 768->1024)
        # e' la proiezione contrastiva CLIP presente nel checkpoint, non un
        # classificatore ImageNet da buttare.
        self.model = timm.create_model(timm_name, pretrained=True)
        self._freeze()

        # risoluzione fissa: meglio fallire qui che dopo aver caricato il dataset
        native_size = self.model.pretrained_cfg["input_size"][-1]
        if image_size != native_size:
            raise ValueError(
                f"image_size={image_size} non supportato da {timm_name}: il "
                f"checkpoint e' a risoluzione fissa {native_size}px (RoPE su "
                f"griglia fissa). Per cambiare risoluzione serve un'altra "
                f"variante PE-Core (es. large_patch14_336)."
            )
        self.image_size = image_size

        # mean/std dal pretrained_cfg del checkpoint: 0.5/0.5, NON ImageNet
        data_cfg  = resolve_data_config({}, model=self.model)
        self.mean = list(data_cfg["mean"])
        self.std  = list(data_cfg["std"])

        # pooling + layer di estrazione
        self.pooling          = pooling
        self.gem_p            = gem_p
        self.extraction_layer = extraction_layer

        # D dedotta con un forward di prova (su CPU) perche' dipende dal pooling:
        # l'attention pool proietta a 1024, i patch token restano a 768.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            self._embedding_dim = self.forward(dummy).shape[-1]

        print(f"[PECoreEncoder] Pronto: {self._embedding_dim}d, pooling={pooling}, "
              f"layer={extraction_layer}, input {self.image_size}px")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, 224, 224] normalizzate con mean=std=0.5
        if self.extraction_layer == -1:
            if self.pooling == "natural":
                # pooling nativo: attention pool (8 teste) + proiezione CLIP.
                # E' il forward completo del modello timm -> [B, 1024].
                pooled = self.model(images)
            else:
                # ultimo layer, POST-layernorm finale: [B, 1+N, D] (1 prefix token).
                tokens  = self.model.forward_features(images)
                patches = tokens[:, self.model.num_prefix_tokens:, :]   # [B, N, D]
                pooled  = pool_global(None, patches, self.pooling, self.gem_p)
        else:
            # layer intermedio, PRE-layernorm (norm=False): stessa semantica di
            # hidden_states[layer] in HF. L'indice negativo va convertito in
            # posizione assoluta del blocco, come in TIPSv2.
            n_blocks = len(self.model.blocks)
            layer    = (n_blocks + self.extraction_layer
                        if self.extraction_layer < 0 else self.extraction_layer)
            (patches, prefix), = self.model.forward_intermediates(
                images, indices=[layer], return_prefix_tokens=True,
                norm=False, output_fmt="NLC", intermediates_only=True,
            )                                        # [B, N, D], [B, 1, D]
            natural = prefix[:, 0, :]                # token CLS del layer scelto
            pooled  = pool_global(natural, patches, self.pooling, self.gem_p)

        return nn.functional.normalize(pooled, p=2, dim=1)

    def build_transform(self) -> Callable:
        return compose_transform(self.image_size, self.mean, self.std)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
