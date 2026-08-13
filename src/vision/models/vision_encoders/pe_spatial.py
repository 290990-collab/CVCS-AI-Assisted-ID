# src/vision/models/vision_encoders/pe_spatial.py

from typing import Callable

import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config

from src.vision.data.preprocess import compose_transform
from src.vision.models.vision_encoders.base import BaseVisionEncoder, pool_global


class PESpatialEncoder(BaseVisionEncoder):
    """
    Encoder basato su PE-Spatial (Perception Encoder, Meta — arXiv 2504.13181),
    variante B/16-512.

    E' il ramo "spatial" della stessa famiglia di PE-Core: parte dallo stesso
    backbone contrastivo vision-language e lo ri-allinea con un obiettivo
    DENSO (distillazione spaziale + self-distillation), in modo che a contare
    non sia piu' solo il vettore globale ma la qualita' di OGNI patch token.
    E' esattamente il regime che interessa al retrieval di planimetrie, dove
    l'informazione utile e' la disposizione delle stanze nello spazio, non
    l'etichetta semantica dell'immagine.

    Coppia controllata con `pecore`: stesso ViT-B/16, stessa famiglia, stessa
    normalizzazione (mean=std=0.5), stesso caricamento via timm. L'unica cosa
    che cambia e' il training -> confronto quasi appaiato fra feature
    vision-language e feature spatially-aligned.

    Particolarita' rispetto a PE-Core:
      1. **Niente testa contrastiva**: il checkpoint ha `num_classes=0` e
         `head=Identity`, quindi NON esiste l'attention pool + proiezione CLIP
         di PE-Core. L'embedding resta a **768d per tutti e tre i pooling**
         (in PE-Core "natural" saliva a 1024d).
      2. **`model(images)` NON e' il pooling naturale**: il `global_pool` del
         checkpoint e' "avg", quindi il forward completo restituisce la MEDIA
         dei patch token (verificato: cos = 1.0000 con la media esplicita).
         Usarlo per "natural" farebbe collassare natural == mean e sprecherebbe
         una colonna della griglia. Qui "natural" legge il **token CLS**
         (`num_prefix_tokens = 1`), come in DINOv2/DINOv3.
      3. **La risoluzione NON e' bloccata**, nonostante `fixed_input_size=True`
         nel pretrained_cfg: quel flag e' indicativo e timm ricampiona il
         `pos_embed` al caricamento (1025 -> 197 token passando da 512 a 224).
         E' la differenza pratica con PE-Core, che invece e' davvero fisso a
         224. Qui il default e' **224** per allineare PE-Spatial al resto della
         griglia; 512 e' la risoluzione nativa del checkpoint e resta
         disponibile come variante di uno sweep.

    Args:
        timm_name:  nome del modello timm (i pesi arrivano da HuggingFace).
        image_size: lato input in pixel, multiplo del patch size (16).
                    224 = allineato alla griglia; 512 = nativo del checkpoint.
        pooling:    "natural" (token CLS) | "mean" | "gem" sui patch token.
                    Tutti e tre restituiscono 768d.
        gem_p:      esponente del GeM (usato solo se pooling="gem").
        extraction_layer: indice del layer da cui leggere i token (-1 = ultimo).
                    La tesi di PE ("le feature migliori non sono all'ultimo
                    layer") nasce sul ramo Core, dove l'ultimo layer e'
                    specializzato sul testo; su Spatial il fine-tuning denso
                    lavora proprio sull'uscita, quindi qui l'ablation e' meno
                    scontata ma resta da misurare.

    Note:
        Il repo HF `facebook/PE-Spatial-B16-512` pubblica un `.pt` grezzo
        utilizzabile solo col pacchetto `perception_models` di Meta. Gli stessi
        pesi stanno su `timm/vit_pe_spatial_base_patch16_512.fb` (apache-2.0,
        non gated) dietro la classe `Eva` gia' installata: e' la via usata qui,
        identica a quella di PE-Core.
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

        # La risoluzione va passata a create_model: e' li' che timm costruisce
        # la griglia di patch e ricampiona il pos_embed del checkpoint. Deve
        # essere un multiplo del patch size, altrimenti la patch embedding
        # fallisce con un assert poco leggibile a forward time.
        patch = timm.get_pretrained_cfg(timm_name).to_dict().get("input_size")[-1]
        if image_size % 16 != 0:
            raise ValueError(
                f"image_size={image_size} non valido per {timm_name}: deve "
                f"essere un multiplo del patch size (16). Nativa del "
                f"checkpoint: {patch}px; la griglia del progetto usa 224."
            )
        self.image_size = image_size

        # num_classes=0 nel checkpoint -> head=Identity, nessuna proiezione da
        # preservare (a differenza di PE-Core, dove la head E' la proiezione CLIP).
        self.model = timm.create_model(
            timm_name, pretrained=True, img_size=image_size
        )
        self._freeze()

        # mean/std dal pretrained_cfg del checkpoint: 0.5/0.5, NON ImageNet
        data_cfg  = resolve_data_config({}, model=self.model)
        self.mean = list(data_cfg["mean"])
        self.std  = list(data_cfg["std"])

        # pooling + layer di estrazione
        self.pooling          = pooling
        self.gem_p            = gem_p
        self.extraction_layer = extraction_layer

        # D dedotta con un forward di prova (su CPU), come in PE-Core/RADIO:
        # qui vale 768 per tutti i pooling, ma dedurla evita di hardcodare una
        # costante che cambierebbe silenziosamente con un'altra variante PE.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            self._embedding_dim = self.forward(dummy).shape[-1]

        print(f"[PESpatialEncoder] Pronto: {self._embedding_dim}d, pooling={pooling}, "
              f"layer={extraction_layer}, input {self.image_size}px")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, S, S] normalizzate con mean=std=0.5
        n_prefix = self.model.num_prefix_tokens

        if self.extraction_layer == -1:
            # ultimo layer, POST-layernorm finale: [B, 1+N, D] (1 prefix token).
            # NON si usa self.model(images): quello e' global_pool="avg", cioe'
            # gia' la media dei patch (vedi docstring, punto 2).
            tokens  = self.model.forward_features(images)
            natural = tokens[:, 0, :]                      # token CLS
            patches = tokens[:, n_prefix:, :]              # [B, N, D]
        else:
            # layer intermedio, PRE-layernorm (norm=False): stessa semantica di
            # hidden_states[layer] in HF. L'indice negativo va convertito in
            # posizione assoluta del blocco, come in PE-Core/TIPSv2.
            n_blocks = len(self.model.blocks)
            layer    = (n_blocks + self.extraction_layer
                        if self.extraction_layer < 0 else self.extraction_layer)
            (patches, prefix), = self.model.forward_intermediates(
                images, indices=[layer], return_prefix_tokens=True,
                norm=False, output_fmt="NLC", intermediates_only=True,
            )                                              # [B, N, D], [B, 1, D]
            natural = prefix[:, 0, :]                      # token CLS del layer scelto

        pooled = pool_global(natural, patches, self.pooling, self.gem_p)
        return nn.functional.normalize(pooled, p=2, dim=1)

    def build_transform(self) -> Callable:
        return compose_transform(self.image_size, self.mean, self.std)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
