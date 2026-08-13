# src/vision/data/preprocess.py

from typing import Callable

from torchvision import transforms
from PIL import Image
import torch

# Ogni encoder dichiara esplicitamente il proprio image_size (default 224)
IMAGE_SIZE = 224

# Valori di normalizzazione ImageNet (default per DINOv2 / DINOv3)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PAD_FILL_RGB = (255, 255, 255)


# Piante --> forme diverse
# ResizeWithPad ridimensiona le immagini mantenendo le proporzioni e ed evitare distorsione.
class ResizeWithPad:
    """
    Resize: PRESERVA l'aspect ratio; padding a quadrato.

      Ridimensionamento lato lungo --> 'target_size'
      Padding lato corto

      Output --> 'target_size x target_size'
    """

    def __init__(self, target_size: int, fill: tuple = PAD_FILL_RGB):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", (self.target_size, self.target_size), self.fill)
        offset = ((self.target_size - new_w) // 2, (self.target_size - new_h) // 2)
        canvas.paste(img, offset)
        return canvas


def compose_transform(
    image_size: int,
    mean: list,
    std: list,
    fill: tuple = PAD_FILL_RGB,
) -> transforms.Compose:
    """
    Costruisce il preprocessing per un encoder: ResizeWithPad + ToTensor + Normalize.
    ResizeWithPad comune, normalizzazione e risoluzione parametriche (specifiche).

    Args:
        image_size: lato dell'immagine quadrata in output.
        mean, std:  costanti di normalizzazione per canale.
        fill:       colore di padding.
        
    Returns:
        transforms.Compose (callable: PIL.Image -> Tensor [3, image_size, image_size]).
    """
    return transforms.Compose([
        ResizeWithPad(image_size, fill=fill),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Pipeline di trasformazione di DEFAULT (normalizzazione ImageNet, fallback).
    """
    return compose_transform(image_size, IMAGENET_MEAN, IMAGENET_STD)


def load_image(path: str, transform: Callable | None = None) -> torch.Tensor:
    """
    Carica una singola immagine da disco (PIL -> RGB) e applica un transform.

    Args:
        path:      percorso al file PNG.
        transform: callable PIL.Image -> Tensor.

    Returns:
        tensore di shape [3, image_size, image_size].
    """
    image = Image.open(path).convert("RGB")
    transform = transform or get_transform()
    return transform(image)
