from typing import Callable

from torchvision import transforms
from PIL import Image
import torch

IMAGE_SIZE = 224

# ImageNet normalisation (DINOv2 / DINOv3 default)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PAD_FILL_RGB = (255, 255, 255)


class ResizeWithPad:
    """Resize the long side to `target_size` keeping aspect ratio, pad to a square canvas."""

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
    """ResizeWithPad + ToTensor + Normalize: PIL image -> tensor [3, image_size, image_size]."""
    return transforms.Compose([
        ResizeWithPad(image_size, fill=fill),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Default pipeline (ImageNet normalisation)."""
    return compose_transform(image_size, IMAGENET_MEAN, IMAGENET_STD)


def load_image(path: str, transform: Callable | None = None) -> torch.Tensor:
    """Load an image as RGB and apply `transform` (default pipeline if None)."""
    image = Image.open(path).convert("RGB")
    transform = transform or get_transform()
    return transform(image)
