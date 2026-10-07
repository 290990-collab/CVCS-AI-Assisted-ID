
from src.vision.data.loader import get_dataloader

DATA_DIR = "/work/cvcs2026/ai_interior_design/datasets/RPLAN/Interface/static/Data/snapshot_train"


def test_loader():
    """RPLAN DataLoader smoke test: builds on the real dataset, batch [B, 3, 224, 224] float32, ImageNet-normalised."""
    loader = get_dataloader(data_dir=DATA_DIR, batch_size=4)
    images, paths = next(iter(loader))

    # images: [B, C, H, W], expected [4, 3, 224, 224]
    print(f"\nShape (B, C, H, W): {images.shape}")
    print(f"Dtype:              {images.dtype}")        # expected: torch.float32
    print(f"Primo path:         {paths[0]}")
    # typical range after ImageNet Normalize: ~[-2.1, 2.6]
    print(f"Min/Max:            {images.min():.2f} / {images.max():.2f}\n")


if __name__ == "__main__":
    test_loader()
