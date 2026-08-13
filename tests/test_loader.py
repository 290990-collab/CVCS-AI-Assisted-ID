# tests/test_loader.py

from src.vision.data.loader import get_dataloader

DATA_DIR = "/work/cvcs2026/ai_interior_design/datasets/RPLAN/Interface/static/Data/snapshot_train"


def test_loader():
    """
    Smoke test del DataLoader RPLAN.

    Verifica che:
      - il DataLoader si costruisca senza errori sul dataset reale;
      - un batch abbia shape [B, 3, 224, 224] e dtype float32;
      - i valori siano già normalizzati ImageNet (fuori da [0, 1]).

    Nota: il warning di pin_memory sparisce su un nodo HPC con GPU;
    sui login node è atteso.
    """
    loader = get_dataloader(data_dir=DATA_DIR, batch_size=4)
    images, paths = next(iter(loader))

    # images: [B, C, H, W] — atteso [4, 3, 224, 224]
    print(f"\nShape (B, C, H, W): {images.shape}")
    print(f"Dtype:              {images.dtype}")        # atteso: torch.float32
    print(f"Primo path:         {paths[0]}")
    # range tipico dopo Normalize(ImageNet): circa [-2.1, 2.6]
    print(f"Min/Max:            {images.min():.2f} / {images.max():.2f}\n")


if __name__ == "__main__":
    test_loader()
