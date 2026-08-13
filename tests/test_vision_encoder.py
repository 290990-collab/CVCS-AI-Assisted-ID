# tests/test_vision_encoder.py

import torch

from src.vision.models.vision_encoders import build_encoder


def test_vision_encoder():
    # encoder via registry; cambia "dinov2" in "dinov3"/"siglip2"/"radio"/"ijepa"
    # per testare gli altri (richiede il download del checkpoint).
    encoder = build_encoder("dinov2", hf_name="facebook/dinov2-base")
    encoder.eval()

    # simula un batch di 4 immagini — non servono immagini reali per questo test
    dummy_images = torch.randn(4, 3, encoder.image_size, encoder.image_size)

    with torch.no_grad():
        embeddings = encoder(dummy_images)

    print(f"Embedding dim dichiarata: {encoder.embedding_dim}")
    print(f"Shape embeddings:         {embeddings.shape}")   # atteso: [4, embedding_dim]
    print(f"Norma L2 (dovrebbe essere 1.0): {embeddings.norm(dim=1)}")


if __name__ == "__main__":
    test_vision_encoder()
