
import torch

from src.vision.models.vision_encoders import build_encoder


def test_vision_encoder():
    # encoder via registry; swap "dinov2" for "dinov3"/"siglip2"/"radio"/"ijepa"
    # to test the others (needs checkpoint download).
    encoder = build_encoder("dinov2", hf_name="facebook/dinov2-base")
    encoder.eval()

    # fake batch of 4 images
    dummy_images = torch.randn(4, 3, encoder.image_size, encoder.image_size)

    with torch.no_grad():
        embeddings = encoder(dummy_images)

    print(f"Embedding dim dichiarata: {encoder.embedding_dim}")
    print(f"Shape embeddings:         {embeddings.shape}")   # expected: [4, embedding_dim]
    print(f"Norma L2 (dovrebbe essere 1.0): {embeddings.norm(dim=1)}")


if __name__ == "__main__":
    test_vision_encoder()
