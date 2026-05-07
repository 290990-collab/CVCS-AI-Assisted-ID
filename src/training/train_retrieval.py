import torch
from omegaconf import DictConfig

from src.models.retrieval_model import RetrievalEncoder


def train_retrieval(cfg: DictConfig) -> None:
    print("Starting retrieval training...")

    device = torch.device(cfg.training.device)
    model = RetrievalEncoder(
        in_dim=cfg.model.in_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    print(f"Device: {device}")
    print(f"Epochs: {cfg.training.epochs}")

    # TODO: load graph dataset / dataloader

    for epoch in range(cfg.training.epochs):
        print(f"Epoch {epoch + 1}/{cfg.training.epochs}")

        # TODO: implement contrastive/triplet training step for retrieval embeddings
        loss = 0.0

        optimizer.zero_grad()
        # TODO: backward pass when real loss exists
        print(f"Loss: {loss:.4f}")

    print("Training completed.")