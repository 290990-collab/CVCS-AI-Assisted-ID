from omegaconf import DictConfig


def train_generation(cfg: DictConfig) -> None:
    print("Starting generation training...")
    print(f"Epochs: {cfg.training.epochs}")
    print(f"Device: {cfg.training.device}")

    # TODO: initialize generation model (diffusion or autoregressive)
    # TODO: load conditioning constraints (room types, counts, adjacency)
    # TODO: training loop with constraint-aware objective

    for epoch in range(cfg.training.epochs):
        print(f"Epoch {epoch + 1}/{cfg.training.epochs}")
        loss = 0.0
        print(f"Loss: {loss:.4f}")

    print("Generation training completed.")
