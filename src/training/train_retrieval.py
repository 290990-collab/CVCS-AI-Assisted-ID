import torch
from src.utils.config import load_config


def train_retrieval(config_path):
    config = load_config(config_path)

    print("Starting retrieval training...")

    # TODO: load dataset
    # TODO: init model
    # TODO: init optimizer

    for epoch in range(config["training"]["epochs"]):
        print(f"Epoch {epoch}")

        # TODO: training step
        loss = 0.0

        print(f"Loss: {loss}")

    print("Training completed.")