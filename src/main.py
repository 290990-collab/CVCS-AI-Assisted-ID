import hydra
from omegaconf import DictConfig

from src.data.preprocess import preprocess_data
from src.evaluation.evaluate import evaluate
from src.training.train_generation import train_generation
from src.training.train_retrieval import train_retrieval


@hydra.main(config_path="../configs", config_name="local", version_base=None)
def main(cfg: DictConfig):
    mode = cfg.mode

    if mode == "preprocess":
        preprocess_data(cfg)
    elif mode == "train_retrieval":
        train_retrieval(cfg)
    elif mode == "train_generation":
        train_generation(cfg)
    elif mode == "evaluate":
        evaluate(cfg)
    else:
        valid_modes = ["preprocess", "train_retrieval", "train_generation", "evaluate"]
        raise ValueError(f"Unknown mode: {mode}. Choose one of {valid_modes}.")


if __name__ == "__main__":
    main()