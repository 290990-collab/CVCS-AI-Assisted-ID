# Controller

import hydra
from omegaconf import DictConfig

from src.data import preprocess
from src.training import train_retrieval, train_generation
from src.evaluation import evaluate


@hydra.main(config_path="configs", config_name="local", version_base=None)
def main(cfg: DictConfig):

    mode = cfg.mode  # passato da CLI: python main.py mode=train_retrieval

    if mode == "preprocess":
        preprocess(cfg)

    elif mode == "train_retrieval":
        train_retrieval(cfg)

    elif mode == "train_generation":
        train_generation(cfg)

    elif mode == "evaluate":
        evaluate(cfg)

    else:
        raise ValueError(f"Modalità non riconosciuta: {mode}. "
                         f"Scegli tra: preprocess, train_retrieval, train_generation, evaluate")


if __name__ == "__main__":
    main()