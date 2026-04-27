# Controller
import argparse

from src.training import train_retrieval
from src.training import train_generation
from src.data import preprocess
from src.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["preprocess", "train_retrieval", "train_generation", "evaluate"])
    parser.add_argument("--config", type=str, default="configs/local.yaml")

    args = parser.parse_args()

    if args.mode == "preprocess":
        preprocess(args.config)

    elif args.mode == "train_retrieval":
        train_retrieval(args.config)

    elif args.mode == "train_generation":
        train_generation(args.config)

    elif args.mode == "evaluate":
        evaluate(args.config)


if __name__ == "__main__":
    main()