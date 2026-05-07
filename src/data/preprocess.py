from pathlib import Path

from omegaconf import DictConfig


def preprocess_data(cfg: DictConfig) -> None:
    raw_root = Path(cfg.data.raw_root)
    processed_root = Path(cfg.data.processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    print("Starting preprocessing...")
    print(f"Raw data root: {raw_root}")
    print(f"Processed data root: {processed_root}")

    # TODO: load raw CAD files from raw_root
    # TODO: convert each sample to graph with src.data.graph_builder.build_graph
    # TODO: serialize processed dataset to processed_root

    print("Preprocessing completed.")