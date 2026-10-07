import argparse
from pathlib import Path

from src.vision.models.retrieval_model import VisionRetrievalPipeline
from src.vision.models.vision_model_manager import VisionModelManager
from src.vision.utils.config import load_vision_config

CONFIG_PATH = "configs/vision_retrieval.yaml"


def run_retrieval():
    # encoder/variant: set model.name/model.variant or CLI overrides (e.g. `model.kwargs.pooling=gem`)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    args, overrides = parser.parse_known_args()

    config  = load_vision_config(args.config, overrides)
    manager = VisionModelManager(config)
    save_dir = config.retrieval.save_dir
    print(f"[Test] encoder={config.model.name} variante={config.model.variant} "
          f"device={manager.device}")

    pipeline = VisionRetrievalPipeline(
        encoder=manager.encoder,
        transform=manager.transform,
        device=manager.device,
    )

    # raw embeddings of the whole dataset; whitening/head are applied on the fly in evaluate
    pipeline.extract_embeddings(
        data_dir=config.retrieval.data_dir,
        batch_size=config.retrieval.batch_size,
        save_path=save_dir,
    )

    # sanity index in raw space
    pipeline.prepare_index(head=None, whiten=False)

    # first image as query: top-1 is the self-match (score ~1.0)
    query_path = pipeline.image_paths[0]
    print(f"\nQuery image: {Path(query_path).name}")

    results = pipeline.query(pipeline.load_query(str(query_path)), top_k=5)

    print("\n--- Top 5 floor plan simili ---")
    for i, r in enumerate(results):
        print(f"  {i+1}. score={r['score']:.4f}  →  {r['path']}")


if __name__ == "__main__":
    run_retrieval()
