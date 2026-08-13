# tests/test_vision_retrieval.py

import argparse
from pathlib import Path

from src.vision.models.retrieval_model import VisionRetrievalPipeline
from src.vision.models.vision_model_manager import VisionModelManager
from src.vision.utils.config import load_vision_config

CONFIG_PATH = "configs/vision_retrieval.yaml"


def test_vision_retrieval():
    # Cambiare encoder/variante = cambiare model.name/model.variant nel config,
    # oppure passare override CLI (es. `model.kwargs.pooling=gem model.variant=gem`).
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

    # 1. estrai e salva gli embedding RAW dell'intero dataset (gallery = query
    #    pool). Whitening e head NON entrano qui: sono trasformazioni applicate
    #    al volo in evaluate, così lo stesso raw alimenta tutti i contributi.
    pipeline.extract_embeddings(
        data_dir=config.retrieval.data_dir,
        batch_size=config.retrieval.batch_size,
        save_path=save_dir,
    )

    # 2. indice di sanity in spazio RAW (no head, no whitening)
    pipeline.prepare_index(head=None, whiten=False)

    # 3. QUERY con la prima immagine del dataset: essendo anche in gallery,
    #    il top-1 sarà il self-match (score ~1.0)
    query_path = pipeline.image_paths[0]
    print(f"\nQuery image: {Path(query_path).name}")

    results = pipeline.query(pipeline.load_query(str(query_path)), top_k=5)

    print("\n--- Top 5 floor plan simili ---")
    for i, r in enumerate(results):
        print(f"  {i+1}. score={r['score']:.4f}  →  {r['path']}")


if __name__ == "__main__":
    test_vision_retrieval()
