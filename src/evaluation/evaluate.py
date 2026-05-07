from omegaconf import DictConfig

from src.evaluation.metrics import adjacency_accuracy, recall_at_k


def evaluate(cfg: DictConfig) -> None:
    print("Starting evaluation...")
    print(f"Processed data root: {cfg.data.processed_root}")

    # TODO: load trained checkpoints and evaluation set
    # Placeholder values to keep the pipeline executable.
    adj_acc = adjacency_accuracy([1, 0, 1], [1, 1, 1])
    r_at_5 = recall_at_k([0, 2, 5, 7, 10], gt_index=2, k=5)
    print(f"Adjacency accuracy (placeholder): {adj_acc:.4f}")
    print(f"Recall@5 (placeholder): {r_at_5:.4f}")

    print("Evaluation completed.")