def adjacency_accuracy(pred: list[int], gt: list[int]) -> float:
    if len(pred) != len(gt) or len(gt) == 0:
        return 0.0
    correct = sum(int(p == g) for p, g in zip(pred, gt))
    return correct / len(gt)


def recall_at_k(retrieved_indices: list[int], gt_index: int, k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved_indices[:k]
    return 1.0 if gt_index in top_k else 0.0