# Evaluation metrics for graph reconstruction
# Similarity, Retrieval Accuracy

def adjacency_accuracy(pred, gt):
    return (pred == gt).mean()