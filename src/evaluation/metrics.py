def adjacency_accuracy(pred, gt):
    return (pred == gt).mean()