from typing import Any

import numpy as np


# Converts a Tensor or raw array-like into a numpy array
def _as_array(x: Any) -> np.ndarray:
    return np.asarray(x.data if hasattr(x, "data") else x)


# Returns the fraction of predicted classes matching targets
def accuracy(predictions: Any, targets: Any) -> float:
    pred = _as_array(predictions)
    target = _as_array(targets)
    if pred.ndim > target.ndim:
        pred = pred.argmax(axis=-1)
    return float((pred == target).mean())


# Returns the mean squared error
def mse(predictions: Any, targets: Any) -> float:
    pred, target = _as_array(predictions), _as_array(targets)
    return float(np.mean((pred - target) ** 2))


# Returns the mean absolute error
def mae(predictions: Any, targets: Any) -> float:
    pred, target = _as_array(predictions), _as_array(targets)
    return float(np.mean(np.abs(pred - target)))


# Returns the precision for the given positive label
def precision(predictions: Any, targets: Any, *, positive_label: int = 1) -> float:
    pred, target = _as_array(predictions), _as_array(targets)
    predicted_positive = pred == positive_label
    true_positive = predicted_positive & (target == positive_label)
    denom = predicted_positive.sum()
    return float(true_positive.sum() / denom) if denom else 0.0


# Returns the recall for the given positive label
def recall(predictions: Any, targets: Any, *, positive_label: int = 1) -> float:
    pred, target = _as_array(predictions), _as_array(targets)
    actual_positive = target == positive_label
    true_positive = actual_positive & (pred == positive_label)
    denom = actual_positive.sum()
    return float(true_positive.sum() / denom) if denom else 0.0


# Returns the F1 score (harmonic mean of precision and recall) for the given positive label
def f1(predictions: Any, targets: Any, *, positive_label: int = 1) -> float:
    p = precision(predictions, targets, positive_label=positive_label)
    r = recall(predictions, targets, positive_label=positive_label)
    return 2 * p * r / (p + r) if (p + r) else 0.0


__all__ = ["accuracy", "f1", "mae", "mse", "precision", "recall"]
