"""Evaluation metrics — operate on raw arrays or Tensors, outside the autograd graph."""

from typing import Any

import numpy as np


def _as_array(x: Any) -> np.ndarray:
    return np.asarray(x.data if hasattr(x, "data") else x)


def accuracy(predictions: Any, targets: Any) -> float:
    """Fraction of predicted classes matching targets.

    ``predictions`` may be class indices already, or per-class
    scores/probabilities (argmax'd over the last axis).
    """
    pred = _as_array(predictions)
    target = _as_array(targets)
    if pred.ndim > target.ndim:
        pred = pred.argmax(axis=-1)
    return float((pred == target).mean())


def mse(predictions: Any, targets: Any) -> float:
    pred, target = _as_array(predictions), _as_array(targets)
    return float(np.mean((pred - target) ** 2))


def mae(predictions: Any, targets: Any) -> float:
    pred, target = _as_array(predictions), _as_array(targets)
    return float(np.mean(np.abs(pred - target)))


def precision(predictions: Any, targets: Any, *, positive_label: int = 1) -> float:
    pred, target = _as_array(predictions), _as_array(targets)
    predicted_positive = pred == positive_label
    true_positive = predicted_positive & (target == positive_label)
    denom = predicted_positive.sum()
    return float(true_positive.sum() / denom) if denom else 0.0


def recall(predictions: Any, targets: Any, *, positive_label: int = 1) -> float:
    pred, target = _as_array(predictions), _as_array(targets)
    actual_positive = target == positive_label
    true_positive = actual_positive & (pred == positive_label)
    denom = actual_positive.sum()
    return float(true_positive.sum() / denom) if denom else 0.0


def f1(predictions: Any, targets: Any, *, positive_label: int = 1) -> float:
    p = precision(predictions, targets, positive_label=positive_label)
    r = recall(predictions, targets, positive_label=positive_label)
    return 2 * p * r / (p + r) if (p + r) else 0.0


__all__ = ["accuracy", "f1", "mae", "mse", "precision", "recall"]
