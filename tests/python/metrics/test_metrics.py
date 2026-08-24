"""Tests for evaluation metrics."""

import numpy as np

from onemoreepoch.core import Tensor
from onemoreepoch.metrics import accuracy, f1, mae, mse, precision, recall


class TestRegressionMetrics:
    def test_mse(self):
        assert mse([1.0, 2.0], [3.0, 2.0]) == 2.0  # (4 + 0) / 2

    def test_mae(self):
        assert mae([1.0, 2.0], [3.0, 0.0]) == 2.0  # (2 + 2) / 2

    def test_accepts_tensors(self):
        pred = Tensor([1.0, 2.0])
        target = Tensor([1.0, 4.0])
        assert mse(pred, target) == 2.0  # (0 + 4) / 2


class TestClassificationMetrics:
    def test_accuracy_with_class_indices(self):
        assert accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75

    def test_accuracy_with_logits(self):
        logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = np.array([1, 0, 0])
        assert accuracy(logits, targets) == 2 / 3

    def test_precision_recall_f1(self):
        pred = [1, 1, 0, 0, 1]
        target = [1, 0, 0, 1, 1]
        # TP=2 (idx 0,4), FP=1 (idx 1), FN=1 (idx 3)
        assert precision(pred, target) == 2 / 3
        assert recall(pred, target) == 2 / 3
        assert f1(pred, target) == 2 / 3

    def test_precision_zero_predicted_positive(self):
        assert precision([0, 0, 0], [1, 0, 1]) == 0.0

    def test_recall_zero_actual_positive(self):
        assert recall([0, 0, 0], [0, 0, 0]) == 0.0
