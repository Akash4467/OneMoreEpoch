import numpy as np

from onemoreepoch.core import Tensor
from onemoreepoch.metrics import accuracy, f1, mae, mse, precision, recall


# Tests regression metrics
class TestRegressionMetrics:
    # Checks mse computes the mean squared error
    def test_mse(self):
        assert mse([1.0, 2.0], [3.0, 2.0]) == 2.0

    # Checks mae computes the mean absolute error
    def test_mae(self):
        assert mae([1.0, 2.0], [3.0, 0.0]) == 2.0

    # Checks metrics accept Tensor inputs directly
    def test_accepts_tensors(self):
        pred = Tensor([1.0, 2.0])
        target = Tensor([1.0, 4.0])
        assert mse(pred, target) == 2.0


# Tests classification metrics
class TestClassificationMetrics:
    # Checks accuracy with pre-computed class indices
    def test_accuracy_with_class_indices(self):
        assert accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75

    # Checks accuracy argmaxes per-class logits before comparing
    def test_accuracy_with_logits(self):
        logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = np.array([1, 0, 0])
        assert accuracy(logits, targets) == 2 / 3

    # Checks precision/recall/f1 on a mixed set of predictions
    def test_precision_recall_f1(self):
        pred = [1, 1, 0, 0, 1]
        target = [1, 0, 0, 1, 1]
        assert precision(pred, target) == 2 / 3
        assert recall(pred, target) == 2 / 3
        assert f1(pred, target) == 2 / 3

    # Checks precision is 0 when nothing is predicted positive
    def test_precision_zero_predicted_positive(self):
        assert precision([0, 0, 0], [1, 0, 1]) == 0.0

    # Checks recall is 0 when nothing is actually positive
    def test_recall_zero_actual_positive(self):
        assert recall([0, 0, 0], [0, 0, 0]) == 0.0
