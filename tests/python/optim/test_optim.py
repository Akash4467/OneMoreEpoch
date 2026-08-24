"""Tests for optimizers + an end-to-end training convergence check."""

import numpy as np
import pytest

from onemoreepoch import nn
from onemoreepoch.core.backend import get_backend
from onemoreepoch.core import Parameter, Tensor
from onemoreepoch.exceptions import OptimizerError
from onemoreepoch.optim import SGD, AdaGrad, Adam, AdamW, Optimizer, RMSProp


class TestOptimizerBase:
    def test_rejects_empty_parameters(self):
        with pytest.raises(ValueError):
            SGD([], lr=0.1)

    def test_rejects_non_positive_lr(self):
        with pytest.raises(ValueError):
            SGD([Parameter([1.0])], lr=0.0)

    def test_errors_are_optimizer_errors(self):
        from onemoreepoch.exceptions import OptimizerError

        with pytest.raises(OptimizerError):
            SGD([], lr=0.1)
        with pytest.raises(OptimizerError):
            SGD([Parameter([1.0])], lr=-0.5)

    def test_base_update_is_abstract(self):
        opt = Optimizer([Parameter([1.0])], lr=0.1)
        opt.parameters[0].grad = np.array([1.0])
        with pytest.raises(NotImplementedError):
            opt.step()

    def test_zero_grad(self):
        param = Parameter([1.0])
        param.grad = np.array([5.0])
        SGD([param], lr=0.1).zero_grad()
        assert param.grad is None

    def test_step_skips_params_without_grad(self):
        param = Parameter([1.0])
        SGD([param], lr=0.1).step()  # no grad — must not raise or move
        np.testing.assert_array_equal(param.data, [1.0])


class TestSGD:
    def test_vanilla_update(self):
        param = Parameter([10.0])
        param.grad = np.array([2.0])
        SGD([param], lr=0.5).step()
        np.testing.assert_allclose(param.data, [9.0])

    def test_momentum_accumulates(self):
        param = Parameter([0.0])
        opt = SGD([param], lr=1.0, momentum=0.5)
        param.grad = np.array([1.0])
        opt.step()  # v=1, param=-1
        param.grad = np.array([1.0])
        opt.step()  # v=1.5, param=-2.5
        np.testing.assert_allclose(param.data, [-2.5])

    def test_invalid_momentum(self):
        with pytest.raises(ValueError):
            SGD([Parameter([1.0])], lr=0.1, momentum=1.0)


@pytest.mark.parametrize(
    "opt_cls,kwargs",
    [
        (Adam, {"lr": 0.1}),
        (AdamW, {"lr": 0.1, "weight_decay": 0.0}),
        (RMSProp, {"lr": 0.05}),
        (AdaGrad, {"lr": 0.5}),
    ],
)
class TestAdaptiveOptimizersConverge:
    def test_minimizes_quadratic(self, opt_cls, kwargs):
        param = Parameter(np.array([5.0, -3.0]))
        opt = opt_cls([param], **kwargs)
        for _ in range(500):
            param.grad = 2 * param.data  # grad of sum(w^2)
            opt.step()
            opt.zero_grad()
        np.testing.assert_allclose(param.data, [0.0, 0.0], atol=0.2)


class TestAdaptiveOptimizerValidation:
    def test_adam_rejects_bad_beta(self):
        with pytest.raises(OptimizerError):
            Adam([Parameter([1.0])], betas=(1.5, 0.999))

    def test_adamw_rejects_negative_weight_decay(self):
        with pytest.raises(OptimizerError):
            AdamW([Parameter([1.0])], weight_decay=-0.1)

    def test_rmsprop_rejects_bad_alpha(self):
        with pytest.raises(OptimizerError):
            RMSProp([Parameter([1.0])], alpha=1.5)

    def test_adagrad_rejects_bad_eps(self):
        with pytest.raises(OptimizerError):
            AdaGrad([Parameter([1.0])], eps=0.0)


class TestEndToEnd:
    def test_linear_regression_converges(self):
        """Full pipeline: model → loss → backward → step → convergence."""
        get_backend().seed(0)
        true_w = np.array([[2.0], [-3.0]])
        true_b = 0.5
        x_data = np.random.default_rng(0).standard_normal((64, 2))
        y_data = x_data @ true_w + true_b

        model = nn.Linear(2, 1)
        criterion = nn.MSELoss()
        optimizer = SGD(model.parameters(), lr=0.1)

        loss_value = None
        for _ in range(200):
            optimizer.zero_grad()
            loss = criterion(model(Tensor(x_data)), Tensor(y_data))
            loss.backward()
            optimizer.step()
            loss_value = loss.item()

        assert loss_value < 1e-6
        np.testing.assert_allclose(model.weight.data, true_w, atol=1e-3)
        np.testing.assert_allclose(model.bias.data, [true_b], atol=1e-3)

    def test_mlp_learns_xor(self):
        """Nonlinear problem — proves gradients flow through activations."""
        get_backend().seed(3)
        x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = Tensor([[0.0], [1.0], [1.0], [0.0]])

        model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        criterion = nn.MSELoss()
        optimizer = SGD(model.parameters(), lr=0.5, momentum=0.9)

        for _ in range(500):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        predictions = model(x).data
        np.testing.assert_allclose(predictions, y.data, atol=0.1)
