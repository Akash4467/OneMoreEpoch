"""Tests for autograd: gradient correctness per op + engine behavior.

Analytic gradients are cross-checked against central-difference
numerical gradients where it adds confidence.
"""

import numpy as np

from onemoreepoch.core import Tensor


def numerical_grad(fn, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of a scalar-valued fn at x."""
    grad = np.zeros_like(x)
    flat = x.reshape(-1)
    grad_flat = grad.reshape(-1)
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        plus = fn(x)
        flat[i] = original - eps
        minus = fn(x)
        flat[i] = original
        grad_flat[i] = (plus - minus) / (2 * eps)
    return grad


class TestElementwiseGrads:
    def test_add_sub(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        (a + b).sum().backward()
        np.testing.assert_array_equal(a.grad, [1, 1])
        np.testing.assert_array_equal(b.grad, [1, 1])

        a.zero_grad(), b.zero_grad()
        (a - b).sum().backward()
        np.testing.assert_array_equal(a.grad, [1, 1])
        np.testing.assert_array_equal(b.grad, [-1, -1])

    def test_mul_div(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        (a * b).sum().backward()
        np.testing.assert_array_equal(a.grad, [4, 5])
        np.testing.assert_array_equal(b.grad, [2, 3])

        a.zero_grad(), b.zero_grad()
        (a / b).sum().backward()
        np.testing.assert_allclose(a.grad, [0.25, 0.2])
        np.testing.assert_allclose(b.grad, [-2 / 16, -3 / 25])

    def test_neg_pow(self):
        a = Tensor([3.0], requires_grad=True)
        (-a).backward()
        np.testing.assert_array_equal(a.grad, [-1])

        a.zero_grad()
        (a**3).backward()
        np.testing.assert_allclose(a.grad, [27])  # 3 * 3^2

    def test_exp_log(self):
        a = Tensor([1.5], requires_grad=True)
        a.exp().backward()
        np.testing.assert_allclose(a.grad, np.exp([1.5]))

        a.zero_grad()
        a.log().backward()
        np.testing.assert_allclose(a.grad, [1 / 1.5])


class TestMatMulGrad:
    def test_matches_numerical(self):
        rng = np.random.default_rng(0)
        a_data = rng.standard_normal((3, 4))
        b_data = rng.standard_normal((4, 2))

        a = Tensor(a_data.copy(), requires_grad=True)
        b = Tensor(b_data.copy(), requires_grad=True)
        (a @ b).sum().backward()

        num_a = numerical_grad(lambda x: (x @ b_data).sum(), a_data.copy())
        num_b = numerical_grad(lambda x: (a_data @ x).sum(), b_data.copy())
        np.testing.assert_allclose(a.grad, num_a, atol=1e-4)
        np.testing.assert_allclose(b.grad, num_b, atol=1e-4)


class TestBroadcastGrads:
    def test_bias_broadcast(self):
        x = Tensor(np.ones((3, 4)), requires_grad=True)
        bias = Tensor(np.zeros(4), requires_grad=True)
        (x + bias).sum().backward()
        assert bias.grad.shape == (4,)
        np.testing.assert_array_equal(bias.grad, [3, 3, 3, 3])

    def test_scalar_broadcast(self):
        a = Tensor(np.ones((2, 2)), requires_grad=True)
        (a * 3.0).sum().backward()
        np.testing.assert_array_equal(a.grad, np.full((2, 2), 3.0))


class TestReductionGrads:
    def test_sum_axis_keepdims(self):
        a = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        a.sum(axis=1).sum().backward()
        np.testing.assert_array_equal(a.grad, np.ones((2, 3)))

    def test_mean_scales_gradient(self):
        a = Tensor(np.ones((2, 4)), requires_grad=True)
        a.mean().backward()
        np.testing.assert_allclose(a.grad, np.full((2, 4), 1 / 8))

    def test_mean_axis(self):
        a = Tensor(np.ones((2, 4)), requires_grad=True)
        a.mean(axis=0).sum().backward()
        np.testing.assert_allclose(a.grad, np.full((2, 4), 0.5))


class TestShapeGrads:
    def test_reshape_roundtrip(self):
        a = Tensor(np.arange(6.0), requires_grad=True)
        (a.reshape(2, 3) * 2).sum().backward()
        assert a.grad.shape == (6,)
        np.testing.assert_array_equal(a.grad, np.full(6, 2.0))

    def test_transpose_grad(self):
        a = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        weights = Tensor(np.arange(6.0).reshape(3, 2))
        (a.transpose() * weights).sum().backward()
        np.testing.assert_array_equal(a.grad, weights.data.T)


class TestActivationGrads:
    def test_activations_match_numerical(self):
        from onemoreepoch.autograd.functions import ReLU, Sigmoid, Tanh

        rng = np.random.default_rng(1)
        x_data = rng.standard_normal((5,)) + 0.1  # nudge off relu kink

        for op, ref in [
            (ReLU, lambda x: np.maximum(x, 0).sum()),
            (Sigmoid, lambda x: (1 / (1 + np.exp(-x))).sum()),
            (Tanh, lambda x: np.tanh(x).sum()),
        ]:
            x = Tensor(x_data.copy(), requires_grad=True)
            op.apply(x).sum().backward()
            expected = numerical_grad(ref, x_data.copy())
            np.testing.assert_allclose(x.grad, expected, atol=1e-4)


class TestEngine:
    def test_gradient_accumulates_on_reuse(self):
        # a used twice: grad must be the sum of both paths
        a = Tensor([2.0], requires_grad=True)
        ((a * a) + a).backward()  # d/da (a^2 + a) = 2a + 1 = 5
        np.testing.assert_allclose(a.grad, [5.0])

    def test_diamond_graph(self):
        a = Tensor([3.0], requires_grad=True)
        b = a * 2
        c = a * 4
        (b + c).backward()  # d/da (2a + 4a) = 6
        np.testing.assert_allclose(a.grad, [6.0])

    def test_repeated_backward_accumulates_into_grad(self):
        a = Tensor([1.0], requires_grad=True)
        (a * 2).backward()
        (a * 2).backward()
        np.testing.assert_allclose(a.grad, [4.0])

    def test_explicit_gradient_seed(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        (a * 3).backward(Tensor([1.0, 10.0]))
        np.testing.assert_allclose(a.grad, [3.0, 30.0])

    def test_grad_stops_at_non_requires_grad(self):
        a = Tensor([1.0], requires_grad=True)
        frozen = Tensor([5.0])
        (a * frozen).backward()
        assert frozen.grad is None
        np.testing.assert_allclose(a.grad, [5.0])

    def test_deep_chain(self):
        # 100-op chain exercises the iterative (non-recursive) topo sort
        a = Tensor([1.0], requires_grad=True)
        out = a
        for _ in range(100):
            out = out + a
        out.backward()
        np.testing.assert_allclose(a.grad, [101.0])


class TestFriendlyErrors:
    """New exceptions must stay catchable as their builtin bases."""

    def test_matmul_mismatch_raises_shape_error(self):
        import pytest

        from onemoreepoch.exceptions import ShapeError

        with pytest.raises(ShapeError) as excinfo:
            Tensor.randn(64, 128) @ Tensor.randn(32, 10)
        assert isinstance(excinfo.value, ValueError)  # backward compatible
        assert "(64, 128)" in str(excinfo.value)

    def test_broadcast_failure_raises_shape_error(self):
        import pytest

        from onemoreepoch.exceptions import ShapeError

        with pytest.raises(ShapeError) as excinfo:
            Tensor.randn(3, 2) + Tensor.randn(4, 5)
        assert "Add" in str(excinfo.value)

    def test_backward_without_requires_grad(self):
        import pytest

        from onemoreepoch.exceptions import AutogradError

        with pytest.raises(AutogradError) as excinfo:
            Tensor([1.0]).backward()
        assert isinstance(excinfo.value, RuntimeError)  # backward compatible

    def test_backward_non_scalar_without_seed(self):
        import pytest

        from onemoreepoch.exceptions import AutogradError

        with pytest.raises(AutogradError) as excinfo:
            Tensor.randn(4, 4, requires_grad=True).backward()
        assert "(4, 4)" in str(excinfo.value)


class TestGradientHealthChecks:
    def test_explosion_warns_when_debug_checks_on(self):
        import warnings

        from onemoreepoch import config
        from onemoreepoch.exceptions import GradientWarning

        config.set_debug_checks(True)
        try:
            x = Tensor([50.0], requires_grad=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", GradientWarning)
                ((x * x) * x).sum().backward()  # d/dx = 3x^2 = 7500
            assert any(isinstance(w.message, GradientWarning) for w in caught)
        finally:
            config.set_debug_checks(False)

    def test_no_warning_when_debug_checks_off(self):
        import warnings

        from onemoreepoch.exceptions import GradientWarning

        x = Tensor([50.0], requires_grad=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", GradientWarning)
            ((x * x) * x).sum().backward()
        assert not any(isinstance(w.message, GradientWarning) for w in caught)
