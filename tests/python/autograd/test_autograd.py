import numpy as np

from onemoreepoch.core import Tensor


# Computes a central-difference numerical gradient of a scalar-valued fn at x
def numerical_grad(fn, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
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


# Tests gradients of elementwise arithmetic ops
class TestElementwiseGrads:
    # Checks gradients of addition and subtraction
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

    # Checks gradients of multiplication and division
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

    # Checks gradients of negation and power
    def test_neg_pow(self):
        a = Tensor([3.0], requires_grad=True)
        (-a).backward()
        np.testing.assert_array_equal(a.grad, [-1])

        a.zero_grad()
        (a**3).backward()
        np.testing.assert_allclose(a.grad, [27])

    # Checks gradients of exp and log
    def test_exp_log(self):
        a = Tensor([1.5], requires_grad=True)
        a.exp().backward()
        np.testing.assert_allclose(a.grad, np.exp([1.5]))

        a.zero_grad()
        a.log().backward()
        np.testing.assert_allclose(a.grad, [1 / 1.5])


# Tests matmul gradients against a numerical reference
class TestMatMulGrad:
    # Checks matmul gradients match central-difference numerical gradients
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


# Tests gradients through broadcasted operations
class TestBroadcastGrads:
    # Checks a broadcast bias's gradient sums over the broadcast axis
    def test_bias_broadcast(self):
        x = Tensor(np.ones((3, 4)), requires_grad=True)
        bias = Tensor(np.zeros(4), requires_grad=True)
        (x + bias).sum().backward()
        assert bias.grad.shape == (4,)
        np.testing.assert_array_equal(bias.grad, [3, 3, 3, 3])

    # Checks a scalar broadcast multiply produces the correct gradient
    def test_scalar_broadcast(self):
        a = Tensor(np.ones((2, 2)), requires_grad=True)
        (a * 3.0).sum().backward()
        np.testing.assert_array_equal(a.grad, np.full((2, 2), 3.0))


# Tests gradients of sum/mean reductions
class TestReductionGrads:
    # Checks sum(axis=1) then sum() gradient is all-ones
    def test_sum_axis_keepdims(self):
        a = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        a.sum(axis=1).sum().backward()
        np.testing.assert_array_equal(a.grad, np.ones((2, 3)))

    # Checks mean() scales the gradient by 1/N
    def test_mean_scales_gradient(self):
        a = Tensor(np.ones((2, 4)), requires_grad=True)
        a.mean().backward()
        np.testing.assert_allclose(a.grad, np.full((2, 4), 1 / 8))

    # Checks mean(axis=0) scales the gradient by 1/axis-size
    def test_mean_axis(self):
        a = Tensor(np.ones((2, 4)), requires_grad=True)
        a.mean(axis=0).sum().backward()
        np.testing.assert_allclose(a.grad, np.full((2, 4), 0.5))


# Tests gradients of reshape/transpose
class TestShapeGrads:
    # Checks reshape then multiply gradient reshapes back correctly
    def test_reshape_roundtrip(self):
        a = Tensor(np.arange(6.0), requires_grad=True)
        (a.reshape(2, 3) * 2).sum().backward()
        assert a.grad.shape == (6,)
        np.testing.assert_array_equal(a.grad, np.full(6, 2.0))

    # Checks transpose gradient is the transpose of the upstream gradient
    def test_transpose_grad(self):
        a = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        weights = Tensor(np.arange(6.0).reshape(3, 2))
        (a.transpose() * weights).sum().backward()
        np.testing.assert_array_equal(a.grad, weights.data.T)


# Tests activation function gradients against numerical references
class TestActivationGrads:
    # Checks ReLU/Sigmoid/Tanh gradients match central-difference numerical gradients
    def test_activations_match_numerical(self):
        from onemoreepoch.autograd.functions import ReLU, Sigmoid, Tanh

        rng = np.random.default_rng(1)
        x_data = rng.standard_normal((5,)) + 0.1

        for op, ref in [
            (ReLU, lambda x: np.maximum(x, 0).sum()),
            (Sigmoid, lambda x: (1 / (1 + np.exp(-x))).sum()),
            (Tanh, lambda x: np.tanh(x).sum()),
        ]:
            x = Tensor(x_data.copy(), requires_grad=True)
            op.apply(x).sum().backward()
            expected = numerical_grad(ref, x_data.copy())
            np.testing.assert_allclose(x.grad, expected, atol=1e-4)


# Tests the backward engine's graph-traversal behavior
class TestEngine:
    # Checks gradient accumulates correctly when a tensor is used twice
    def test_gradient_accumulates_on_reuse(self):
        a = Tensor([2.0], requires_grad=True)
        ((a * a) + a).backward()
        np.testing.assert_allclose(a.grad, [5.0])

    # Checks gradient accumulates correctly across a diamond-shaped graph
    def test_diamond_graph(self):
        a = Tensor([3.0], requires_grad=True)
        b = a * 2
        c = a * 4
        (b + c).backward()
        np.testing.assert_allclose(a.grad, [6.0])

    # Checks repeated backward() calls accumulate into the same gradient
    def test_repeated_backward_accumulates_into_grad(self):
        a = Tensor([1.0], requires_grad=True)
        (a * 2).backward()
        (a * 2).backward()
        np.testing.assert_allclose(a.grad, [4.0])

    # Checks an explicit seed gradient is used instead of the default ones-seed
    def test_explicit_gradient_seed(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        (a * 3).backward(Tensor([1.0, 10.0]))
        np.testing.assert_allclose(a.grad, [3.0, 30.0])

    # Checks gradient does not propagate into a tensor with requires_grad=False
    def test_grad_stops_at_non_requires_grad(self):
        a = Tensor([1.0], requires_grad=True)
        frozen = Tensor([5.0])
        (a * frozen).backward()
        assert frozen.grad is None
        np.testing.assert_allclose(a.grad, [5.0])

    # Checks a long chain of ops is handled by the iterative (non-recursive) topo sort
    def test_deep_chain(self):
        a = Tensor([1.0], requires_grad=True)
        out = a
        for _ in range(100):
            out = out + a
        out.backward()
        np.testing.assert_allclose(a.grad, [101.0])


# Tests that friendly exceptions stay catchable as their builtin bases
class TestFriendlyErrors:
    # Checks a matmul shape mismatch raises ShapeError (and is still a ValueError)
    def test_matmul_mismatch_raises_shape_error(self):
        import pytest

        from onemoreepoch.exceptions import ShapeError

        with pytest.raises(ShapeError) as excinfo:
            Tensor.randn(64, 128) @ Tensor.randn(32, 10)
        assert isinstance(excinfo.value, ValueError)
        assert "(64, 128)" in str(excinfo.value)

    # Checks a broadcast failure raises ShapeError
    def test_broadcast_failure_raises_shape_error(self):
        import pytest

        from onemoreepoch.exceptions import ShapeError

        with pytest.raises(ShapeError) as excinfo:
            Tensor.randn(3, 2) + Tensor.randn(4, 5)
        assert "Add" in str(excinfo.value)

    # Checks backward() without requires_grad raises AutogradError (and is still a RuntimeError)
    def test_backward_without_requires_grad(self):
        import pytest

        from onemoreepoch.exceptions import AutogradError

        with pytest.raises(AutogradError) as excinfo:
            Tensor([1.0]).backward()
        assert isinstance(excinfo.value, RuntimeError)

    # Checks backward() on a non-scalar without a seed raises AutogradError
    def test_backward_non_scalar_without_seed(self):
        import pytest

        from onemoreepoch.exceptions import AutogradError

        with pytest.raises(AutogradError) as excinfo:
            Tensor.randn(4, 4, requires_grad=True).backward()
        assert "(4, 4)" in str(excinfo.value)


# Tests the opt-in gradient-health diagnostics
class TestGradientHealthChecks:
    # Checks a GradientWarning fires when debug checks are enabled and gradients explode
    def test_explosion_warns_when_debug_checks_on(self):
        import warnings

        from onemoreepoch import config
        from onemoreepoch.exceptions import GradientWarning

        config.set_debug_checks(True)
        try:
            x = Tensor([50.0], requires_grad=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", GradientWarning)
                ((x * x) * x).sum().backward()
            assert any(isinstance(w.message, GradientWarning) for w in caught)
        finally:
            config.set_debug_checks(False)

    # Checks no GradientWarning fires when debug checks are disabled
    def test_no_warning_when_debug_checks_off(self):
        import warnings

        from onemoreepoch.exceptions import GradientWarning

        x = Tensor([50.0], requires_grad=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", GradientWarning)
            ((x * x) * x).sum().backward()
        assert not any(isinstance(w.message, GradientWarning) for w in caught)
