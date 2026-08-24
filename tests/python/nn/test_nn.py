"""Tests for nn: Module registration, layers, activations, losses."""

import numpy as np
import pytest

from onemoreepoch import nn
from onemoreepoch.core import Parameter, Tensor
from onemoreepoch.exceptions import ModuleError


class TestModule:
    def test_parameter_registration(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = Parameter([1.0])
                self.child = nn.Linear(2, 3)

        model = Model()
        params = model.parameters()
        assert model.w in params
        assert model.child.weight in params
        assert model.child.bias in params
        assert len(params) == 3

    def test_named_parameters_use_dotted_paths(self):
        model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        names = dict(model.named_parameters()).keys()
        assert "0.weight" in names
        assert "1.bias" in names

    def test_train_eval_recurse(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        model.eval()
        assert not model.training and not model[0].training
        model.train()
        assert model.training and model[0].training

    def test_zero_grad(self):
        model = nn.Linear(2, 2)
        x = Tensor(np.ones((1, 2)))
        model(x).sum().backward()
        assert model.weight.grad is not None
        model.zero_grad()
        assert model.weight.grad is None

    def test_forward_not_implemented(self):
        with pytest.raises(NotImplementedError):
            nn.Module()(Tensor([1.0]))

    def test_state_dict_roundtrip(self):
        src = nn.Linear(3, 2)
        dst = nn.Linear(3, 2)
        dst.load_state_dict(src.state_dict())
        np.testing.assert_array_equal(src.weight.data, dst.weight.data)
        np.testing.assert_array_equal(src.bias.data, dst.bias.data)

    def test_load_state_dict_rejects_mismatched_keys(self):
        model = nn.Linear(2, 2)
        with pytest.raises(ModuleError):
            model.load_state_dict({"nope": np.zeros((2, 2))})


class TestLinear:
    def test_output_shape(self):
        layer = nn.Linear(4, 3)
        out = layer(Tensor(np.ones((5, 4))))
        assert out.shape == (5, 3)

    def test_no_bias(self):
        layer = nn.Linear(4, 3, bias=False)
        assert layer.bias is None
        assert len(layer.parameters()) == 1

    def test_computation(self):
        layer = nn.Linear(2, 1)
        layer.weight.data = np.array([[2.0], [3.0]])
        layer.bias.data = np.array([1.0])
        out = layer(Tensor([[1.0, 1.0]]))
        assert out.item() == 6.0


class TestSequential:
    def test_chains_in_order(self):
        model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
        out = model(Tensor(np.ones((3, 2))))
        assert out.shape == (3, 1)

    def test_len_and_getitem(self):
        model = nn.Sequential(nn.Linear(1, 1), nn.Tanh())
        assert len(model) == 2
        assert isinstance(model[1], nn.Tanh)


class TestActivations:
    def test_relu_clamps_negatives(self):
        out = nn.ReLU()(Tensor([-1.0, 0.0, 2.0]))
        np.testing.assert_array_equal(out.data, [0, 0, 2])

    def test_sigmoid_range(self):
        out = nn.Sigmoid()(Tensor([0.0]))
        np.testing.assert_allclose(out.data, [0.5])

    def test_tanh(self):
        out = nn.Tanh()(Tensor([0.0]))
        np.testing.assert_allclose(out.data, [0.0])


class TestMSELoss:
    def test_value(self):
        loss = nn.MSELoss()(Tensor([1.0, 2.0]), Tensor([3.0, 2.0]))
        assert loss.item() == 2.0  # ((2)^2 + 0) / 2

    def test_gradient_flows_to_prediction(self):
        pred = Tensor([1.0, 2.0], requires_grad=True)
        nn.MSELoss()(pred, Tensor([0.0, 0.0])).backward()
        np.testing.assert_allclose(pred.grad, [1.0, 2.0])  # 2 * pred / n


def _numerical_grad(fn, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    grad = np.zeros_like(x)
    flat, gflat = x.reshape(-1), grad.reshape(-1)
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        plus = fn(x)
        flat[i] = original - eps
        minus = fn(x)
        flat[i] = original
        gflat[i] = (plus - minus) / (2 * eps)
    return grad


class TestConv2D:
    def test_output_shape_with_stride_and_padding(self):
        layer = nn.Conv2D(3, 4, kernel_size=3, stride=2, padding=1)
        out = layer(Tensor(np.random.randn(2, 3, 9, 9)))
        assert out.shape == (2, 4, 5, 5)

    def test_no_bias(self):
        layer = nn.Conv2D(2, 3, kernel_size=3, bias=False)
        assert layer.bias is None
        assert len(layer.parameters()) == 1

    def test_gradients_match_numerical(self):
        rng = np.random.default_rng(2)
        x_data = rng.standard_normal((2, 2, 5, 5))
        layer = nn.Conv2D(2, 3, kernel_size=3, stride=2, padding=1)
        w_data, b_data = layer.weight.data.copy(), layer.bias.data.copy()

        def forward(x, w=w_data, b=b_data):
            from onemoreepoch.core.backend.numpy_backend import NumPyBackend

            backend = NumPyBackend()
            n, c, h, wd = x.shape
            co, k = w.shape[0], w.shape[1] * w.shape[2] * w.shape[3]
            cols = backend.im2col(x, (3, 3), (2, 2), (1, 1))
            ho = (h + 2 - 3) // 2 + 1
            wo = (wd + 2 - 3) // 2 + 1
            w2d = w.reshape(co, k)
            cols2d = cols.transpose(1, 0, 2).reshape(k, n * ho * wo)
            out = (w2d @ cols2d).reshape(co, n, ho * wo).transpose(1, 0, 2)
            return (out.reshape(n, co, ho, wo) + b.reshape(1, co, 1, 1)).sum()

        x = Tensor(x_data.copy(), requires_grad=True)
        layer(x).sum().backward()

        num_gx = _numerical_grad(forward, x_data.copy())
        np.testing.assert_allclose(x.grad, num_gx, atol=1e-3)


class TestDropout:
    def test_eval_mode_is_identity(self):
        layer = nn.Dropout(0.5)
        layer.eval()
        x = Tensor(np.ones(100))
        np.testing.assert_array_equal(layer(x).data, x.data)

    def test_training_mode_scales_survivors(self):
        layer = nn.Dropout(0.5)
        x = Tensor(np.ones(5000))
        out = layer(x).data
        survivors = out[out != 0]
        np.testing.assert_allclose(survivors, 2.0)  # 1 / (1 - 0.5)
        assert 0.3 < (out != 0).mean() < 0.7

    def test_zero_p_is_identity(self):
        layer = nn.Dropout(0.0)
        x = Tensor(np.ones(10))
        np.testing.assert_array_equal(layer(x).data, x.data)

    def test_rejects_invalid_p(self):
        from onemoreepoch.exceptions import ModuleError

        with pytest.raises(ModuleError):
            nn.Dropout(1.5)


class TestBatchNorm:
    def test_normalizes_training_batch(self):
        layer = nn.BatchNorm(4)
        x = Tensor(np.random.randn(32, 4) * 5 + 3, requires_grad=True)
        out = layer(x)
        np.testing.assert_allclose(out.data.mean(axis=0), np.zeros(4), atol=1e-6)
        np.testing.assert_allclose(out.data.std(axis=0), np.ones(4), atol=1e-4)

    def test_eval_uses_running_stats(self):
        layer = nn.BatchNorm(3)
        for _ in range(5):
            layer(Tensor(np.random.randn(16, 3) * 2 + 1))
        layer.eval()
        x = Tensor(np.random.randn(4, 3))
        out = layer(x)
        assert out.shape == (4, 3)

    def test_gradient_flows(self):
        layer = nn.BatchNorm(3)
        x = Tensor(np.random.randn(8, 3), requires_grad=True)
        layer(x).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (8, 3)

    def test_rejects_invalid_eps(self):
        from onemoreepoch.exceptions import ModuleError

        with pytest.raises(ModuleError):
            nn.BatchNorm(3, eps=0.0)


class TestInit:
    def test_kaiming_uniform_bounds(self):
        t = Tensor.zeros(100, 50)
        nn.init.kaiming_uniform_(t, fan_in=100)
        bound = 100**-0.5
        assert t.data.min() >= -bound and t.data.max() <= bound

    def test_zeros_and_ones(self):
        t = Tensor.ones(3, 3)
        nn.init.zeros_(t)
        np.testing.assert_array_equal(t.data, np.zeros((3, 3)))
        nn.init.ones_(t)
        np.testing.assert_array_equal(t.data, np.ones((3, 3)))
