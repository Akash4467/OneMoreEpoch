"""Tests for the Tensor primitive."""

import numpy as np
import pytest

from onemoreepoch.core import Device, Parameter, Tensor


class TestConstruction:
    def test_from_list(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.ndim == 1
        assert t.size == 3

    def test_defaults(self):
        t = Tensor([1.0])
        assert t.requires_grad is False
        assert t.grad is None
        assert t.creator is None
        assert t.parents == ()
        assert t.device == Device.cpu()

    def test_factories(self):
        assert Tensor.zeros(2, 3).shape == (2, 3)
        assert Tensor.ones(2).data.sum() == 2
        assert Tensor.randn(4, 4).shape == (4, 4)
        assert Tensor.rand(5).shape == (5,)

    def test_item_and_numpy(self):
        t = Tensor(3.5)
        assert t.item() == 3.5
        assert isinstance(t.numpy(), np.ndarray)


class TestGraphBookkeeping:
    def test_op_result_records_creator_and_parents(self):
        a = Tensor([1.0], requires_grad=True)
        b = Tensor([2.0])
        c = a + b
        assert c.requires_grad
        assert c.creator is not None
        assert c.parents == (a, b)
        assert c.context is not None

    def test_no_graph_without_requires_grad(self):
        c = Tensor([1.0]) + Tensor([2.0])
        assert not c.requires_grad
        assert c.creator is None

    def test_detach_cuts_graph(self):
        a = Tensor([1.0], requires_grad=True)
        d = (a * 2).detach()
        assert not d.requires_grad
        assert d.creator is None


class TestOperators:
    def test_arithmetic_values(self):
        a, b = Tensor([6.0]), Tensor([3.0])
        assert (a + b).item() == 9
        assert (a - b).item() == 3
        assert (a * b).item() == 18
        assert (a / b).item() == 2
        assert (-a).item() == -6
        assert (a**2).item() == 36

    def test_reflected_operators_with_scalars(self):
        a = Tensor([2.0])
        assert (3 + a).item() == 5
        assert (3 - a).item() == 1
        assert (3 * a).item() == 6
        assert (8 / a).item() == 4

    def test_matmul(self):
        a = Tensor([[1.0, 2.0]])
        b = Tensor([[3.0], [4.0]])
        assert (a @ b).item() == 11

    def test_shape_methods(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.reshape(4).shape == (4,)
        assert t.transpose().shape == (2, 2)
        assert t.T.data[0, 1] == 3.0

    def test_reductions(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.sum().item() == 10
        assert t.mean().item() == 2.5
        np.testing.assert_array_equal(t.sum(axis=0).data, [4, 6])


class TestParameter:
    def test_is_tensor(self):
        p = Parameter([1.0, 2.0])
        assert isinstance(p, Tensor)

    def test_requires_grad_by_default(self):
        assert Parameter([1.0]).requires_grad is True


class TestBackwardErrors:
    def test_backward_without_requires_grad_raises(self):
        with pytest.raises(RuntimeError):
            Tensor([1.0]).backward()

    def test_backward_on_non_scalar_requires_gradient(self):
        t = Tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(RuntimeError):
            (t * 2).backward()
