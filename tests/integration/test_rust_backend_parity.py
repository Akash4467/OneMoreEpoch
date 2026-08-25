import numpy as np
import pytest

from onemoreepoch.core.backend.numpy_backend import NumPyBackend

rust_backend_module = pytest.importorskip(
    "onemoreepoch.core.backend.rust_backend", reason="Rust extension not built"
)
RustBackend = rust_backend_module.RustBackend


# Converts a numpy array or RustArray into a numpy array
def _to_numpy(raw) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        return raw
    return np.array(raw.tolist()).reshape(raw.shape)


# Returns a fresh (NumPyBackend, RustBackend) pair for each test
@pytest.fixture
def backends():
    return NumPyBackend(), RustBackend()


# Tests array-creation parity between backends
class TestArrayCreation:
    # Checks zeros/ones/full match between backends
    def test_zeros_ones_full(self, backends):
        npb, rb = backends
        np.testing.assert_allclose(
            _to_numpy(npb.zeros((2, 3))), _to_numpy(rb.zeros((2, 3)))
        )
        np.testing.assert_allclose(
            _to_numpy(npb.ones((2, 3))), _to_numpy(rb.ones((2, 3)))
        )
        np.testing.assert_allclose(
            _to_numpy(npb.full((2, 2), 7.0)), _to_numpy(rb.full((2, 2), 7.0))
        )

    # Checks array() from a nested list matches between backends
    def test_array_from_nested_list(self, backends):
        npb, rb = backends
        data = [[1.0, 2.0], [3.0, 4.0]]
        np.testing.assert_allclose(
            _to_numpy(npb.array(data)), _to_numpy(rb.array(data))
        )


# Tests arithmetic parity between backends
class TestArithmeticParity:
    # Returns two random arrays for arithmetic tests
    @pytest.fixture
    def arrays(self):
        rng = np.random.default_rng(0)
        return rng.standard_normal((3, 4)), rng.standard_normal((3, 4))

    # Checks binary elementwise ops match between backends
    def test_binary_ops(self, backends, arrays):
        npb, rb = backends
        a_np, b_np = arrays
        for op in ("add", "subtract", "multiply", "divide", "maximum", "greater"):
            np_result = _to_numpy(getattr(npb, op)(npb.array(a_np), npb.array(b_np)))
            rust_result = _to_numpy(getattr(rb, op)(rb.array(a_np), rb.array(b_np)))
            np.testing.assert_allclose(np_result, rust_result, atol=1e-10, err_msg=op)

    # Checks binary ops with a scalar operand on either side match between backends
    def test_binary_ops_with_scalar(self, backends, arrays):
        npb, rb = backends
        a_np, _ = arrays
        for op in ("add", "subtract", "multiply", "divide"):
            np_result = _to_numpy(getattr(npb, op)(npb.array(a_np), 2.5))
            rust_result = _to_numpy(getattr(rb, op)(rb.array(a_np), 2.5))
            np.testing.assert_allclose(np_result, rust_result, atol=1e-10, err_msg=op)
            np_result2 = _to_numpy(getattr(npb, op)(2.5, npb.array(a_np)))
            rust_result2 = _to_numpy(getattr(rb, op)(2.5, rb.array(a_np)))
            np.testing.assert_allclose(np_result2, rust_result2, atol=1e-10, err_msg=op)

    # Checks unary elementwise ops match between backends
    def test_unary_ops(self, backends, arrays):
        npb, rb = backends
        a_np, _ = arrays
        positive = np.abs(a_np) + 0.1
        for op, arr in [
            ("negative", a_np),
            ("absolute", a_np),
            ("exp", a_np),
            ("log", positive),
            ("tanh", a_np),
            ("sqrt", positive),
        ]:
            np_result = _to_numpy(getattr(npb, op)(npb.array(arr)))
            rust_result = _to_numpy(getattr(rb, op)(rb.array(arr)))
            np.testing.assert_allclose(np_result, rust_result, atol=1e-8, err_msg=op)

    # Checks power matches between backends
    def test_power(self, backends, arrays):
        npb, rb = backends
        a_np, _ = arrays
        positive = np.abs(a_np) + 0.1
        np_result = _to_numpy(npb.power(npb.array(positive), 3.0))
        rust_result = _to_numpy(rb.power(rb.array(positive), 3.0))
        np.testing.assert_allclose(np_result, rust_result, atol=1e-8)


# Tests matmul parity between backends
class TestMatMulParity:
    # Checks 2D @ 2D matmul matches between backends
    def test_2d_2d(self, backends):
        npb, rb = backends
        rng = np.random.default_rng(1)
        a, b = rng.standard_normal((5, 3)), rng.standard_normal((3, 4))
        np.testing.assert_allclose(
            _to_numpy(npb.matmul(npb.array(a), npb.array(b))),
            _to_numpy(rb.matmul(rb.array(a), rb.array(b))),
            atol=1e-8,
        )

    # Checks 2D @ 1D matmul matches between backends
    def test_2d_1d(self, backends):
        npb, rb = backends
        rng = np.random.default_rng(2)
        a, b = rng.standard_normal((4, 3)), rng.standard_normal((3,))
        np.testing.assert_allclose(
            _to_numpy(npb.matmul(npb.array(a), npb.array(b))),
            _to_numpy(rb.matmul(rb.array(a), rb.array(b))),
            atol=1e-8,
        )


# Tests reduction parity between backends across axis/keepdims combinations
class TestReductionParity:
    # Returns a random 3D array for reduction tests
    @pytest.fixture
    def a(self):
        return np.random.default_rng(3).standard_normal((3, 4, 2))

    # Checks sum/mean/max match between backends for every axis/keepdims combination
    @pytest.mark.parametrize("op", ["sum", "mean", "max"])
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 2), (1, 2)])
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_reduction(self, backends, a, op, axis, keepdims):
        npb, rb = backends
        np_result = _to_numpy(
            getattr(npb, op)(npb.array(a), axis=axis, keepdims=keepdims)
        )
        rust_result = _to_numpy(
            getattr(rb, op)(rb.array(a), axis=axis, keepdims=keepdims)
        )
        np.testing.assert_allclose(np_result, rust_result, atol=1e-8)


# Tests shape-manipulation parity between backends
class TestShapeParity:
    # Checks reshape matches between backends
    def test_reshape(self, backends):
        npb, rb = backends
        a = np.arange(24.0)
        np.testing.assert_allclose(
            _to_numpy(npb.reshape(npb.array(a), (2, 3, 4))),
            _to_numpy(rb.reshape(rb.array(a), (2, 3, 4))),
        )

    # Checks transpose (default and explicit axes) matches between backends
    def test_transpose_default_and_explicit_axes(self, backends):
        npb, rb = backends
        a = np.arange(24.0).reshape(2, 3, 4)
        np.testing.assert_allclose(
            _to_numpy(npb.transpose(npb.array(a))), _to_numpy(rb.transpose(rb.array(a)))
        )
        np.testing.assert_allclose(
            _to_numpy(npb.transpose(npb.array(a), (1, 0, 2))),
            _to_numpy(rb.transpose(rb.array(a), (1, 0, 2))),
        )

    # Checks broadcast_to matches between backends
    def test_broadcast_to(self, backends):
        npb, rb = backends
        a = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            _to_numpy(npb.broadcast_to(npb.array(a), (4, 3))),
            _to_numpy(rb.broadcast_to(rb.array(a), (4, 3))),
        )


# Tests that Conv2D's NumPy-delegated im2col/col2im round-trip correctly under RustBackend
class TestConv2DUnderRustBackend:
    # Checks Conv2D forward/backward under the Rust backend matches the NumPy backend
    def test_forward_backward_matches_numpy_backend(self):
        import onemoreepoch.core.backend.registry as registry_mod

        # Runs a fixed Conv2D forward/backward pass under the given backend name
        def run(backend_name):
            original = registry_mod._DEFAULT_BACKEND
            registry_mod._DEFAULT_BACKEND = backend_name
            try:
                from onemoreepoch import nn
                from onemoreepoch.core import Tensor

                rng = np.random.default_rng(7)
                x_data = rng.standard_normal((2, 3, 6, 6))
                conv = nn.Conv2D(3, 4, kernel_size=3, stride=2, padding=1)
                weight_data = np.linspace(-0.3, 0.3, num=conv.weight.size).reshape(
                    conv.weight.shape
                )
                bias_data = np.linspace(-0.1, 0.1, num=conv.bias.size)
                conv.weight.data = registry_mod.get_backend().array(weight_data)
                conv.bias.data = registry_mod.get_backend().array(bias_data)

                x = Tensor(x_data, requires_grad=True)
                out = conv(x)
                out.sum().backward()
                return out.numpy().copy(), _to_numpy(x.grad)
            finally:
                registry_mod._DEFAULT_BACKEND = original

        numpy_out, numpy_grad = run("numpy")
        rust_out, rust_grad = run("rust")
        np.testing.assert_allclose(numpy_out, rust_out, atol=1e-6)
        np.testing.assert_allclose(numpy_grad, rust_grad, atol=1e-6)


# Tests a full fixed-weight training pipeline matches across backends
class TestFullPipelineParity:
    # Runs a fixed Linear-ReLU-Linear model's forward/backward/step under the given backend
    def _run_pipeline(self, backend_name: str) -> tuple:
        import onemoreepoch.core.backend.registry as registry_mod

        original = registry_mod._DEFAULT_BACKEND
        registry_mod._DEFAULT_BACKEND = backend_name
        try:
            from onemoreepoch import nn
            from onemoreepoch.core import Tensor
            from onemoreepoch.optim import SGD

            model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
            weight_shapes = [p.shape for p in model.parameters()]
            fixed_weights = [
                np.linspace(-0.5, 0.5, num=int(np.prod(s))).reshape(s)
                for s in weight_shapes
            ]
            for param, values in zip(model.parameters(), fixed_weights):
                param.data = registry_mod.get_backend().array(values)

            x = Tensor(np.linspace(-1, 1, num=6).reshape(2, 3), requires_grad=True)
            target = Tensor(np.array([[1.0], [0.0]]))
            criterion = nn.MSELoss()
            optimizer = SGD(model.parameters(), lr=0.1)

            loss = criterion(model(x), target)
            loss_value = loss.item()
            loss.backward()
            grad_values = [_to_numpy(p.grad) for p in model.parameters()]
            optimizer.step()
            updated_values = [p.numpy().copy() for p in model.parameters()]
            return loss_value, grad_values, updated_values
        finally:
            registry_mod._DEFAULT_BACKEND = original

    # Checks loss, gradients, and updated weights all match between backends
    def test_linear_relu_linear_matches_across_backends(self):
        numpy_loss, numpy_grads, numpy_updated = self._run_pipeline("numpy")
        rust_loss, rust_grads, rust_updated = self._run_pipeline("rust")

        assert numpy_loss == pytest.approx(rust_loss, abs=1e-6)
        for grad_np, grad_rust in zip(numpy_grads, rust_grads):
            np.testing.assert_allclose(grad_np, grad_rust, atol=1e-6)
        for updated_np, updated_rust in zip(numpy_updated, rust_updated):
            np.testing.assert_allclose(updated_np, updated_rust, atol=1e-6)
