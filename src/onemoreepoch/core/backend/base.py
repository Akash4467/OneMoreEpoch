"""Abstract backend interface for array operations.

The backend layer exposes only low-level array primitives. It must never
know about Tensor, autograd, or any layer above it (ADR-007).
"""

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """Base class for execution backends."""

    name: str

    # -- array creation -------------------------------------------------

    @abstractmethod
    def is_native(self, data: Any) -> bool:
        """Return True if ``data`` is already a native array of this backend."""

    @abstractmethod
    def array(self, data: Any, dtype: Any = None) -> Any:
        """Create a backend array from raw data."""

    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Return an array filled with zeros."""

    @abstractmethod
    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Return an array filled with ones."""

    @abstractmethod
    def full(self, shape: tuple[int, ...], fill_value: Any, dtype: Any = None) -> Any:
        """Return an array filled with ``fill_value``."""

    @abstractmethod
    def zeros_like(self, array: Any) -> Any:
        """Return a zero-filled array with the same shape and dtype."""

    @abstractmethod
    def ones_like(self, array: Any) -> Any:
        """Return a one-filled array with the same shape and dtype."""

    # -- random ----------------------------------------------------------

    @abstractmethod
    def seed(self, value: int) -> None:
        """Seed the backend's random number generator."""

    @abstractmethod
    def randn(self, shape: tuple[int, ...]) -> Any:
        """Return an array of samples from the standard normal distribution."""

    @abstractmethod
    def rand(self, shape: tuple[int, ...]) -> Any:
        """Return an array of samples from the uniform [0, 1) distribution."""

    # -- arithmetic ------------------------------------------------------

    @abstractmethod
    def add(self, a: Any, b: Any) -> Any:
        """Element-wise addition."""

    @abstractmethod
    def subtract(self, a: Any, b: Any) -> Any:
        """Element-wise subtraction."""

    @abstractmethod
    def multiply(self, a: Any, b: Any) -> Any:
        """Element-wise multiplication."""

    @abstractmethod
    def divide(self, a: Any, b: Any) -> Any:
        """Element-wise division."""

    @abstractmethod
    def negative(self, a: Any) -> Any:
        """Element-wise negation."""

    @abstractmethod
    def absolute(self, a: Any) -> Any:
        """Element-wise absolute value."""

    @abstractmethod
    def power(self, a: Any, exponent: Any) -> Any:
        """Element-wise exponentiation."""

    @abstractmethod
    def sqrt(self, a: Any) -> Any:
        """Element-wise square root."""

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication."""

    @abstractmethod
    def exp(self, a: Any) -> Any:
        """Element-wise natural exponential."""

    @abstractmethod
    def log(self, a: Any) -> Any:
        """Element-wise natural logarithm."""

    @abstractmethod
    def tanh(self, a: Any) -> Any:
        """Element-wise hyperbolic tangent."""

    @abstractmethod
    def maximum(self, a: Any, b: Any) -> Any:
        """Element-wise maximum of two arrays."""

    @abstractmethod
    def greater(self, a: Any, b: Any) -> Any:
        """Element-wise ``a > b`` comparison."""

    # -- reductions ------------------------------------------------------

    @abstractmethod
    def sum(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any:
        """Sum over the given axis (or all elements)."""

    @abstractmethod
    def mean(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any:
        """Mean over the given axis (or all elements)."""

    @abstractmethod
    def max(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any:
        """Maximum over the given axis (or all elements)."""

    # -- shape manipulation ----------------------------------------------

    @abstractmethod
    def reshape(self, a: Any, shape: tuple[int, ...]) -> Any:
        """Return the array with a new shape."""

    @abstractmethod
    def transpose(self, a: Any, axes: tuple[int, ...] | None = None) -> Any:
        """Return the array with permuted axes."""

    @abstractmethod
    def broadcast_to(self, a: Any, shape: tuple[int, ...]) -> Any:
        """Broadcast the array to a new shape."""

    # -- windowed extraction (Conv2D) -------------------------------------

    @abstractmethod
    def im2col(
        self,
        a: Any,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> Any:
        """Extract sliding windows from a zero-padded (N, C, H, W) array.

        Returns shape (N, C * KH * KW, H_out * W_out). Padding is applied
        internally so callers never need to slice a padded array back
        down themselves.
        """

    @abstractmethod
    def col2im(
        self,
        cols: Any,
        input_shape: tuple[int, int, int, int],
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> Any:
        """Adjoint of ``im2col``: scatter-add columns back to (N, C, H, W).

        Overlapping windows accumulate. ``input_shape`` is the original
        *unpadded* shape; the padding margin is cropped internally.
        """
