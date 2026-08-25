from abc import ABC, abstractmethod
from typing import Any


# Abstract interface every array-computation backend (NumPy, Rust, ...) must implement
class Backend(ABC):
    name: str

    # Returns True if data is already a native array of this backend
    @abstractmethod
    def is_native(self, data: Any) -> bool: ...

    # Creates a backend array from raw data
    @abstractmethod
    def array(self, data: Any, dtype: Any = None) -> Any: ...

    # Returns an array filled with zeros
    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any: ...

    # Returns an array filled with ones
    @abstractmethod
    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any: ...

    # Returns an array filled with fill_value
    @abstractmethod
    def full(
        self, shape: tuple[int, ...], fill_value: Any, dtype: Any = None
    ) -> Any: ...

    # Returns a zero-filled array with the same shape and dtype as array
    @abstractmethod
    def zeros_like(self, array: Any) -> Any: ...

    # Returns a one-filled array with the same shape and dtype as array
    @abstractmethod
    def ones_like(self, array: Any) -> Any: ...

    # Seeds the backend's random number generator
    @abstractmethod
    def seed(self, value: int) -> None: ...

    # Returns an array of standard-normal random samples
    @abstractmethod
    def randn(self, shape: tuple[int, ...]) -> Any: ...

    # Returns an array of uniform [0, 1) random samples
    @abstractmethod
    def rand(self, shape: tuple[int, ...]) -> Any: ...

    # Element-wise addition
    @abstractmethod
    def add(self, a: Any, b: Any) -> Any: ...

    # Element-wise subtraction
    @abstractmethod
    def subtract(self, a: Any, b: Any) -> Any: ...

    # Element-wise multiplication
    @abstractmethod
    def multiply(self, a: Any, b: Any) -> Any: ...

    # Element-wise division
    @abstractmethod
    def divide(self, a: Any, b: Any) -> Any: ...

    # Element-wise negation
    @abstractmethod
    def negative(self, a: Any) -> Any: ...

    # Element-wise absolute value
    @abstractmethod
    def absolute(self, a: Any) -> Any: ...

    # Element-wise exponentiation
    @abstractmethod
    def power(self, a: Any, exponent: Any) -> Any: ...

    # Element-wise square root
    @abstractmethod
    def sqrt(self, a: Any) -> Any: ...

    # Matrix multiplication
    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any: ...

    # Element-wise natural exponential
    @abstractmethod
    def exp(self, a: Any) -> Any: ...

    # Element-wise natural logarithm
    @abstractmethod
    def log(self, a: Any) -> Any: ...

    # Element-wise hyperbolic tangent
    @abstractmethod
    def tanh(self, a: Any) -> Any: ...

    # Element-wise maximum of two arrays
    @abstractmethod
    def maximum(self, a: Any, b: Any) -> Any: ...

    # Element-wise a > b comparison
    @abstractmethod
    def greater(self, a: Any, b: Any) -> Any: ...

    # Sums over the given axis (or all elements)
    @abstractmethod
    def sum(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any: ...

    # Averages over the given axis (or all elements)
    @abstractmethod
    def mean(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any: ...

    # Maximum over the given axis (or all elements)
    @abstractmethod
    def max(self, a: Any, axis: Any = None, keepdims: bool = False) -> Any: ...

    # Returns the array reshaped
    @abstractmethod
    def reshape(self, a: Any, shape: tuple[int, ...]) -> Any: ...

    # Returns the array with permuted axes
    @abstractmethod
    def transpose(self, a: Any, axes: tuple[int, ...] | None = None) -> Any: ...

    # Broadcasts the array to a new shape
    @abstractmethod
    def broadcast_to(self, a: Any, shape: tuple[int, ...]) -> Any: ...

    # Extracts sliding windows from a zero-padded (N, C, H, W) array for Conv2D
    @abstractmethod
    def im2col(
        self,
        a: Any,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> Any: ...

    # Scatter-adds columns back to (N, C, H, W), the adjoint of im2col
    @abstractmethod
    def col2im(
        self,
        cols: Any,
        input_shape: tuple[int, int, int, int],
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> Any: ...
