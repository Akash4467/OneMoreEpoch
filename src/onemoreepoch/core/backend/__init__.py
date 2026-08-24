"""Backend abstraction for array operations."""

from onemoreepoch.core.backend.base import Backend
from onemoreepoch.core.backend.factory import register_backend_factory
from onemoreepoch.core.backend.numpy_backend import NumPyBackend
from onemoreepoch.core.backend.registry import get_backend, register_backend

__all__ = [
    "Backend",
    "NumPyBackend",
    "get_backend",
    "register_backend",
    "register_backend_factory",
]
