import importlib
import os

from onemoreepoch.core.backend import factory as _factory
from onemoreepoch.core.backend.base import Backend
from onemoreepoch.core.backend.numpy_backend import NumPyBackend
from onemoreepoch.exceptions import BackendError

_BACKENDS: dict[str, Backend] = {}
_DEFAULT_BACKEND = os.environ.get("ONEMOREEPOCH_BACKEND", "").strip() or "numpy"


# Registers a backend instance by its name
def register_backend(backend: Backend) -> None:
    _BACKENDS[backend.name] = backend


# Returns a registered backend by name (defaulting to NumPy), lazily importing/building it if needed
def get_backend(name: str | None = None) -> Backend:
    backend_name = name or _DEFAULT_BACKEND
    if (
        backend_name not in _BACKENDS
        and backend_name not in _factory.available_factory_names()
    ):
        try:
            importlib.import_module(f"onemoreepoch.core.backend.{backend_name}_backend")
        except ImportError:
            pass
    if backend_name not in _BACKENDS:
        built = _factory.create_backend(backend_name)
        if built is not None:
            _BACKENDS[backend_name] = built
    if backend_name not in _BACKENDS:
        raise BackendError(
            "unknown_backend",
            name=backend_name,
            available=sorted(set(_BACKENDS) | set(_factory.available_factory_names())),
        )
    return _BACKENDS[backend_name]


register_backend(NumPyBackend())
