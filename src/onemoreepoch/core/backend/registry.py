"""Backend registration and lookup."""

import importlib
import os

from onemoreepoch.core.backend import factory as _factory
from onemoreepoch.core.backend.base import Backend
from onemoreepoch.core.backend.numpy_backend import NumPyBackend
from onemoreepoch.exceptions import BackendError

_BACKENDS: dict[str, Backend] = {}
# ONEMOREEPOCH_BACKEND opts into a non-default backend (e.g. "rust") without
# code changes, mirroring config.py's ONEMOREEPOCH_MESSAGES pattern. Not
# validated here — an invalid/unbuilt choice surfaces its real error the
# first time get_backend() actually resolves it, not at import time.
_DEFAULT_BACKEND = os.environ.get("ONEMOREEPOCH_BACKEND", "").strip() or "numpy"


def register_backend(backend: Backend) -> None:
    """Register a backend by name."""
    _BACKENDS[backend.name] = backend


def get_backend(name: str | None = None) -> Backend:
    """Return a registered backend, defaulting to NumPy.

    If ``name`` isn't a known instance or factory yet, this makes one
    lazy attempt to import ``core.backend.<name>_backend`` — the
    convention optional backends (e.g. ``rust_backend``) follow to
    self-register a factory as a side effect of being imported, without
    forcing that import (and whatever it might fail to build) at
    package-import time.
    """
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
