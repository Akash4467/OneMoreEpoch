"""Backend factory: lazily builds optional backends by name.

Separates "a backend named X exists" from "construct an X" (Factory
pattern, doc §30). This matters for backends that may not be usable in
every environment — e.g. the Rust backend, whose compiled extension
might not be built — so registering *how* to build one must not force
building it, or even importing the module that would build it, during
``import onemoreepoch``.
"""

from collections.abc import Callable

from onemoreepoch.core.backend.base import Backend

_FACTORIES: dict[str, Callable[[], Backend]] = {}


def register_backend_factory(name: str, factory: Callable[[], Backend]) -> None:
    """Register a zero-arg callable that builds the named backend on demand."""
    _FACTORIES[name] = factory


def create_backend(name: str) -> Backend | None:
    """Build the backend registered for ``name``, or None if none is registered."""
    factory = _FACTORIES.get(name)
    return factory() if factory is not None else None


def available_factory_names() -> list[str]:
    """Names with a registered (not necessarily buildable) factory."""
    return sorted(_FACTORIES)
