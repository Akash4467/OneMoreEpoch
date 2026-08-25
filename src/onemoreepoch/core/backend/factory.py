from collections.abc import Callable

from onemoreepoch.core.backend.base import Backend

_FACTORIES: dict[str, Callable[[], Backend]] = {}


# Registers a zero-arg callable that builds the named backend on demand
def register_backend_factory(name: str, factory: Callable[[], Backend]) -> None:
    _FACTORIES[name] = factory


# Builds the backend registered for name, or returns None if none is registered
def create_backend(name: str) -> Backend | None:
    factory = _FACTORIES.get(name)
    return factory() if factory is not None else None


# Returns the names with a registered (not necessarily buildable) factory
def available_factory_names() -> list[str]:
    return sorted(_FACTORIES)
