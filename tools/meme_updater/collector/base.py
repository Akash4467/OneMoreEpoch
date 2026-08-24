"""MemeCollector: the pluggable source-of-candidates interface.

Only ``LocalSeedCollector`` is implemented — it never touches the
network. A real network-backed collector (e.g. pulling from some
external source) would implement this same interface; none is built
here, deliberately (see package docstring).
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from tools.meme_updater.models import CandidateMeme


class MemeCollector(ABC):
    """Produces raw candidates, not yet classified/moderated/ranked."""

    @abstractmethod
    def collect(self) -> Iterable[CandidateMeme]: ...
