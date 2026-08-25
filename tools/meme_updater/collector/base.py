from abc import ABC, abstractmethod
from collections.abc import Iterable

from tools.meme_updater.models import CandidateMeme


# Pluggable source-of-candidates interface for the meme pipeline
class MemeCollector(ABC):
    # Returns raw candidates, not yet classified/moderated/ranked
    @abstractmethod
    def collect(self) -> Iterable[CandidateMeme]: ...
