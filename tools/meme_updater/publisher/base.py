"""MemePublisher: persist a finished catalog."""

from abc import ABC, abstractmethod
from pathlib import Path

from onemoreepoch.messages.memes.models import MemeCatalog


class MemePublisher(ABC):
    @abstractmethod
    def publish(self, catalog: MemeCatalog) -> Path: ...
