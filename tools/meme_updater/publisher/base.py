from abc import ABC, abstractmethod
from pathlib import Path

from onemoreepoch.messages.memes.models import MemeCatalog


# Persists a finished catalog somewhere durable
class MemePublisher(ABC):
    # Publishes the catalog and returns the path it was written to
    @abstractmethod
    def publish(self, catalog: MemeCatalog) -> Path: ...
