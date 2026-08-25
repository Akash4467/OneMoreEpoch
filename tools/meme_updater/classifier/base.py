from abc import ABC, abstractmethod

from tools.meme_updater.models import CandidateMeme


# Assigns categories to a candidate; mode is trusted from the collector
class MemeClassifier(ABC):
    # Returns a new CandidateMeme with categories populated
    @abstractmethod
    def classify(self, candidate: CandidateMeme) -> CandidateMeme: ...
