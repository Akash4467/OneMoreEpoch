"""MemeClassifier: assigns doc §20 categories to a candidate."""

from abc import ABC, abstractmethod

from tools.meme_updater.models import CandidateMeme


class MemeClassifier(ABC):
    """Returns a new CandidateMeme with ``categories`` populated."""

    @abstractmethod
    def classify(self, candidate: CandidateMeme) -> CandidateMeme: ...
