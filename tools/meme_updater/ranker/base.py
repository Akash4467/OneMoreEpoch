from abc import ABC, abstractmethod

from tools.meme_updater.models import CandidateMeme


# Scores and orders approved candidates for publishing
class MemeRanker(ABC):
    # Returns the candidates ranked/deduplicated for publishing
    @abstractmethod
    def rank(self, candidates: list[CandidateMeme]) -> list[CandidateMeme]: ...
