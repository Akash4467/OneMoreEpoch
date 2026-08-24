"""MemeRanker: score and order approved candidates for publishing."""

from abc import ABC, abstractmethod

from tools.meme_updater.models import CandidateMeme


class MemeRanker(ABC):
    @abstractmethod
    def rank(self, candidates: list[CandidateMeme]) -> list[CandidateMeme]: ...
