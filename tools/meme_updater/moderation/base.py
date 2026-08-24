"""MemeModerator: approve or reject a classified candidate."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tools.meme_updater.models import CandidateMeme


@dataclass(frozen=True)
class ModerationResult:
    approved: bool
    reason: str | None = None


class MemeModerator(ABC):
    @abstractmethod
    def moderate(self, candidate: CandidateMeme) -> ModerationResult: ...
