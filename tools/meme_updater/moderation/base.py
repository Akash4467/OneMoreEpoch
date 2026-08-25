from abc import ABC, abstractmethod
from dataclasses import dataclass

from tools.meme_updater.models import CandidateMeme


# Approval decision for a candidate, with an optional rejection reason
@dataclass(frozen=True)
class ModerationResult:
    approved: bool
    reason: str | None = None


# Decides whether a classified candidate may be published
class MemeModerator(ABC):
    # Returns a ModerationResult for the candidate
    @abstractmethod
    def moderate(self, candidate: CandidateMeme) -> ModerationResult: ...
