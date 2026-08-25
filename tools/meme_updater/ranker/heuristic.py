import difflib

from tools.meme_updater.models import CandidateMeme
from tools.meme_updater.ranker.base import MemeRanker

_TARGET_LENGTH = 80
_DEDUP_THRESHOLD = 0.9


# Scores candidates by a length heuristic, then drops near-duplicate text
class HeuristicRanker(MemeRanker):
    # Scores every candidate, sorts by score, and drops near-duplicates
    def rank(self, candidates: list[CandidateMeme]) -> list[CandidateMeme]:
        scored = [self._scored(c) for c in candidates]
        scored.sort(key=lambda c: c.quality_score, reverse=True)

        kept: list[CandidateMeme] = []
        for candidate in scored:
            if not any(self._similar(candidate, other) for other in kept):
                kept.append(candidate)
        return kept

    # Sets and returns the candidate's quality_score based on distance from the target length
    @staticmethod
    def _scored(candidate: CandidateMeme) -> CandidateMeme:
        distance = abs(len(candidate.text) - _TARGET_LENGTH)
        candidate.quality_score = max(0.0, 1.0 - distance / 200)
        return candidate

    # Returns True if a and b are near-duplicate text in the same mode
    @staticmethod
    def _similar(a: CandidateMeme, b: CandidateMeme) -> bool:
        if a.mode != b.mode:
            return False
        return difflib.SequenceMatcher(None, a.text, b.text).ratio() > _DEDUP_THRESHOLD
