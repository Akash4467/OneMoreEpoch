"""Scores by a length heuristic, then drops near-duplicate text."""

import difflib

from tools.meme_updater.models import CandidateMeme
from tools.meme_updater.ranker.base import MemeRanker

_TARGET_LENGTH = 80  # short enough to read on one terminal line
_DEDUP_THRESHOLD = 0.9


class HeuristicRanker(MemeRanker):
    def rank(self, candidates: list[CandidateMeme]) -> list[CandidateMeme]:
        scored = [self._scored(c) for c in candidates]
        scored.sort(key=lambda c: c.quality_score, reverse=True)

        kept: list[CandidateMeme] = []
        for candidate in scored:
            if not any(self._similar(candidate, other) for other in kept):
                kept.append(candidate)
        return kept

    @staticmethod
    def _scored(candidate: CandidateMeme) -> CandidateMeme:
        distance = abs(len(candidate.text) - _TARGET_LENGTH)
        candidate.quality_score = max(0.0, 1.0 - distance / 200)
        return candidate

    @staticmethod
    def _similar(a: CandidateMeme, b: CandidateMeme) -> bool:
        if a.mode != b.mode:
            return False
        return difflib.SequenceMatcher(None, a.text, b.text).ratio() > _DEDUP_THRESHOLD
