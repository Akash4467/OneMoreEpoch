"""Simple blocklist + length + category-presence moderation."""

from tools.meme_updater.models import CandidateMeme
from tools.meme_updater.moderation.base import MemeModerator, ModerationResult

# Not exhaustive — a real deployment would want something more robust.
_BLOCKLIST = ("http://", "https://", "<script")
_MAX_LENGTH = 200


class RuleBasedModerator(MemeModerator):
    def moderate(self, candidate: CandidateMeme) -> ModerationResult:
        if not candidate.categories:
            return ModerationResult(False, "no category assigned")
        if len(candidate.text) > _MAX_LENGTH:
            return ModerationResult(False, "text too long")
        lowered = candidate.text.lower()
        if any(bad in lowered for bad in _BLOCKLIST):
            return ModerationResult(False, "blocklisted content")
        return ModerationResult(True, None)
