"""Keyword -> category classifier. Trusts the collector-supplied mode
and, when the collector already assigned categories (a seed file can do
this for lines too oblique for keyword matching), trusts those too."""

from dataclasses import replace

from tools.meme_updater.classifier.base import MemeClassifier
from tools.meme_updater.models import CandidateMeme

# Substring (lowercase) -> category. A candidate can match several.
_KEYWORD_CATEGORIES: dict[str, str] = {
    "checkpoint": "checkpoint_saved",
    "gradient": "gradient_error",
    "exploded": "gradient_error",
    "early stopping": "early_stopping",
    "overfit": "overfitting",
    "underfit": "underfitting",
    "memory": "out_of_memory",
    "nan": "nan_error",
    "optimizer": "optimizer_error",
    "diverged": "optimizer_error",
    "shape": "shape_error",
    "dtype": "dtype_error",
    "device": "device_error",
    "epoch": "epoch_complete",
    "training complete": "training_complete",
    "training done": "training_complete",
}


class HeuristicClassifier(MemeClassifier):
    """Scans the text for known keywords and collects their categories."""

    def classify(self, candidate: CandidateMeme) -> CandidateMeme:
        if candidate.categories:
            return candidate
        lowered = candidate.text.lower()
        categories: list[str] = []
        for keyword, category in _KEYWORD_CATEGORIES.items():
            if keyword in lowered and category not in categories:
                categories.append(category)
        return replace(candidate, categories=tuple(categories))
