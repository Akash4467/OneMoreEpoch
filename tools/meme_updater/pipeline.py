"""Orchestrates collector -> classifier -> moderator -> ranker -> publisher.

Incremental, not destructive: the existing catalog at ``output_path``
(if any) is loaded and merged with newly approved candidates before
publishing, so running the updater never discards previously curated
content — only adds to it and drops near-duplicates.
"""

from datetime import datetime, timezone
from pathlib import Path

from onemoreepoch.messages.memes.catalog import load_active_catalog
from onemoreepoch.messages.memes.models import Meme, MemeCatalog
from tools.meme_updater.classifier.heuristic import HeuristicClassifier
from tools.meme_updater.collector.local_seed import LocalSeedCollector
from tools.meme_updater.models import CandidateMeme
from tools.meme_updater.moderation.rule_based import RuleBasedModerator
from tools.meme_updater.publisher.local_file import LocalFilePublisher
from tools.meme_updater.ranker.heuristic import HeuristicRanker


def run(output_path: Path) -> Path:
    """Run the full pipeline once and publish the resulting catalog."""
    now = datetime.now(timezone.utc)

    existing = load_active_catalog(output_path if output_path.exists() else None)
    existing_candidates = [
        CandidateMeme(
            text=meme.text,
            mode=meme.mode,
            categories=meme.categories,
            source=meme.source,
            created_at=meme.created_at,
            quality_score=meme.quality_score,
        )
        for meme in (existing.memes if existing else ())
    ]

    classifier = HeuristicClassifier()
    new_candidates = [classifier.classify(c) for c in LocalSeedCollector().collect()]

    moderator = RuleBasedModerator()
    approved_new = [c for c in new_candidates if moderator.moderate(c).approved]

    combined = HeuristicRanker().rank(existing_candidates + approved_new)

    memes = tuple(
        Meme(
            id=f"catalog-{index:04d}",
            text=candidate.text,
            mode=candidate.mode,
            categories=candidate.categories,
            quality_score=candidate.quality_score,
            source=candidate.source,
            created_at=candidate.created_at or now.isoformat(),
        )
        for index, candidate in enumerate(combined)
    )
    catalog = MemeCatalog(
        schema_version=1,
        catalog_version=now.strftime("%Y.%m"),
        generated_at=now.isoformat(),
        memes=memes,
    )
    return LocalFilePublisher(output_path).publish(catalog)
