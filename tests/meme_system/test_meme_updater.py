import json

from tools.meme_updater.classifier.heuristic import HeuristicClassifier
from tools.meme_updater.collector.local_seed import LocalSeedCollector
from tools.meme_updater.models import CandidateMeme
from tools.meme_updater.moderation.rule_based import RuleBasedModerator
from tools.meme_updater.pipeline import run
from tools.meme_updater.ranker.heuristic import HeuristicRanker


# Tests LocalSeedCollector
class TestLocalSeedCollector:
    # Checks every seed file is collected and modes are all represented
    def test_collects_every_seed_file(self):
        candidates = list(LocalSeedCollector().collect())
        modes = {c.mode for c in candidates}
        assert modes == {"classic", "hindi", "roast"}
        assert len(candidates) == 9 + 61


# Tests HeuristicClassifier
class TestHeuristicClassifier:
    # Checks keyword matches assign the correct categories
    def test_assigns_categories_from_keywords(self):
        classifier = HeuristicClassifier()
        candidate = CandidateMeme(
            text="Checkpoint saved after the gradient exploded.", mode="classic"
        )
        classified = classifier.classify(candidate)
        assert "checkpoint_saved" in classified.categories
        assert "gradient_error" in classified.categories

    # Checks pre-assigned categories are trusted and not overwritten
    def test_trusts_collector_supplied_categories(self):
        classifier = HeuristicClassifier()
        candidate = CandidateMeme(
            text="Model tota ban gaya, bas rata maar raha hai.",
            mode="hindi",
            categories=("overfitting",),
        )
        classified = classifier.classify(candidate)
        assert classified.categories == ("overfitting",)

    # Checks text with no keyword matches yields empty categories
    def test_no_keyword_match_yields_empty_categories(self):
        classifier = HeuristicClassifier()
        classified = classifier.classify(
            CandidateMeme(text="hello world", mode="classic")
        )
        assert classified.categories == ()


# Tests RuleBasedModerator
class TestRuleBasedModerator:
    # Checks an uncategorized candidate is rejected
    def test_rejects_uncategorized(self):
        result = RuleBasedModerator().moderate(CandidateMeme(text="hi", mode="classic"))
        assert not result.approved

    # Checks a candidate containing a URL is rejected
    def test_rejects_urls(self):
        candidate = CandidateMeme(
            text="checkpoint at http://example.com",
            mode="classic",
            categories=("checkpoint_saved",),
        )
        assert not RuleBasedModerator().moderate(candidate).approved

    # Checks clean, categorized text is approved
    def test_approves_clean_categorized_text(self):
        candidate = CandidateMeme(
            text="checkpoint saved", mode="classic", categories=("checkpoint_saved",)
        )
        assert RuleBasedModerator().moderate(candidate).approved


# Tests HeuristicRanker
class TestHeuristicRanker:
    # Checks near-duplicate text in the same mode is deduplicated
    def test_drops_near_duplicates_within_same_mode(self):
        candidates = [
            CandidateMeme(
                text="checkpoint saved successfully",
                mode="classic",
                categories=("checkpoint_saved",),
            ),
            CandidateMeme(
                text="checkpoint saved successfully!",
                mode="classic",
                categories=("checkpoint_saved",),
            ),
        ]
        ranked = HeuristicRanker().rank(candidates)
        assert len(ranked) == 1

    # Checks identical text in different modes is kept, not deduplicated
    def test_keeps_similar_text_across_different_modes(self):
        candidates = [
            CandidateMeme(
                text="checkpoint saved successfully",
                mode="classic",
                categories=("checkpoint_saved",),
            ),
            CandidateMeme(
                text="checkpoint saved successfully",
                mode="roast",
                categories=("checkpoint_saved",),
            ),
        ]
        ranked = HeuristicRanker().rank(candidates)
        assert len(ranked) == 2


# Tests the full meme_updater pipeline
class TestPipeline:
    # Checks the pipeline publishes a valid, non-trivial catalog
    def test_publishes_a_valid_catalog(self, tmp_path):
        output = tmp_path / "catalog.json"
        published = run(output)
        assert published == output
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1
        assert len(data["memes"]) >= 9

    # Checks running the pipeline twice doesn't duplicate entries
    def test_is_idempotent(self, tmp_path):
        output = tmp_path / "catalog.json"
        run(output)
        first_count = len(json.loads(output.read_text(encoding="utf-8"))["memes"])
        run(output)
        second_count = len(json.loads(output.read_text(encoding="utf-8"))["memes"])
        assert first_count == second_count

    # Checks running the pipeline preserves hand-written entries already in the catalog
    def test_merges_with_existing_catalog_without_losing_entries(self, tmp_path):
        output = tmp_path / "catalog.json"
        existing = {
            "schema_version": 1,
            "catalog_version": "2020.01",
            "generated_at": "",
            "memes": [
                {
                    "id": "custom-001",
                    "text": "A hand-written meme nothing else will produce.",
                    "mode": "classic",
                    "categories": ["general_success"],
                    "quality_score": 0.9,
                    "source": "manual",
                    "created_at": "",
                }
            ],
        }
        output.write_text(json.dumps(existing), encoding="utf-8")
        run(output)
        data = json.loads(output.read_text(encoding="utf-8"))
        texts = {meme["text"] for meme in data["memes"]}
        assert "A hand-written meme nothing else will produce." in texts
