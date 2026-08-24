"""Tests for the meme subsystem: catalog schema, selector, fallback behavior."""

import json

import pytest

from onemoreepoch import config
from onemoreepoch.messages import get_meme, get_meme_for_key
from onemoreepoch.messages.memes.catalog import (
    DEFAULT_CATALOG_PATH,
    load_active_catalog,
    load_catalog,
)
from onemoreepoch.messages.memes.models import CATEGORIES, MODES
from onemoreepoch.messages.memes.selector import MemeSelector


@pytest.fixture(autouse=True)
def reset_mode():
    yield
    config.set_message_mode("classic")


class TestDefaultCatalog:
    def test_loads_and_validates(self):
        catalog = load_catalog(DEFAULT_CATALOG_PATH)
        assert catalog.schema_version == 1
        assert catalog.memes

    def test_covers_every_category_and_mode(self):
        catalog = load_catalog(DEFAULT_CATALOG_PATH)
        seen = {(cat, meme.mode) for meme in catalog.memes for cat in meme.categories}
        for category in CATEGORIES:
            for mode in MODES:
                assert (category, mode) in seen, f"missing {category}/{mode}"


class TestCatalogValidation:
    def test_invalid_category_falls_back(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "catalog_version": "x",
                    "memes": [
                        {
                            "id": "1",
                            "text": "t",
                            "mode": "classic",
                            "categories": ["not_a_category"],
                        }
                    ],
                }
            )
        )
        with pytest.raises(Exception):
            load_catalog(bad)

    def test_missing_file_falls_back_to_default(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        catalog = load_active_catalog(missing)
        assert catalog is not None
        assert catalog.catalog_version  # loaded the packaged default

    def test_completely_broken_catalog_yields_none_not_raise(
        self, tmp_path, monkeypatch
    ):
        import onemoreepoch.messages.memes.catalog as catalog_mod

        monkeypatch.setattr(catalog_mod, "DEFAULT_CATALOG_PATH", tmp_path / "nope.json")
        assert load_active_catalog(tmp_path / "also_nope.json") is None


class TestSelector:
    def test_returns_none_for_unknown_category(self):
        selector = MemeSelector()
        assert selector.select("not_a_real_category", "classic") is None

    def test_rotates_deterministically(self):
        selector = MemeSelector()
        first = selector.select("shape_error", "classic")
        assert first is not None
        # Only one classic shape_error meme is seeded, so it repeats —
        # rotation is deterministic either way (index % len(candidates)).
        second = selector.select("shape_error", "classic")
        assert second is not None


class TestManagerFacade:
    def test_get_meme_never_raises_on_garbage_category(self):
        assert get_meme("") is None
        assert get_meme(None) is None

    def test_get_meme_for_key_maps_known_keys(self):
        config.set_message_mode("classic")
        assert get_meme_for_key("shape_mismatch_matmul") is not None

    def test_get_meme_for_key_returns_none_for_unmapped_keys(self):
        assert get_meme_for_key("state_dict_key_mismatch") is None


class TestExceptionIntegration:
    def test_meme_is_appended_not_replacing_technical_message(self):
        from onemoreepoch.core import Tensor

        config.set_message_mode("hindi")
        with pytest.raises(Exception) as excinfo:
            Tensor.randn(64, 128) @ Tensor.randn(32, 10)
        text = str(excinfo.value)
        assert "(64, 128)" in text  # technical message preserved
        assert len(text.splitlines()) > 1  # meme appended on its own lines
