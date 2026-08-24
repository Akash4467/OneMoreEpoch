"""Reads candidate text from bundled, self-authored JSON seed files."""

import json
from pathlib import Path
from typing import Iterable

from tools.meme_updater.collector.base import MemeCollector
from tools.meme_updater.models import CandidateMeme

SEEDS_DIR = Path(__file__).parent / "seeds"


class LocalSeedCollector(MemeCollector):
    """Loads every ``*_seeds.json`` file in ``collector/seeds/``."""

    def __init__(self, seeds_dir: Path = SEEDS_DIR) -> None:
        self.seeds_dir = seeds_dir

    def collect(self) -> Iterable[CandidateMeme]:
        for seed_file in sorted(self.seeds_dir.glob("*_seeds.json")):
            entries = json.loads(seed_file.read_text(encoding="utf-8"))
            for entry in entries:
                yield CandidateMeme(
                    text=entry["text"],
                    mode=entry["mode"],
                    # Optional: a seed file may pre-assign categories for
                    # lines too oblique for keyword matching to catch.
                    categories=tuple(entry.get("categories", ())),
                    source=f"seed:{seed_file.name}",
                )
