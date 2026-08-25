import json
from collections.abc import Iterable
from pathlib import Path

from tools.meme_updater.collector.base import MemeCollector
from tools.meme_updater.models import CandidateMeme

SEEDS_DIR = Path(__file__).parent / "seeds"


# Reads candidate text from bundled, self-authored JSON seed files
class LocalSeedCollector(MemeCollector):
    # Stores the directory to scan for seed files
    def __init__(self, seeds_dir: Path = SEEDS_DIR) -> None:
        self.seeds_dir = seeds_dir

    # Yields a CandidateMeme for every entry in every *_seeds.json file
    def collect(self) -> Iterable[CandidateMeme]:
        for seed_file in sorted(self.seeds_dir.glob("*_seeds.json")):
            entries = json.loads(seed_file.read_text(encoding="utf-8"))
            for entry in entries:
                yield CandidateMeme(
                    text=entry["text"],
                    mode=entry["mode"],
                    categories=tuple(entry.get("categories", ())),
                    source=f"seed:{seed_file.name}",
                )
