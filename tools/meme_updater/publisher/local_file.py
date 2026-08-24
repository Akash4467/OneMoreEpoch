"""Writes the catalog to a local JSON file, atomically and validated.

Reuses ``messages.memes.catalog.load_catalog`` — the runtime's own
validator — as the single source of truth for "valid", instead of
duplicating validation rules here.
"""

import dataclasses
import json
import os
from pathlib import Path

from onemoreepoch.messages.memes.catalog import load_catalog
from onemoreepoch.messages.memes.models import MemeCatalog
from tools.meme_updater.publisher.base import MemePublisher


class LocalFilePublisher(MemePublisher):
    def __init__(self, output_path: Path) -> None:
        self.output_path = Path(output_path)

    def publish(self, catalog: MemeCatalog) -> Path:
        payload = {
            "schema_version": catalog.schema_version,
            "catalog_version": catalog.catalog_version,
            "generated_at": catalog.generated_at,
            "memes": [dataclasses.asdict(meme) for meme in catalog.memes],
        }
        tmp_path = self.output_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        load_catalog(
            tmp_path
        )  # round-trip through the real validator before swapping in
        os.replace(tmp_path, self.output_path)
        return self.output_path
