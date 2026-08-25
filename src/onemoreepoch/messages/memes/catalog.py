import json
from pathlib import Path
from typing import Any

from onemoreepoch.messages.memes.models import CATEGORIES, MODES, Meme, MemeCatalog

DEFAULT_CATALOG_PATH = Path(__file__).parent / "data" / "catalog.json"


# Internal validation error, never escapes this module
class _CatalogValidationError(Exception):
    pass


# Validates a raw parsed-JSON dict and converts it into a MemeCatalog
def _validate(raw: dict[str, Any]) -> MemeCatalog:
    try:
        memes = []
        for entry in raw["memes"]:
            categories = tuple(entry["categories"])
            if not categories or not set(categories) <= CATEGORIES:
                raise _CatalogValidationError(f"invalid categories: {categories!r}")
            if entry["mode"] not in MODES:
                raise _CatalogValidationError(f"invalid mode: {entry['mode']!r}")
            memes.append(
                Meme(
                    id=str(entry["id"]),
                    text=str(entry["text"]),
                    mode=entry["mode"],
                    categories=categories,
                    quality_score=float(entry.get("quality_score", 0.5)),
                    source=str(entry.get("source", "seed:local")),
                    created_at=str(entry.get("created_at", "")),
                )
            )
        return MemeCatalog(
            schema_version=int(raw["schema_version"]),
            catalog_version=str(raw["catalog_version"]),
            generated_at=str(raw.get("generated_at", "")),
            memes=tuple(memes),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _CatalogValidationError(str(exc)) from exc


# Loads and validates a catalog file, raising on any problem
def load_catalog(path: Path) -> MemeCatalog:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _validate(raw)


# Tries an updated catalog then the packaged default, returning None instead of raising
def load_active_catalog(updated_path: Path | None = None) -> MemeCatalog | None:
    for candidate in (updated_path, DEFAULT_CATALOG_PATH):
        if candidate is None:
            continue
        try:
            return load_catalog(candidate)
        except Exception:  # noqa: BLE001, S112
            continue
    return None
