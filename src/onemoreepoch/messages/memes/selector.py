"""MemeSelector: deterministic category+mode lookup into the active catalog."""

from onemoreepoch.messages.memes.catalog import load_active_catalog


class MemeSelector:
    """Picks a meme for a (category, mode) pair, rotating deterministically."""

    def __init__(self) -> None:
        self._catalog = load_active_catalog()
        self._counters: dict[tuple[str, str], int] = {}

    def select(self, category: str, mode: str) -> str | None:
        """Return a meme's text, or None if the catalog/category/mode has none."""
        if self._catalog is None:
            return None
        candidates = [
            meme
            for meme in self._catalog.memes
            if category in meme.categories and meme.mode == mode
        ]
        if not candidates:
            return None
        key = (category, mode)
        index = self._counters.get(key, 0)
        self._counters[key] = index + 1
        return candidates[index % len(candidates)].text
