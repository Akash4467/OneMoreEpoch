from onemoreepoch.messages.memes.catalog import load_active_catalog


# Picks a meme for a (category, mode) pair, rotating deterministically
class MemeSelector:
    # Loads the active catalog and initializes rotation counters
    def __init__(self) -> None:
        self._catalog = load_active_catalog()
        self._counters: dict[tuple[str, str], int] = {}

    # Returns a meme's text for (category, mode), or None if none match
    def select(self, category: str, mode: str) -> str | None:
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
