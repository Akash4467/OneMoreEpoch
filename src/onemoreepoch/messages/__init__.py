"""Personality-driven messaging for errors and training output.

Thin re-export — the implementation lives in ``manager`` (facade over
classic/hindi/roast text plus the meme subsystem).
"""

from onemoreepoch.messages.manager import (
    get_banter,
    get_meme,
    get_meme_for_key,
    get_message,
)

__all__ = ["get_banter", "get_meme", "get_meme_for_key", "get_message"]
