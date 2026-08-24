"""Meme content pipeline: collector -> classifier -> moderator -> ranker -> publisher.

Maintainer/CI-only — run via ``python -m tools.meme_updater`` from a
source checkout (see ``__main__.py`` for why this isn't an installed
console-script). Every collector here reads self-authored, locally
seeded text, never the live internet — doc §21/§27's "normal execution
must not require internet" extends to this pipeline too; only a real
future network collector implementing ``MemeCollector`` would change
that, and none is built here.
"""
