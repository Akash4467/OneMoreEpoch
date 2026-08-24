"""CandidateMeme: the mutable, in-progress shape a candidate has while
moving through the pipeline, before it becomes a frozen ``Meme``
(onemoreepoch.messages.memes.models) at publish time.
"""

from dataclasses import dataclass


@dataclass
class CandidateMeme:
    """A meme candidate as it moves through collect -> classify -> moderate -> rank."""

    text: str
    mode: str
    categories: tuple[str, ...] = ()
    source: str = "seed:local"
    created_at: str = ""
    quality_score: float = 0.0
