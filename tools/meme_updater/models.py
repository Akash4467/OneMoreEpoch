from dataclasses import dataclass


# A meme candidate as it moves through collect -> classify -> moderate -> rank
@dataclass
class CandidateMeme:
    text: str
    mode: str
    categories: tuple[str, ...] = ()
    source: str = "seed:local"
    created_at: str = ""
    quality_score: float = 0.0
