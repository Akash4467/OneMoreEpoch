from dataclasses import dataclass

CATEGORIES = frozenset(
    {
        "shape_error",
        "dtype_error",
        "device_error",
        "gradient_error",
        "nan_error",
        "out_of_memory",
        "training_complete",
        "epoch_complete",
        "overfitting",
        "underfitting",
        "early_stopping",
        "checkpoint_saved",
        "optimizer_error",
        "general_success",
        "general_warning",
    }
)

MODES = frozenset({"classic", "hindi", "roast"})


# One catalog entry: a short piece of text tagged by mode and category
@dataclass(frozen=True)
class Meme:
    id: str
    text: str
    mode: str
    categories: tuple[str, ...]
    quality_score: float = 0.5
    source: str = "seed:local"
    created_at: str = ""


# A versioned collection of memes
@dataclass(frozen=True)
class MemeCatalog:
    schema_version: int
    catalog_version: str
    generated_at: str
    memes: tuple[Meme, ...]
