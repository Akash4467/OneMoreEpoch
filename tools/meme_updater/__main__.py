import sys
from pathlib import Path

from tools.meme_updater.pipeline import run

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT = (
    _REPO_ROOT / "src" / "onemoreepoch" / "messages" / "memes" / "data" / "catalog.json"
)


# Runs the meme pipeline and publishes to the given path (or the default catalog location)
def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUTPUT
    path = run(output)
    print(f"Published meme catalog to {path}")


if __name__ == "__main__":
    main()
