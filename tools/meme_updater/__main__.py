"""Maintainer/CI entry point: ``python -m tools.meme_updater [output_path]``.

Not an installed console-script — ``tools/`` isn't shipped in the
wheel, so a console-script here would either bundle a maintainer-only
tree into every install or silently fail for end users once installed.
Run from a source checkout instead (locally, or from CI).
"""

import sys
from pathlib import Path

from tools.meme_updater.pipeline import run

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT = (
    _REPO_ROOT / "src" / "onemoreepoch" / "messages" / "memes" / "data" / "catalog.json"
)


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUTPUT
    path = run(output)
    print(f"Published meme catalog to {path}")


if __name__ == "__main__":
    main()
