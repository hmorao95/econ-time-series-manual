"""Execute the ``python`` code blocks of a chapter, in order, in one namespace.

Conventions inside a chapter:

* Fenced blocks tagged ``python`` run; any other tag (``py``, ``text``, ``bash``)
  does not.
* A block whose first line is ``# requires: <exe>`` is skipped when ``<exe>``
  is not on PATH (used for the X-13ARIMA-SEATS binary).
* A block that contains a line ``# figure: <name>`` saves the current
  matplotlib figure to ``docs/img/<name>.png`` when figures are requested.

Usage: ``uv run python -m tsmanual.runner [--figures] [docs/05-*.md ...]``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
IMG = DOCS / "img"
BLOCK = re.compile(r"^```python[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
FIGURE = re.compile(r"^# figure: (\S+)", re.MULTILINE)
REQUIRES = re.compile(r"\A# requires: (\S+)")

STYLE = {
    "figure.figsize": (8, 3.6),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


def blocks(chapter: Path) -> list[str]:
    """Return the runnable code blocks of a chapter."""
    return BLOCK.findall(chapter.read_text(encoding="utf-8"))


def run_chapter(chapter: Path, figures: bool = False) -> list[str]:
    """Run every block of ``chapter``; return the names of skipped blocks.

    Raises:
        RuntimeError: a block failed; the message names the block.

    """
    namespace: dict[str, object] = {"__name__": "__chapter__"}
    skipped = []
    plt.rcdefaults()  # a chapter may restyle matplotlib (housestyle does)
    plt.rcParams.update(STYLE)
    for i, code in enumerate(blocks(chapter), start=1):
        req = REQUIRES.match(code)
        if req and shutil.which(req.group(1)) is None:
            skipped.append(f"{chapter.name} block {i} (needs {req.group(1)})")
            continue
        try:
            exec(compile(code, f"{chapter.name}:block{i}", "exec"), namespace)
        except Exception as err:
            raise RuntimeError(f"{chapter.name} block {i} failed: {err!r}") from err
        name = FIGURE.search(code)
        if name and figures:
            IMG.mkdir(exist_ok=True)
            for ax in plt.gcf().axes:  # pandas labels the x axis "date"
                if ax.get_xlabel() == "date":
                    ax.set_xlabel("")
            plt.gcf().savefig(IMG / f"{name.group(1)}.png", bbox_inches="tight")
        plt.close("all")
    return skipped


def chapters() -> list[Path]:
    """Return the numbered chapters, in order."""
    return sorted(DOCS.glob("[0-9][0-9]-*.md"))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--figures", action="store_true", help="write docs/img")
    args = parser.parse_args()
    for chapter in args.files or chapters():
        skipped = run_chapter(chapter, figures=args.figures)
        print(f"ok  {chapter.name}" + "".join(f"\n    skipped {s}" for s in skipped))


if __name__ == "__main__":
    sys.exit(main())
