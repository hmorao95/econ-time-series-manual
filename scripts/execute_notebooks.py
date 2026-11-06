"""Execute the book's notebooks.

Run ``uv run python scripts/execute_notebooks.py`` to re-execute every
notebook and save its outputs in place (figures are stored in the notebook,
so GitHub, nbviewer and the book show them). Pass notebook paths to run only
those. Cells tagged ``skip-execution`` are not run (the X-13 example).

``execute(path, save=False, strict=True)`` is what the tests call: a fresh
run with warnings raised as errors, nothing written.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("[0-9][0-9]/*.ipynb"))
STRICT = "import warnings\nwarnings.simplefilter('error')"


def _use_this_python() -> None:
    """Make the "python3" kernel run this interpreter, not a global Jupyter install."""
    spec = Path(tempfile.gettempdir()) / "econts-jupyter" / "kernels" / "python3"
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "Python 3",
                "language": "python",
            }
        )
    )
    os.environ["JUPYTER_PATH"] = os.pathsep.join(
        [str(spec.parents[1]), *filter(None, [os.environ.get("JUPYTER_PATH")])]
    )


_use_this_python()


def execute(
    path: Path, save: bool = True, strict: bool = False, timeout: int = 900
) -> None:
    """Run a notebook top to bottom in its own folder; optionally save outputs."""
    nb = nbformat.read(path, as_version=4)
    skipped = {}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and "skip-execution" in cell.metadata.get(
            "tags", []
        ):
            skipped[i] = cell.source
            cell.source = "# (not executed: needs an external program)"
    if strict:
        nb.cells.insert(0, nbformat.v4.new_code_cell(STRICT))
    NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    if strict:
        nb.cells.pop(0)
    for i, source in skipped.items():
        nb.cells[i].source = source
        nb.cells[i].outputs = []
        nb.cells[i].execution_count = None
    if save:
        nbformat.write(nb, path)


def main() -> None:
    """Execute the notebooks given on the command line, or all of them."""
    paths = [Path(p).resolve() for p in sys.argv[1:]] or NOTEBOOKS
    for path in paths:
        execute(path)
        print("executed", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
