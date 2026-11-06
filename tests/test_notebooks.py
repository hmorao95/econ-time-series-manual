"""Every notebook of the book runs top to bottom, with warnings raised as errors."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "execute_notebooks", ROOT / "scripts" / "execute_notebooks.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("path", runner.NOTEBOOKS, ids=lambda p: p.parent.name)
def test_notebook_runs(path):
    runner.execute(path, save=False, strict=True)
