"""Every python block in every chapter runs, with warnings as errors."""

import warnings

import pytest

from tsmanual import SERIES, load
from tsmanual.runner import blocks, chapters, run_chapter


@pytest.mark.parametrize("chapter", chapters(), ids=lambda p: p.stem)
def test_chapter_runs(chapter):
    assert blocks(chapter), f"{chapter.name} has no python blocks"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_chapter(chapter)


@pytest.mark.parametrize("series_id", SERIES)
def test_bundled_series_load(series_id):
    s = load(series_id)
    assert s.name == series_id
    assert s.index.is_monotonic_increasing
    assert s.notna().sum() > 100
