"""Crisis dummies, episode tables and shaded bands (chapter 10)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def episodes(dummy: pd.Series) -> pd.DataFrame:
    """One row per run of ones: ``start``, ``end``, ``length`` and ``peak``.

    ``peak`` is the period before ``start`` (the last one before the
    recession). NaN counts as 0.
    """
    d = dummy.fillna(0).astype(int)
    starts = d.index[(d == 1) & (d.shift(1, fill_value=0) == 0)]
    ends = d.index[(d == 1) & (d.shift(-1, fill_value=0) == 0)]
    positions = [d.index.get_loc(s) for s in starts]
    return pd.DataFrame(
        {
            "start": starts,
            "end": ends,
            "length": [
                d.index.get_loc(e) - d.index.get_loc(s) + 1
                for s, e in zip(starts, ends, strict=True)
            ],
            "peak": [d.index[p - 1] if p > 0 else pd.NaT for p in positions],
        }
    )


def dummy_from_episodes(index: pd.Index, spans: Iterable[tuple[Any, Any]]) -> pd.Series:
    """0/1 series on ``index``, 1 inside each inclusive (start, end) span."""
    out = pd.Series(0, index=index)
    for start, end in spans:
        out.loc[start:end] = 1
    return out


def episode_labels(dummy: pd.Series, names: list[str] | None = None) -> pd.Series:
    """Label each period by its episode name (NaN outside episodes).

    Default names are ``"1983-84"``-style year spans. Feed the result to
    ``pd.get_dummies`` for one dummy per episode, or to ``groupby``.
    """
    ep = episodes(dummy)
    if names is None:
        names = []
        for s, e in zip(ep["start"], ep["end"], strict=True):
            names.append(
                str(s.year) if s.year == e.year else f"{s.year}-{str(e.year)[2:]}"
            )
    label = pd.Series(pd.NA, index=dummy.index, dtype=object)
    for name, (s, e) in zip(
        names, zip(ep["start"], ep["end"], strict=True), strict=True
    ):
        label.loc[s:e] = name
    return label


def _period(index: pd.Index, at: Any) -> Any:
    return (
        pd.Period(at, index.freq)
        if isinstance(index, pd.PeriodIndex)
        else pd.Timestamp(at)
    )


def pulse(index: pd.Index, at: Any) -> pd.Series:
    """Additive outlier: 1 at ``at``, 0 elsewhere."""
    return pd.Series((index == _period(index, at)).astype(float), index=index)


def step(index: pd.Index, at: Any) -> pd.Series:
    """Level shift: 0 before ``at``, 1 from ``at`` on."""
    return pd.Series((index >= _period(index, at)).astype(float), index=index)


def ramp(index: pd.Index, at: Any) -> pd.Series:
    """Trend change: 0 before ``at``, then 1, 2, 3, ..."""
    return step(index, at).cumsum()


def temporary_change(index: pd.Index, at: Any, delta: float = 0.7) -> pd.Series:
    """Geometrically decaying shock: 1, delta, delta^2, ... from ``at``."""
    s = step(index, at)
    return (delta ** (s.cumsum() - 1)).where(s == 1, 0.0)


def shade(ax: Any, dummy: pd.Series, **style: Any) -> None:
    """Shade the periods where ``dummy`` == 1 on a matplotlib axis.

    Each band covers whole periods (start of the first to the end of the
    last), so a quarterly dummy shades a monthly or daily chart correctly.
    ``style`` goes to ``ax.axvspan``; the default is a light grey band. For
    a hatched band pass e.g. ``fc="none", ec="C3", hatch="///"``.
    """
    if not {"color", "fc", "facecolor"} & style.keys():
        style["color"] = "0.85"
    style = {"lw": 0, "zorder": 0, **style}
    for _, row in episodes(dummy).iterrows():
        start, end = row["start"], row["end"]
        if isinstance(start, pd.Period):
            start, end = start.start_time, end.end_time
        ax.axvspan(start, end, **style)
        style.pop("label", None)
