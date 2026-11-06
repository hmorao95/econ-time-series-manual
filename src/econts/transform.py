"""Growth rates, rebasing, deflating, splicing, demeaning and scaling (chapter 2)."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Literal

import numpy as np
import pandas as pd

GrowthKind = Literal["pop", "yoy", "annualized", "log", "log_annualized"]


def periods_per_year(index: pd.Index) -> int:
    """Return the number of periods per year implied by a Period or Datetime index.

    Raises:
        ValueError: the frequency cannot be determined.

    """
    if isinstance(index, pd.PeriodIndex):
        code = index.freqstr[0]
    else:
        freq = index.freqstr if index.freq is not None else pd.infer_freq(index[:8])
        if freq is None:
            raise ValueError("cannot infer the frequency; use .asfreq() first")
        code = freq.lstrip("0123456789")[0]
    table = {"A": 1, "Y": 1, "Q": 4, "M": 12, "W": 52, "B": 260, "D": 365}
    if code not in table:
        raise ValueError(f"unsupported frequency {code!r}")
    return table[code]


def growth(s: pd.Series, kind: GrowthKind = "pop", ppy: int | None = None) -> pd.Series:
    """Growth rate in percent.

    Args:
        s: series in levels with a regular index.
        kind: ``"pop"`` period on period, ``"yoy"`` year on year,
            ``"annualized"`` compounded period-on-period rate, ``"log"``
            100 x log difference, ``"log_annualized"`` the log difference
            times periods per year.
        ppy: periods per year; inferred from the index when omitted.

    Returns:
        The growth rate in percent. Gaps give NaN, never a filled value.

    """
    ppy = ppy or periods_per_year(s.index)
    if kind == "pop":
        return 100 * s.pct_change(fill_method=None)
    if kind == "yoy":
        return 100 * s.pct_change(ppy, fill_method=None)
    if kind == "annualized":
        return 100 * ((s / s.shift(1)) ** ppy - 1)
    if kind == "log":
        return 100 * np.log(s).diff()
    if kind == "log_annualized":
        return 100 * ppy * np.log(s).diff()
    raise ValueError(f"unknown kind {kind!r}")


def carry_over(q: pd.Series, year: int) -> float:
    """Annual growth in ``year`` if the series stays at its last level of year-1.

    Args:
        q: quarterly (or monthly) series in levels, PeriodIndex or DatetimeIndex.
        year: the year whose growth is "already in the bag".

    """
    prev = q[q.index.year == year - 1]
    return float(100 * (prev.iloc[-1] / prev.mean() - 1))


def rebase(s: pd.Series, base: str | slice, value: float = 100.0) -> pd.Series:
    """Rescale an index so that its average over ``base`` equals ``value``."""
    return value * s / s.loc[base].mean()


def deflate(nominal: pd.Series, prices: pd.Series, base: float = 100.0) -> pd.Series:
    """Real series: nominal divided by a price index with ``base`` in the reference year."""
    return nominal / prices * base


def ratio_splice(
    old: pd.Series, new: pd.Series, at: Hashable | None = None
) -> pd.Series:
    """Rescale ``old`` to the level of ``new`` at ``at`` and join them.

    The growth rates of ``old`` are kept before ``at``; ``new`` is used from
    ``at`` on. ``at`` defaults to the first date of ``new``.
    """
    at = new.index[0] if at is None else at
    factor = new[at] / old[at]
    return pd.concat([old[old.index < at] * factor, new[new.index >= at]])


def demean(s: pd.Series, by: pd.Series | np.ndarray | None = None) -> pd.Series:
    """Subtract the mean, ignoring NaN (EMT ``demean``).

    Args:
        s: the series.
        by: optional group labels aligned with ``s`` (a regime, the calendar
            month, a country): each group is demeaned by its own mean.

    """
    if by is None:
        return s - s.mean()
    return s - s.groupby(by).transform("mean")


def within(
    df: pd.DataFrame, value: str, unit: str, time: str | None = None
) -> pd.Series:
    """Panel within transformation on a long DataFrame.

    Subtracts each unit's mean of ``value``; with ``time`` also each period's
    cross-unit mean of the result (two-way transformation).
    """
    out = df[value] - df.groupby(unit)[value].transform("mean")
    if time is not None:
        out = out - out.groupby(df[time]).transform("mean")
    return out


def zscore(s: pd.Series) -> pd.Series:
    """Standardize to mean 0 and standard deviation 1 (EMT ``standard``)."""
    return (s - s.mean()) / s.std()


def minmax(s: pd.Series) -> pd.Series:
    """Rescale to the interval [0, 1]."""
    return (s - s.min()) / (s.max() - s.min())
