"""Seasonal adjustment helpers (chapter 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil.easter import easter
from statsmodels.tsa.seasonal import STL

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def calendar_regressors(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Monthly trading-day contrasts, month length and an Easter regressor.

    Columns: the count of each weekday minus the count of Sundays in the
    month, the demeaned month length, and the share of the eight days before
    Easter that fall in the month (March or April).

    Args:
        index: month-start DatetimeIndex.

    """
    days = pd.date_range(index[0], index[-1] + pd.offsets.MonthEnd(0), freq="D")
    counts = pd.crosstab(days.to_period("M"), days.day_name())
    X = counts[WEEKDAYS].sub(counts["Sunday"], axis=0)
    length = counts.sum(axis=1)
    X["length"] = length - length.mean()
    pre_easter = {
        yr: pd.date_range(easter(yr) - pd.Timedelta(days=8), periods=8)
        for yr in index.year.unique()
    }
    X["easter"] = [
        (pre_easter[p.year].month == p.month).mean() if p.month in (3, 4) else 0.0
        for p in X.index
    ]
    X.index = index
    return X.astype(float)


def stl_adjust(
    s: pd.Series,
    period: int = 12,
    seasonal: int = 13,
    log: bool = True,
    calendar: bool = False,
    iterations: int = 2,
) -> pd.DataFrame:
    """Seasonally adjust with robust STL, optionally with calendar effects.

    Args:
        s: series with a regular month-start index (quarterly works with
            ``period=4`` and ``calendar=False``).
        period: seasonal period.
        seasonal: length of the STL seasonal smoother (odd, >= 7).
        log: adjust log(s) (multiplicative seasonality); results are
            returned in levels.
        calendar: also estimate trading-day and Easter effects by regressing
            the STL irregular on :func:`calendar_regressors`, alternating
            with STL ``iterations`` times.
        iterations: STL / calendar-regression rounds when ``calendar``.

    Returns:
        DataFrame with ``sa`` (adjusted series, in levels), ``seasonal`` and
        ``calendar`` factors (in logs when ``log``) and the STL ``trend``.

    """
    y = np.log(s) if log else s.astype(float)
    cal = pd.Series(0.0, index=y.index)
    X = calendar_regressors(y.index) if calendar else None
    for _ in range(iterations if calendar else 1):
        fit = STL(y - cal, period=period, seasonal=seasonal, robust=True).fit()
        if X is not None:
            irregular = y - fit.seasonal - fit.trend
            reg = sm.OLS(irregular, sm.add_constant(X)).fit()
            cal = X @ reg.params[X.columns]
    adjusted = y - fit.seasonal - cal
    return pd.DataFrame(
        {
            "sa": np.exp(adjusted) if log else adjusted,
            "seasonal": fit.seasonal,
            "calendar": cal,
            "trend": fit.trend,
        }
    )
