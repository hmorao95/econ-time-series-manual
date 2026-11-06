"""Temporal aggregation from high to low frequency (chapter 3)."""

from __future__ import annotations

import pandas as pd


def aggregate(
    s: pd.Series, freq: str, how: str = "mean", min_obs: int = 1
) -> pd.Series:
    """Resample ``s`` with ``how``, setting periods with fewer than ``min_obs`` values to NaN.

    Args:
        s: series with a DatetimeIndex (or PeriodIndex).
        freq: target frequency alias, e.g. ``"MS"``, ``"QS"``, ``"YE-SEP"``.
        how: ``"sum"`` for flows, ``"mean"`` for rates, prices, indices and
            flows at annual rates, ``"last"`` for stocks; any resample
            aggregation name works.
        min_obs: minimum number of non-missing observations per period.

    """
    r = s.resample(freq)
    return r.agg(how).where(r.count() >= min_obs)


def weekly_to_monthly(w: pd.Series, how: str = "sum") -> pd.Series:
    """Convert a weekly series stamped on the week's last day to monthly.

    Each week is spread evenly over its seven days, so a week straddling two
    months is split between them. ``how="sum"`` suits flows (claims, sales);
    ``how="mean"`` suits weekly averages (rates, prices). The incomplete
    first and last months are dropped.
    """
    days = pd.date_range(w.index[0] - pd.Timedelta(days=6), w.index[-1], freq="D")
    daily = w.reindex(days, method="bfill")
    if how == "sum":
        daily = daily / 7
    monthly = daily.resample("MS").agg(how)
    return monthly.iloc[1:-1]
