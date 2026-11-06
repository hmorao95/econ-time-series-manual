"""Stationarity tests and outlier detection (chapter 9)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


def unit_root_table(
    series: dict[str, tuple[pd.Series, str]], alpha: float = 0.05
) -> pd.DataFrame:
    """ADF and KPSS p-values side by side, with a joint verdict.

    Args:
        series: name -> (series, regression), regression ``"c"`` (constant)
            or ``"ct"`` (constant and trend).
        alpha: significance level for the verdict.

    Returns:
        One row per series. KPSS p-values are bounded to [0.01, 0.10].

    """
    rows = {}
    for name, (s, reg) in series.items():
        s = s.dropna()
        adf_p = adfuller(s, regression=reg, autolag="AIC", result_object=True).pvalue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_p = kpss(s, regression=reg, nlags="auto", result_object=True).pvalue
        if adf_p < alpha and kpss_p > alpha:
            verdict = "stationary"
        elif adf_p > alpha and kpss_p < alpha:
            verdict = "unit root"
        else:
            verdict = "unclear"
        rows[name] = {"ADF p": adf_p, "KPSS p": kpss_p, "verdict": verdict}
    return pd.DataFrame(rows).T.infer_objects()


def robust_zscore(s: pd.Series, window: int = 21) -> pd.Series:
    """Distance from the centered rolling median in robust standard deviations.

    Uses 1.4826 x the rolling median absolute deviation, which equals the
    standard deviation under normality.
    """
    mp = window // 2
    med = s.rolling(window, center=True, min_periods=mp).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=mp).median()
    return (s - med) / (1.4826 * mad.replace(0, np.nan))


def robust_outliers(
    s: pd.Series, window: int = 21, threshold: float = 4.0
) -> pd.Series:
    """Boolean mask of points more than ``threshold`` robust SDs from the rolling median."""
    return robust_zscore(s, window).abs() > threshold


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip a series at its ``lower`` and ``upper`` quantiles."""
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)
