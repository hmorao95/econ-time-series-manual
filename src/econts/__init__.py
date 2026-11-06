"""econts: a toolkit for manipulating economic time series.

The functions of the *Economic Time Series in Python* book, packaged for use
in projects. Every function takes and returns pandas objects.

Modules:
    transform    growth rates, carry-over, rebasing, deflating, splicing,
                 demeaning (by group, panel within), z-scores, min-max
    aggregate    high- to low-frequency aggregation, weekly to monthly
    disagg       temporal disaggregation: Denton, Chow-Lin, Fernandez
    seasonal     STL adjustment with trading-day and Easter effects
    filters      trend-cycle: polynomial (with break), HP, one-sided HP,
                 Hamilton, Baxter-King, Christiano-Fitzgerald,
                 Beveridge-Nelson (uni- and bivariate), Blanchard-Quah, Haar
    cycles       BBQ dating, phases, states, concordance, cycle moments,
                 robust skewness and kurtosis, principal-component factors
    dummies      episodes, dummies from dates, pulse/step/ramp dummies,
                 recession shading for matplotlib
    diagnostics  ADF/KPSS table, robust outliers, winsorizing
    svar         scaled IRFs with bands, historical decompositions
    data         ``load``: a bundled FRED snapshot or any FRED series online

Example:
    >>> from econts import load
    >>> from econts.filters import hamilton_filter
    >>> import numpy as np
    >>> cycle, trend = hamilton_filter(100 * np.log(load("GDPC1")))

"""

from econts import (
    aggregate,
    cycles,
    diagnostics,
    disagg,
    dummies,
    filters,
    seasonal,
    svar,
    transform,
)
from econts.data import DATA_DIR, SERIES, load

__version__ = "0.2.0"

__all__ = [
    "DATA_DIR",
    "SERIES",
    "aggregate",
    "cycles",
    "diagnostics",
    "disagg",
    "dummies",
    "filters",
    "load",
    "seasonal",
    "svar",
    "transform",
]
