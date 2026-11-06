"""Temporal disaggregation: Denton, Chow-Lin and Fernandez (chapter 4).

All functions take a low-frequency series ``Y`` and return a high-frequency
series whose ``m``-period sums or means reproduce ``Y`` exactly. The
high-frequency input must start in the first sub-period of ``Y``'s first
period; it may extend beyond ``Y``'s last period, and those sub-periods are
extrapolated.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize_scalar

How = Literal["mean", "sum"]


def agg_matrix(n_low: int, n_high: int, m: int = 3, how: How = "mean") -> np.ndarray:
    """Build the matrix C such that C @ x aggregates blocks of ``m`` values.

    Columns beyond ``m * n_low`` (sub-periods after the last low-frequency
    observation) are zero.
    """
    if n_high < m * n_low:
        raise ValueError("the high-frequency series is shorter than m * len(Y)")
    C = np.kron(np.eye(n_low), np.ones((1, m)))
    C = np.hstack([C, np.zeros((n_low, n_high - m * n_low))])
    return C / m if how == "mean" else C


def denton(
    Y: pd.Series, indicator: pd.Series, m: int = 3, how: How = "mean"
) -> pd.Series:
    """Proportional first-difference Denton (Denton-Cholette) benchmarking.

    Minimizes the squared changes in the ratio of the result to ``indicator``
    subject to the aggregation constraint. A constant indicator gives a
    smooth interpolation that adds up.
    """
    i = indicator.to_numpy(float)
    T, n = len(i), len(Y)
    C = agg_matrix(n, T, m, how)
    D = np.diff(np.eye(T), axis=0)
    A = C * i
    K = np.block([[D.T @ D, A.T], [A, np.zeros((n, n))]])
    rhs = np.concatenate([np.zeros(T), Y.to_numpy(float)])
    r = np.linalg.solve(K, rhs)[:T]
    return pd.Series(r * i, index=indicator.index, name=Y.name)


def _regression_disagg(
    Y: pd.Series, X: pd.DataFrame | pd.Series, method: str, m: int, how: How
) -> tuple[pd.Series, dict[str, object]]:
    X = X.to_frame() if isinstance(X, pd.Series) else X
    Xh = sm.add_constant(X, has_constant="add").to_numpy(float)
    T, n = len(Xh), len(Y)
    C = agg_matrix(n, T, m, how)
    y, Xl = Y.to_numpy(float), C @ Xh
    t = np.arange(T)

    def gls(rho: float):
        if method == "fernandez":
            S = np.minimum.outer(t, t) + 1.0
        else:
            S = rho ** np.abs(np.subtract.outer(t, t)) / (1 - rho**2)
        V = C @ S @ C.T
        Vi = np.linalg.inv(V)
        beta = np.linalg.solve(Xl.T @ Vi @ Xl, Xl.T @ Vi @ y)
        e = y - Xl @ beta
        loglik = -0.5 * (n * np.log(e @ Vi @ e / n) + np.linalg.slogdet(V)[1])
        return beta, e, S, Vi, loglik

    rho = 1.0
    if method == "chow-lin":
        rho = float(
            minimize_scalar(
                lambda r: -gls(r)[-1], bounds=(0.0, 0.99), method="bounded"
            ).x
        )
    beta, e, S, Vi, _ = gls(rho)
    x = Xh @ beta + S @ C.T @ Vi @ e
    out = pd.Series(x, index=X.index, name=Y.name)
    return out, {"beta": beta, "rho": rho}


def chow_lin(
    Y: pd.Series, X: pd.DataFrame | pd.Series, m: int = 3, how: How = "mean"
) -> tuple[pd.Series, dict[str, object]]:
    """Chow-Lin (1971): GLS regression on indicators with AR(1) errors.

    The AR coefficient is estimated by maximum likelihood on [0, 0.99]; an
    estimate at the upper bound suggests :func:`fernandez` instead.

    Returns:
        The high-frequency series and a dict with ``beta`` (constant first)
        and ``rho``.

    """
    return _regression_disagg(Y, X, "chow-lin", m, how)


def fernandez(
    Y: pd.Series, X: pd.DataFrame | pd.Series, m: int = 3, how: How = "mean"
) -> tuple[pd.Series, dict[str, object]]:
    """Fernandez (1981): regression on indicators with random-walk errors."""
    return _regression_disagg(Y, X, "fernandez", m, how)
