"""Trend-cycle decompositions (chapter 6).

Every function returns ``(cycle, trend)`` as Series on the input's index
(NaN where a filter has no value), except :func:`haar_components`.
Several are ports of Filippo Ferroni's Empirical Macro Toolbox (EMT).
Apply them to ``100 * log(level)`` so that the cycle is in percent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

Pair = tuple[pd.Series, pd.Series]


def poly_detrend(y: pd.Series, order: int = 1, break_at: object = None) -> Pair:
    """Polynomial trend; with ``break_at``, a separate polynomial after it.

    EMT ``polydet`` and ``polydet_break``.
    """
    t = np.arange(len(y), dtype=float)
    X = np.vander(t, order + 1)
    if break_at is not None:
        after = (y.index >= pd.Timestamp(break_at)).astype(float)[:, None]
        X = np.hstack([X * (1 - after), X * after])
    beta, *_ = np.linalg.lstsq(X, y.to_numpy(float), rcond=None)
    trend = pd.Series(X @ beta, index=y.index)
    return y - trend, trend


def hp(y: pd.Series, lamb: float = 1600) -> Pair:
    """Two-sided Hodrick-Prescott filter (1600 quarterly, 129600 monthly, 6.25 annual)."""
    cycle, trend = sm.tsa.filters.hpfilter(y, lamb=lamb)
    return cycle, trend


def hp_one_sided(y: pd.Series, lamb: float = 1600, min_obs: int = 12) -> Pair:
    """One-sided HP: the trend at t is the end point of HP on data up to t.

    EMT ``one_sided_hpfilter_serial``. Equal to the Kalman-filter estimate of
    the HP state-space model; the Basel credit gap uses ``lamb=400_000``.
    """
    trend = pd.Series(np.nan, index=y.index)
    for t in range(min_obs, len(y) + 1):
        trend.iloc[t - 1] = sm.tsa.filters.hpfilter(y.iloc[:t], lamb=lamb)[1].iloc[-1]
    return y - trend, trend


def hamilton_filter(y: pd.Series, h: int = 8, p: int = 4) -> Pair:
    """Hamilton (2018): cycle = residual of y_t on y_{t-h}, ..., y_{t-h-p+1}.

    EMT ``hamfilter``. Use h=8, p=4 for quarterly data, h=24, p=12 monthly.
    """
    X = pd.concat({f"lag{h + j}": y.shift(h + j) for j in range(p)}, axis=1)
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    fit = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit()
    return fit.resid.reindex(y.index), fit.fittedvalues.reindex(y.index)


def bk(y: pd.Series, low: int = 6, high: int = 32, K: int = 12) -> pd.Series:
    """Baxter-King band-pass cycle; loses ``K`` observations at each end."""
    return sm.tsa.filters.bkfilter(y, low=low, high=high, K=K).reindex(y.index)


def cf(y: pd.Series, low: int = 6, high: int = 32, drift: bool = True) -> Pair:
    """Christiano-Fitzgerald asymmetric band-pass filter, full sample.

    The trend is ``y - cycle`` (statsmodels' own trend output excludes the
    drift when ``drift=True``).
    """
    cycle, _ = sm.tsa.filters.cffilter(y, low=low, high=high, drift=drift)
    return cycle, y - cycle


def _companion(coefs: np.ndarray) -> np.ndarray:
    p, k, _ = coefs.shape
    F = np.zeros((k * p, k * p))
    F[:k] = np.hstack(list(coefs))
    F[k:, :-k] = np.eye(k * (p - 1))
    return F


def beveridge_nelson(y: pd.Series, p: int = 4) -> Pair:
    """Univariate Beveridge-Nelson decomposition with an AR(p) for the difference.

    EMT ``BNuniv``. With output alone the BN cycle is small and can have the
    wrong sign; see :func:`bn_bq_bivariate`.
    """
    dy = y.diff().dropna()
    x = dy - dy.mean()
    phi = np.asarray(sm.tsa.AutoReg(x.to_numpy(), lags=p, trend="n").fit().params)
    F = _companion(phi.reshape(p, 1, 1))
    X = pd.concat([x.shift(j) for j in range(p)], axis=1)
    gain = (F @ np.linalg.inv(np.eye(p) - F))[0]
    cycle = -(X @ gain).where(X.notna().all(axis=1)).reindex(y.index)
    return cycle, y - cycle


def bn_bq_bivariate(y: pd.Series, z: pd.Series, p: int = 8) -> pd.DataFrame:
    """Bivariate Beveridge-Nelson and Blanchard-Quah cycles of ``y`` (EMT ``BNBQbiv``).

    Fits a VAR(p) to the first difference of ``y`` (e.g. 100 x log output)
    and a stationary cyclical variable ``z`` (e.g. the unemployment rate).

    Returns:
        DataFrame with ``bn`` (cycle implied by the VAR's long-run forecast)
        and ``bq`` (cumulated contribution of the shock with no long-run
        effect on ``y``, signed to raise ``y`` on impact).

    """
    X = pd.DataFrame({"dy": y.diff(), "z": z}).dropna()
    var = sm.tsa.VAR(X.to_numpy()).fit(p)  # arrays: no date-frequency warnings
    k = 2
    F = _companion(var.coefs)
    Z = pd.concat([(X - X.mean()).shift(j) for j in range(p)], axis=1)
    gain = (F @ np.linalg.inv(np.eye(k * p) - F))[0]
    bn = -(Z @ gain).where(Z.notna().all(axis=1))

    A1 = np.eye(k) - var.coefs.sum(axis=0)
    A1_inv = np.linalg.inv(A1)
    B0 = A1 @ np.linalg.cholesky(A1_inv @ var.sigma_u @ A1_inv.T)
    B0[:, 1] *= np.sign(B0[0, 1])
    e = np.asarray(var.resid) @ np.linalg.inv(B0).T
    theta = var.ma_rep(len(e) - 1) @ B0
    demand = [theta[: t + 1, 0, 1] @ e[t::-1, 1] for t in range(len(e))]
    bq = pd.Series(np.cumsum(demand), index=X.index[p:])
    return pd.DataFrame({"bn": bn, "bq": bq}).reindex(y.index)


def haar_components(y: pd.Series, levels: int = 5) -> pd.DataFrame:
    """One-sided Haar wavelet details D1..D_levels (EMT ``wavefilter``).

    ``D_j`` captures periods of 2^j to 2^(j+1) observations. Remove the drift
    first (e.g. apply to a linearly detrended series): on a trending series
    each detail picks up the growth rate.
    """
    out = {}
    for j in range(1, levels + 1):
        w = 2 ** (j - 1)
        m = y.rolling(w).mean()
        out[f"D{j}"] = (m - m.shift(w)) / 2
    return pd.DataFrame(out)


def haar_cycle(y: pd.Series) -> Pair:
    """Business-cycle component D3 + D4 (8-32 periods) and trend y - D1..D4."""
    D = haar_components(y, levels=4)
    return D["D3"] + D["D4"], y - D.sum(axis=1)
