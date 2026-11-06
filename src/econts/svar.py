"""Recursive SVAR helpers: scaled impulse responses and historical decompositions (chapter 11)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def irf_with_bands(
    var_results: object,
    shock: int = 0,
    horizon: int = 24,
    cumulate: Sequence[bool] | None = None,
    impact: float | None = None,
    signif: float = 0.32,
    repl: int = 500,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Orthogonalized (Cholesky) responses to one shock, with Monte Carlo bands.

    Args:
        var_results: a fitted ``statsmodels`` ``VARResults``.
        shock: position of the shock in the Cholesky ordering.
        horizon: last horizon (0 is impact).
        cumulate: per variable, whether to cumulate the response (variables
            in differences); default none.
        impact: scale the shock so that the response of the shocked variable
            on impact (after cumulation) equals this number, e.g. 10 for a
            10% oil price shock; default one standard deviation.
        signif: 0.32 for 68% bands, 0.10 for 90%.
        repl: Monte Carlo replications.
        seed: random seed for the bands.

    Returns:
        Dict with ``"median"`` (point estimate), ``"lower"`` and ``"upper"``,
        each a (horizon + 1) x k DataFrame indexed by horizon.

    """
    names = list(var_results.names)
    k = len(names)
    cum = np.zeros(k, bool) if cumulate is None else np.asarray(cumulate, bool)
    irf = var_results.irf(horizon)
    point = np.where(cum, irf.orth_cum_effects[:, :, shock], irf.orth_irfs[:, :, shock])
    rng = np.random.default_rng(seed)
    lo_c, hi_c = irf.cum_errband_mc(orth=True, repl=repl, signif=signif, rng=rng)
    lo_l, hi_l = irf.errband_mc(orth=True, repl=repl, signif=signif, rng=rng)
    lo = np.where(cum, lo_c[:, :, shock], lo_l[:, :, shock])
    hi = np.where(cum, hi_c[:, :, shock], hi_l[:, :, shock])
    scale = 1.0 if impact is None else impact / point[0, shock]
    h = pd.RangeIndex(horizon + 1, name="horizon")
    return {
        name: pd.DataFrame(arr * scale, index=h, columns=names)
        for name, arr in (("median", point), ("lower", lo), ("upper", hi))
    }


def historical_decomposition(var_results: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Contribution of each Cholesky shock to each variable, period by period.

    Returns:
        ``contributions``: DataFrame with a (variable, shock) column
        MultiIndex over the residual sample; and ``deterministic``: the data
        minus the sum of the contributions (constant and initial conditions).
        For each variable, deterministic + contributions sum to the data.

    """
    resid = var_results.resid
    names = list(var_results.names)
    eps = resid.to_numpy() @ np.linalg.inv(np.linalg.cholesky(var_results.sigma_u)).T
    T, n = eps.shape
    theta = var_results.orth_ma_rep(maxn=T - 1)
    contrib = np.zeros((T, n, n))
    for t in range(T):
        contrib[t] = np.einsum("jik,jk->ik", theta[: t + 1], eps[t::-1])
    columns = pd.MultiIndex.from_product([names, names], names=["variable", "shock"])
    contributions = pd.DataFrame(
        contrib.reshape(T, n * n), index=resid.index, columns=columns
    )
    data = pd.DataFrame(var_results.endog[-T:], index=resid.index, columns=names)
    deterministic = data - contributions.T.groupby(level="variable").sum().T[names]
    return contributions, deterministic
