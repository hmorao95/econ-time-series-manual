"""Business-cycle dating and cycle statistics (chapter 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bbq(y: pd.Series, window: int = 2, phase: int = 2, cycle: int = 5) -> pd.Series:
    """Harding-Pagan (2002) BBQ turning points: +1 peak, -1 trough, 0 otherwise.

    A port of the dating step of EMT ``date_``. Defaults are for quarterly
    data; for monthly data use ``window=5, phase=6, cycle=15``. Apply it to
    the level (e.g. log GDP), not to a growth rate.
    """
    v = y.to_numpy(float)
    tps: list[tuple[int, int]] = []
    for t in range(window, len(v) - window):
        seg = v[t - window : t + window + 1]
        if v[t] == seg.max():
            tps.append((t, 1))
        elif v[t] == seg.min():
            tps.append((t, -1))

    def alternate(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for t, k in points:
            if out and out[-1][1] == k:
                better = v[t] > v[out[-1][0]] if k == 1 else v[t] < v[out[-1][0]]
                if better:
                    out[-1] = (t, k)
            else:
                out.append((t, k))
        return out

    tps = alternate(tps)
    changed = True
    while changed:
        changed = False
        for i in range(len(tps) - 1):
            if tps[i + 1][0] - tps[i][0] < phase:
                del tps[i : i + 2]
                tps, changed = alternate(tps), True
                break
        if changed:
            continue
        for i in range(len(tps) - 2):
            if tps[i + 2][0] - tps[i][0] < cycle:
                del tps[i + 1 : i + 3]
                tps, changed = alternate(tps), True
                break

    def violates(t: int, k: int, rest: np.ndarray) -> bool:
        return bool(rest.size) and bool(
            rest.max() > v[t] if k == 1 else rest.min() < v[t]
        )

    while tps and violates(*tps[0], v[: tps[0][0]]):
        tps.pop(0)
    while tps and violates(*tps[-1], v[tps[-1][0] + 1 :]):
        tps.pop()

    out = pd.Series(0, index=y.index, name="turning_point")
    for t, k in tps:
        out.iloc[t] = k
    return out


def phases(y: pd.Series, turning_points: pd.Series) -> pd.DataFrame:
    """One row per complete phase: type, start, end, duration and amplitude.

    Amplitude is ``y[end] - y[start]``; with ``y = 100 * log(level)`` it is
    in percent.
    """
    pts = list(turning_points[turning_points != 0].items())
    rows = []
    for (t0, k0), (t1, _) in zip(pts, pts[1:], strict=False):
        i0, i1 = y.index.get_loc(t0), y.index.get_loc(t1)
        rows.append(
            {
                "phase": "contraction" if k0 == 1 else "expansion",
                "start": t0,
                "end": t1,
                "duration": i1 - i0,
                "amplitude": y[t1] - y[t0],
            }
        )
    return pd.DataFrame(rows)


def states(turning_points: pd.Series) -> pd.Series:
    """Expansion (1) / contraction (0) indicator from turning points (EMT ``states``).

    A contraction runs from the period after a peak to the trough, inclusive,
    the NBER convention.
    """
    tp = turning_points.replace(0, np.nan).ffill().shift(1)
    return tp.map({1: 0, -1: 1}).fillna(1).astype(int)


def concordance(a: pd.Series, b: pd.Series) -> float:
    """Share of periods in which two 0/1 indicators agree (Harding-Pagan)."""
    both = pd.concat([a, b], axis=1).dropna()
    return float((both.iloc[:, 0] == both.iloc[:, 1]).mean())


def cycle_moments(
    cycles: pd.DataFrame,
    reference: str,
    leads_lags: tuple[int, ...] = (-4, -2, 0, 2, 4),
) -> pd.DataFrame:
    """Stylized-facts table: std, std relative to ``reference``, AR(1), cross-correlations.

    Column ``corr(t+k)`` is corr(x[t+k], reference[t]): a peak at negative k
    means x leads the reference (EMT ``autocor``).
    """
    ref = cycles[reference]
    out = pd.DataFrame(
        {
            "std": cycles.std(),
            "relative std": cycles.std() / ref.std(),
            "AR(1)": cycles.apply(lambda c: c.autocorr()),
        }
    )
    for k in leads_lags:
        out[f"corr(t{k:+d})"] = cycles.apply(lambda c, k=k: c.shift(-k).corr(ref))
    return out


def robust_skewness(s: pd.Series) -> pd.Series:
    """Classical and three quantile-based skewness estimators (EMT ``skewness_``).

    All are 0 under normality.
    """
    s = s.dropna()
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    return pd.Series(
        {
            "classical": s.skew(),
            "Bowley (quartiles)": (q3 + q1 - 2 * q2) / (q3 - q1),
            "Groeneveld-Meeden": (s.mean() - q2) / (s - q2).abs().mean(),
            "Kendall-Stuart": (s.mean() - q2) / s.std(),
        }
    )


KURTOSIS_NORMAL = pd.Series(
    {
        "classical": 3.0,
        "Moors (octiles)": 1.23,
        "Hogg": 2.59,
        "Crow-Siddiqui": 2.91,
    }
)


def robust_kurtosis(s: pd.Series) -> pd.Series:
    """Classical and three quantile-based kurtosis estimators (EMT ``kurtosis_``).

    Values under normality are in :data:`KURTOSIS_NORMAL`.
    """
    s = s.dropna()
    q = s.quantile([0.025, 0.125, 0.25, 0.375, 0.625, 0.75, 0.875, 0.975])
    v, n = np.sort(s.to_numpy()), len(s)
    tail, half = max(1, round(0.05 * n)), n // 2
    return pd.Series(
        {
            "classical": s.kurt() + 3,
            "Moors (octiles)": (q[0.875] - q[0.625] + q[0.375] - q[0.125])
            / (q[0.75] - q[0.25]),
            "Hogg": (v[-tail:].mean() - v[:tail].mean())
            / (v[-half:].mean() - v[:half].mean()),
            "Crow-Siddiqui": (q[0.975] - q[0.025]) / (q[0.75] - q[0.25]),
        }
    )


def pc_factors(
    panel: pd.DataFrame, n: int = 1, standardize: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Principal-component factors normalized so that F'F/T = I (EMT ``pc_T``).

    Args:
        panel: T x N, no missing values.
        n: number of factors.
        standardize: standardize each column first (recommended unless all
            series share units and volatility).

    Returns:
        ``factors`` (T x n), ``loadings`` (N x n) and the share of variance
        of every component. Each factor is signed so that its loadings sum
        to a positive number.

    """
    Z = (panel - panel.mean()) / panel.std() if standardize else panel - panel.mean()
    U, S, Vt = np.linalg.svd(Z.to_numpy(float), full_matrices=False)
    T = len(Z)
    names = [f"F{i + 1}" for i in range(n)]
    F = U[:, :n] * np.sqrt(T)
    L = (Vt[:n].T * S[:n]) / np.sqrt(T)
    signs = np.sign(L.sum(axis=0))
    signs[signs == 0] = 1
    factors = pd.DataFrame(F * signs, index=panel.index, columns=names)
    loadings = pd.DataFrame(L * signs, index=panel.columns, columns=names)
    return factors, loadings, S**2 / (S**2).sum()
