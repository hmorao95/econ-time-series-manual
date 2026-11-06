# 8. Business cycles: turning points and stylized facts

Two classic descriptive exercises: **dating** the cycle (when did recessions
start and end?) and **summarizing** it (how volatile, persistent and
procyclical is each variable?).

!!! tip "Quick rules"
    * Date turning points on the **level** of (log) activity: the classical
      cycle. Dating a detrended series gives the growth cycle, a different
      object with more turning points.
    * Moments of the cycle depend on the filter (chapter 6). State which one.
    * Exclude 2020 when estimating moments, or show results with and without.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tsmanual import load

y = 100 * np.log(load("GDPC1")).to_period("Q")
rec = load("USRECQ").to_period("Q").reindex(y.index)
```

## Dating turning points: the BBQ algorithm

Harding and Pagan (2002) formalize the NBER's rule of thumb. A quarter is a
peak (trough) if it is the highest (lowest) point within two quarters on
each side; peaks and troughs must alternate; a phase lasts at least two
quarters and a full cycle at least five. This is a compact port of the BBQ
routine in EMT's `date_` (by James Engel, Adrian Pagan, Don Harding and
Fabio Canova).

```python
def bbq(y: pd.Series, window: int = 2, phase: int = 2, cycle: int = 5) -> pd.Series:
    """Harding-Pagan turning points: +1 peak, -1 trough, 0 otherwise.

    Defaults are for quarterly data; for monthly use window=5, phase=6,
    cycle=15.
    """
    v = y.to_numpy(float)
    # 1. local extrema
    tps = []
    for t in range(window, len(v) - window):
        seg = v[t - window : t + window + 1]
        if v[t] == seg.max():
            tps.append((t, 1))
        elif v[t] == seg.min():
            tps.append((t, -1))

    def alternate(tps):
        """Among consecutive peaks keep the highest; troughs, the lowest."""
        out = []
        for t, k in tps:
            if out and out[-1][1] == k:
                if (v[t] > v[out[-1][0]]) if k == 1 else (v[t] < v[out[-1][0]]):
                    out[-1] = (t, k)
            else:
                out.append((t, k))
        return out

    # 2. censoring: alternation, minimum phase, minimum cycle
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

    # 3. end rules: the first/last peak (trough) must not be exceeded
    #    (undercut) between it and the edge of the sample
    def violates(t, k, rest):
        return rest.size and (rest.max() > v[t] if k == 1 else rest.min() < v[t])

    while tps and violates(*tps[0], v[: tps[0][0]]):
        tps.pop(0)
    while tps and violates(*tps[-1], v[tps[-1][0] + 1 :]):
        tps.pop()

    out = pd.Series(0, index=y.index, name="turning_point")
    for t, k in tps:
        out.iloc[t] = k
    return out


tp = bbq(y)
nber_peaks = rec.index[(rec == 0) & (rec.shift(-1) == 1)]
nber_troughs = rec.index[(rec == 1) & (rec.shift(-1) == 0)]
print("BBQ peaks:   ", " ".join(tp.index[tp == 1].astype(str)))
print("NBER peaks:  ", " ".join(nber_peaks.astype(str)))
print("BBQ troughs: ", " ".join(tp.index[tp == -1].astype(str)))
print("NBER troughs:", " ".join(nber_troughs.astype(str)))
```

On GDP alone, BBQ recovers almost every NBER recession within a quarter.
It misses 2001 (GDP never fell for two quarters in the current vintage) and
dates the Great Recession peak in 2008Q2 rather than 2007Q4, because GDP
kept growing into early 2008. The NBER looks at many monthly indicators,
employment above all.

## Phases: duration and amplitude

From the turning points, build the expansion/contraction indicator and
summarize each phase (EMT `states` and the statistics of `date_`).

```python
def phases(y: pd.Series, tp: pd.Series) -> pd.DataFrame:
    """One row per complete phase: type, start, end, duration, amplitude."""
    pts = tp[tp != 0]
    rows = []
    for (t0, k0), (t1, _) in zip(pts.items(), list(pts.items())[1:]):
        rows.append(
            {
                "phase": "contraction" if k0 == 1 else "expansion",
                "start": t0,
                "end": t1,
                "duration": (t1 - t0).n,  # quarters
                "amplitude": y[t1] - y[t0],  # log points x 100 = %
            }
        )
    return pd.DataFrame(rows)


ph = phases(y, tp)
print(ph.tail(4).round({"amplitude": 1}).to_string(index=False))
print(ph.groupby("phase")[["duration", "amplitude"]].mean().round(1))

# expansion (1) / contraction (0) indicator, after EMT `states`
state = tp.replace(0, np.nan).ffill().map({1: 0, -1: 1}).fillna(1).astype(int)
```

```python
fig, ax = plt.subplots()
ts = y.to_timestamp()
ax.plot(ts.index, ts, color="k", lw=1)
contr = state.to_timestamp() == 0
ax.fill_between(
    ts.index,
    ts.min(),
    ts.max(),
    where=contr,
    color="0.8",
    step="post",
    label="BBQ contraction",
)
ax.plot(
    ts.index[tp.to_numpy() == 1], ts[tp.to_numpy() == 1], "v", color="C3", label="peak"
)
ax.plot(
    ts.index[tp.to_numpy() == -1],
    ts[tp.to_numpy() == -1],
    "^",
    color="C2",
    label="trough",
)
ax.set_title("Log real GDP x 100 and BBQ turning points")
ax.legend(loc="upper left")
# figure: bbq
```

![BBQ turning points](img/bbq.png)

## Stylized facts of the cycle

The Kydland–Prescott table: for each variable's cyclical component, its
volatility (absolute and relative to GDP), persistence, and correlation with
the GDP cycle at leads and lags (EMT `autocor`).

```python
q = lambda sid: load(sid).to_period("M").resample("Q").mean()  # monthly -> quarterly
levels = pd.DataFrame(
    {
        "GDP": 100 * np.log(load("GDPC1").to_period("Q")),
        "industrial production": 100 * np.log(q("INDPRO")),
        "payroll employment": 100 * np.log(q("PAYEMS")),
        "unemployment rate": q("UNRATE"),  # already in percent: no log
        "GDP deflator": 100 * np.log(load("GDPDEF").to_period("Q")),
    }
).loc["1960":"2019"]

cyc = levels.apply(lambda s: sm.tsa.filters.hpfilter(s, lamb=1600)[0])

facts = pd.DataFrame(
    {
        "std (%)": cyc.std(),
        "relative std": cyc.std() / cyc["GDP"].std(),
        "AR(1)": cyc.apply(lambda c: c.autocorr()),
    }
)
for k in [-4, -2, 0, 2, 4]:
    # corr(x[t+k], GDP[t]): high at k < 0 means x leads GDP
    facts[f"corr x(t{k:+d}), GDP(t)"] = cyc.apply(
        lambda c: c.shift(-k).corr(cyc["GDP"])
    )
print(facts.round(2).to_string())
```

Read it like this (1960–2019, HP cycles): industrial production is about
twice as volatile as GDP and strongly procyclical. Payroll employment is
procyclical and **lags** GDP: its correlation peaks at $x_{t+2}$. The
unemployment rate is strongly countercyclical, mostly contemporaneously.
The price level is countercyclical, most strongly a year *before* the GDP
cycle (Kydland and Prescott, 1990, read this as evidence of supply shocks).

## Pitfalls

* BBQ on a noisy series finds spurious cycles; smooth or use a longer window
  for monthly data.
* Phase statistics from a handful of recessions have huge sampling error.
* Cross-correlations of two-sided filtered series mix leads and lags (the
  filter itself is two-sided). Leads/lags of a few quarters are not robust
  evidence of causality.
