# 2. Transformations: logs, growth rates, indices, deflators

!!! tip "Quick rules"
    * Work in **logs × 100** for anything that grows: differences are then
      (approximately) percent changes and trends become linear.
    * Say which growth rate you report: **q/q**, **q/q annualized (SAAR)**,
      **y/y**, or **Q4/Q4**. They answer different questions.
    * US national accounts levels are **already annualized** (SAAR). Divide by
      4 for the quarterly level; average, don't sum, to get annual values.

```python
import numpy as np
import pandas as pd
from tsmanual import load

gdp = load("GDPC1").to_period("Q")
cpi = load("CPIAUCSL").to_period("M")
```

## Growth rates

| Rate | Formula | pandas |
|---|---|---|
| period on period | $y_t / y_{t-1} - 1$ | `s.pct_change()` |
| year on year (quarterly data) | $y_t / y_{t-4} - 1$ | `s.pct_change(4)` |
| annualized q/q | $(y_t / y_{t-1})^4 - 1$ | `(1 + s.pct_change()) ** 4 - 1` |
| log difference | $100\,(\ln y_t - \ln y_{t-1})$ | `100 * np.log(s).diff()` |
| annualized log difference | $400\,\Delta \ln y_t$ | `400 * np.log(s).diff()` |

```python
g = pd.DataFrame(
    {
        "q/q": 100 * gdp.pct_change(),
        "q/q saar": 100 * ((1 + gdp.pct_change()) ** 4 - 1),
        "y/y": 100 * gdp.pct_change(4),
        "400 dlog": 400 * np.log(gdp).diff(),
    }
)
print(g.loc["2024Q1":].round(2))
```

The log approximation is excellent for small changes and poor for large ones:
the 2020Q2 GDP collapse is −28.0% SAAR but −32.8 in annualized log points.
Report exact rates; use log differences inside models.

For monthly data replace 4 by 12. Inflation, the classic case:

```python
infl = pd.DataFrame(
    {
        "m/m": 100 * cpi.pct_change(fill_method=None),
        "m/m annualized": 100 * ((1 + cpi.pct_change(fill_method=None)) ** 12 - 1),
        "y/y": 100 * cpi.pct_change(12, fill_method=None),
    }
)
print(infl.tail(4).round(2))
```

## Annual growth: average-over-average vs Q4/Q4

Annual GDP growth in the press is the growth of the **annual average**.
Policy forecasts (e.g. FOMC projections) use **Q4 over Q4**. They can differ
by more than a percentage point.

```python
annual = gdp.groupby(gdp.index.year).mean()  # SAAR levels: average them
avg_growth = 100 * annual.pct_change()
q4q4 = 100 * gdp[gdp.index.quarter == 4].pct_change()
q4q4.index = q4q4.index.year
print(pd.DataFrame({"avg/avg": avg_growth, "Q4/Q4": q4q4}).loc[2019:2025].round(1))
```

## Carry-over (statistical overhang)

How much annual growth is "already in the bag" if the series stays flat at its
last quarter for the rest of the year? It is the last quarter's level relative
to the previous year's average.

```python
def carry_over(q: pd.Series, year: int) -> float:
    """Annual growth in `year` if the series stays at its Q4 level of year-1."""
    prev = q[q.index.year == year - 1]
    return 100 * (prev.iloc[-1] / prev.mean() - 1)


print(f"carry-over into 2026: {carry_over(gdp, 2026):.2f}%")
```

## Rebasing an index

```python
ip = load("INDPRO")
ip_2015 = 100 * ip / ip["2015"].mean()  # 2015 average = 100
print(ip_2015["2015"].mean().round(6), ip_2015.iloc[-1].round(1))
```

Rebasing a single series is harmless. Rebasing **chain-linked volumes**
breaks additivity: real components do not sum to real GDP, before or after.

## Nominal to real: deflating

Real = nominal / price index × 100, with the price index based to 100 in the
reference year of the real series.

```python
ngdp = load("GDP").to_period("Q")
defl = load("GDPDEF").to_period("Q")
real = ngdp / defl * 100  # deflator is 2017 = 100
check = pd.DataFrame({"implied": real, "published": gdp}).dropna()
print((check["implied"] / check["published"] - 1).abs().max())  # < 1e-4: rounding only
```

Deflate with the price index that matches the aggregate (wages by CPI for
purchasing power, output by its own deflator). For mixed frequencies, first
bring the deflator to the frequency of the nominal series (chapter 3).

## Demeaning, standardizing and scaling

Demeaning subtracts a mean; *which* mean is the decision. The full-sample
mean (EMT `demean`) is the default; pandas skips `NaN` when computing it.

```python
g_ip = 100 * np.log(ip).diff()  # monthly IP growth, %
demeaned = g_ip - g_ip.mean()  # full-sample mean
print(round(g_ip.mean(), 3), round(demeaned.mean(), 12))
```

**By regime.** Trend growth was higher before 2000. Demeaning each regime by
its own mean removes the shift that a single mean would leave in:

```python
regime = np.where(g_ip.index < "2000-01-01", "pre-2000", "2000+")
print(g_ip.groupby(regime).mean().round(3))
by_regime = g_ip - g_ip.groupby(regime).transform("mean")
```

**By calendar month.** Subtracting each month's average growth removes a
stable seasonal pattern: a crude seasonal adjustment (chapter 5 does it
properly).

```python
g_nsa = 100 * np.log(load("RSXFSN")).diff()
by_month = g_nsa - g_nsa.groupby(g_nsa.index.month).transform("mean")
print(g_nsa.std().round(2), "->", by_month.std().round(2))  # most variance was seasonal
```

**By unit, in a panel (the within transformation).** With several
countries in long format, subtract each country's mean: this is what a
fixed-effects regression does. Subtracting each period's cross-country mean
as well gives the two-way transformation.

```python
unemp = (
    pd.concat(
        {
            "US": load("UNRATE").to_period("M").resample("Q").mean(),
            "PT": load("LRHUTTTTPTQ156S").to_period("Q"),
        },
        names=["country", "date"],
    )
    .rename("u")
    .loc[:, "1983":"2019"]
    .reset_index()
)

unemp["within"] = unemp["u"] - unemp.groupby("country")["u"].transform("mean")
unemp["two_way"] = unemp["within"] - unemp.groupby("date")["within"].transform("mean")
print(unemp.groupby("country")[["u", "within"]].mean().round(2))
```

**Standardizing** divides the demeaned series by its standard deviation
(EMT `standard`), so series in different units can be compared or combined
(as the factor model of chapter 8 does):

```python
def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


z = zscore(100 * np.log(gdp).diff().dropna())
print(z.describe().round(2))
```

Other common scalings:

```python
scalings = pd.DataFrame(
    {
        "% deviation from mean": 100 * (ip / ip.mean() - 1),
        "index, first obs = 100": 100 * ip / ip.dropna().iloc[0],
        "min-max [0, 1]": (ip - ip.min()) / (ip.max() - ip.min()),
    }
)
print(scalings.tail(2).round(2))
```

Demean and standardize on the **estimation sample** only; reusing
full-sample moments in a forecasting exercise leaks future information
(chapter 7's `expanding()` gives the real-time version). Demeaning a series
with a trend does not make it stationary: detrend it instead (chapter 6).

## Splicing series with a level break

Statistical offices rebase and redefine series. To build one long series,
**splice** the old one onto the new one using growth rates of the old series
(ratio splicing). Here we fake a rebased "old" industrial production index.

```python
old = (1.37 * ip[:"2012"]).rename("old")  # old base, ends 2012
new = ip["2005":].rename("new")  # new base, starts 2005


def ratio_splice(old: pd.Series, new: pd.Series, at=None) -> pd.Series:
    """Rescale `old` to the level of `new` at the link date and join them."""
    at = at or new.index[0]
    factor = new[at] / old[at]
    return pd.concat([old[old.index < at] * factor, new[new.index >= at]])


long = ratio_splice(old, new)
print((long / ip - 1).abs().max())  # 0: growth rates of old preserved
```

Link at a single date, or at an average over the overlap
(`new[overlap].mean() / old[overlap].mean()`) when either series is noisy.
For differences in **levels** (e.g. a percentage), splice additively instead.

## Pitfalls

* **Missing observations.** BLS never published the October 2025 CPI (a
  federal government shutdown), so `CPIAUCSL` has a `NaN` there. pandas 3
  returns `NaN` for any growth rate that touches it; pandas 2 by default
  *fills* the gap with the previous value and reports a spurious 0% m/m
  change. Pass `fill_method=None` so the code behaves the same in both.
* `pct_change` on a series with *dropped* rows compares across the gap. Call
  `.asfreq(...)` first so that the gap produces `NaN` (chapter 1).
* Logs of zero or negative numbers (inventories change, net exports) are
  undefined; use differences scaled by a level instead (contributions).
* Percent vs percentage point: unemployment going from 4% to 5% is a
  1 pp rise and a 25% increase.
