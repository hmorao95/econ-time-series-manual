# 3. Temporal aggregation: from high to low frequency

Going from daily to monthly, or monthly to quarterly, is a one-liner with
`resample`. The economics is in choosing the **aggregation rule**, which
depends on what the series measures.

!!! tip "Quick rules"
    | The series is a… | Example | Aggregate with |
    |---|---|---|
    | **flow** (amount per period) | retail sales, claims, exports | `sum` |
    | **flow at an annual rate** (SAAR) | US GDP, housing starts | `mean` |
    | **stock** (level at a date) | debt, money stock, inventories | `last` |
    | **average / rate / price / index** | unemployment rate, CPI, oil price | `mean` |

    Count observations per period and drop periods that are incomplete.

```python
import numpy as np
import pandas as pd
from tsmanual import load

retail = load("RSXFS")  # monthly flow, $ millions
payems = load("PAYEMS")  # monthly employment level
unrate = load("UNRATE")  # monthly rate
brent = load("DCOILBRENTEU")  # daily price
claims = load("ICSA")  # weekly flow, weeks ending Saturday
```

## The basic pattern

`resample` takes a frequency alias (chapter 1) and an aggregation. With
start-of-period aliases (`QS`, `YS`) the result keeps FRED's labelling; with
`PeriodIndex` input you get periods back.

```python
q = pd.DataFrame(
    {
        "retail (sum)": retail.resample("QS").sum(min_count=3),
        "payrolls (mean)": payems.resample("QS").mean(),
        "payrolls (end)": payems.resample("QS").last(),
        "unrate (mean)": unrate.resample("QS").mean(),
    }
)
print(q.loc["2025":].round(1))

# Same thing on periods
print(unrate.to_period("M").resample("Q").mean().tail(2))
```

`sum` returns 0 for an empty period and would happily sum two months of a
three-month quarter. `min_count=3` returns `NaN` instead, which is what you
want for the last, incomplete quarter.

## Incomplete periods

A general guard: aggregate the observation count alongside the value, and mask
periods with too few observations.

```python
def aggregate(s: pd.Series, freq: str, how: str = "mean", min_obs: int = 1):
    """Resample `s` with `how`, setting periods with < `min_obs` values to NaN."""
    r = s.resample(freq)
    out = r.agg(how)
    return out.where(r.count() >= min_obs)


brent_m = aggregate(brent, "MS", "mean", min_obs=15)
print(brent_m.tail(3).round(2))  # current month is NaN until 15 trading days exist
```

For daily prices, also choose the **concept**: the monthly average
(`mean`), the end-of-month price (`last`), or the full open/high/low/close:

```python
print(brent.resample("MS").ohlc().loc["2026"].round(2).tail(3))
```

## Stocks: end of period vs average

Payroll employment is a level; the quarterly figure can be the average of the
three months (BLS and BEA convention) or the March/June/September/December
value. The difference matters around turning points:

```python
gap = q["payrolls (end)"] - q["payrolls (mean)"]
print(gap["2020"].round(0))  # thousands: about 4 million jobs in 2020Q2
```

## SAAR flows

US national-accounts flows are annualized. The annual value is the **mean**
of the four quarters, not the sum (the sum would be four times too large).

```python
gdp = load("GDPC1")
print(gdp.resample("YS").mean().loc["2023":"2025"].round(0))
```

For non-annualized quarterly flows (most European national accounts), sum.

## Weekly to monthly

Weeks straddle month boundaries. Resampling by the week-ending date puts
all of the week Sunday 30 March to Saturday 5 April into April. For a flow
like jobless claims, spread each week over its seven days and sum by month:

```python
def weekly_flow_to_monthly(w: pd.Series) -> pd.Series:
    """Distribute each weekly total evenly over its days, then sum by month."""
    days = pd.date_range(w.index[0] - pd.Timedelta(days=6), w.index[-1], freq="D")
    daily = (w / 7).reindex(days, method="bfill")  # each day gets 1/7 of its week
    monthly = daily.resample("MS").sum()
    return monthly.iloc[1:-1]  # drop partial first/last months


naive = claims.resample("MS").sum()
exact = weekly_flow_to_monthly(claims)
print(
    pd.DataFrame({"by week-end date": naive, "day-weighted": exact})
    .loc["2025-01":"2025-06"]
    .round(-3)
)
```

For a weekly *average* concept (a rate, a price), use the day-weighted
`mean` instead of `sum`.

## Fiscal years and custom periods

Anchored aliases do fiscal years directly. A fiscal year ending in September
(the US federal year) labelled by the year in which it ends:

```python
fy = retail.resample("YE-SEP").sum(min_count=12)
print(fy.tail(3).round(-3))
```

Any custom grouping works with `groupby`, e.g. half-years:

```python
half = retail.groupby([retail.index.year, (retail.index.month - 1) // 6 + 1]).sum(
    min_count=6
)
print(half.tail(4))
```

## Monthly to quarterly, every option

The Empirical Macro Toolbox's `m2q` offers mean, median, start and end of
quarter. In pandas:

```python
m2q = pd.DataFrame(
    {
        rule: unrate.resample("QS").agg(rule)
        for rule in ["mean", "median", "first", "last"]
    }
)
print(m2q.loc["2020"].round(1))
```

## Pitfalls

* **Never aggregate after seasonal adjustment** of a component and then
  compare it to an aggregate adjusted directly: indirect and direct seasonal
  adjustment differ (chapter 5).
* Aggregating growth rates is not aggregating levels. Aggregate the levels,
  then compute the growth rate.
* Averaging a ratio (e.g. debt/GDP) is not the ratio of averages. Aggregate
  numerator and denominator separately.
* Temporal aggregation changes the dynamics: averaging introduces an MA
  component and smooths turning points (Working, 1960). Keep this in mind
  before reading persistence off quarterly averages of monthly data.
