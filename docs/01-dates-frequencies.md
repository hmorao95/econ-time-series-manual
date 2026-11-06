# 1. Dates, periods and frequencies

Most bugs in applied time-series work are date bugs: a quarter stamped on its
first day in one file and its last day in another, a fiscal year that ends in
June, a holiday that silently drops out of a daily series. Get the index right
first; everything else in this manual assumes it.

!!! tip "Quick rules"
    * Use a **`PeriodIndex`** for economic data that describe a span (a month,
      a quarter, a year). Use a **`DatetimeIndex`** for data observed at an
      instant (daily prices, tick data).
    * Make the frequency **explicit** (`.asfreq`, `.to_period`). An index with
      `freq=None` hides gaps.
    * Merge sources on **periods**, never on raw timestamps.

## Load a series

Every chapter uses `tsmanual.load`, which reads a bundled snapshot of FRED.
Pass `online=True` for the latest vintage, or any other FRED identifier.

```python
import numpy as np
import pandas as pd
from tsmanual import load

gdp = load("GDPC1")  # real GDP, quarterly
ip = load("INDPRO")  # industrial production, monthly
brent = load("DCOILBRENTEU")  # Brent spot, daily
print(gdp.tail(3))
print(gdp.index.freq)  # None
print(gdp.asfreq("QS").index.freq)  # <QuarterBegin: startingMonth=1>
```

FRED stamps each period on its **first day** (`2026-07-01` is 2026Q3) and the
index carries no frequency (`freq` is `None`). `asfreq("QS")` declares it:
quarters starting in January, April, July and October.

## Timestamps vs periods

```python
gdp_q = gdp.to_period("Q")  # 2026-07-01 -> 2026Q3
print(gdp_q.tail(3))
print(gdp_q.index[-1].start_time, gdp_q.index[-1].end_time)

# Back to timestamps, at either end of the period
gdp_end = gdp_q.to_timestamp(how="end")
gdp_end.index = gdp_end.index.normalize()  # 2026-09-30 00:00
```

A period knows its own span, so arithmetic is calendar-aware:

```python
p = pd.Period("2024Q4", freq="Q")
print(p + 1, p.asfreq("M", how="start"), p.asfreq("M", how="end"))
print(pd.period_range("2024-01", periods=4, freq="M"))
```

## Frequency aliases

| Meaning | Timestamp alias (pandas ≥ 2.2) | Period alias |
|---|---|---|
| Daily / business daily | `D` / `B` | `D` / `B` |
| Weekly, ending Saturday | `W-SAT` | `W-SAT` |
| Month start / end | `MS` / `ME` | `M` |
| Quarter start / end | `QS` / `QE` | `Q` |
| Year start / end | `YS` / `YE` | `Y` |
| Fiscal year ending June | `YE-JUN` | `Y-JUN` |
| Fiscal quarters, year ending June | `QE-JUN` | `Q-JUN` |

The old aliases `M`, `Q`, `A` for *timestamps* were deprecated in pandas 2.2.
Use the `-E`/`-S` forms.

## Make gaps visible

`asfreq` reindexes to a regular grid, inserting `NaN` for missing periods.
Without it, a missing month is invisible and `shift(1)` jumps over it.

```python
broken = ip["2019"].drop(pd.Timestamp("2019-06-01"))
print(len(broken), broken.index.freq)  # 11 obs, no freq
regular = broken.asfreq("MS")
print(regular.isna().sum())  # the June gap is now a NaN
```

## Business days and holidays

Daily market data skip weekends and are `NaN` on holidays. Decide whether a
holiday is "no observation" (drop it) or "unchanged price" (carry it forward).

```python
b = brent["2024-12-20":"2025-01-03"]
print(b)  # NaN on Christmas and New Year's Day
print(b.isna().sum(), "holidays in the window")

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

us_bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
trading_days = pd.date_range("2025-01-01", "2025-01-31", freq=us_bday)
print(len(trading_days), "US federal business days in January 2025")
```

## Fiscal years

A fiscal year ending in June is the period alias `Y-JUN`. Its label is the
calendar year in which it **ends**.

```python
months = pd.period_range("2024-05", "2024-08", freq="M")
print(pd.Series(months.asfreq("Y-JUN"), index=months))  # FY2024 -> FY2025 in July
fq = pd.Period("2024-08", freq="M").asfreq("Q-JUN")
print(fq, "fiscal year", fq.qyear, "quarter", fq.quarter)  # 2025Q1
```

## Selecting and slicing

```python
print(gdp["2008":"2009"].round(0))  # partial-string indexing
print(gdp_q.loc["2020Q1":"2020Q4"].round(0))  # same on a PeriodIndex
recent = gdp[gdp.index >= "2015"]
```

## Merging sources with different date conventions

A typical failure: one source stamps months on the first day, another on the
last. A timestamp join silently returns nothing.

```python
a = load("UNRATE")["2024"]  # 2024-01-01, ...
b = load("PAYEMS")["2024"]
b.index = b.index + pd.offsets.MonthEnd(0)  # 2024-01-31, ...
print(pd.concat([a, b], axis=1, join="inner").shape)  # (0, 2): no overlap!

merged = pd.concat([a.to_period("M"), b.to_period("M")], axis=1, join="inner")
print(merged.shape)  # (12, 2)
```

Convert to periods at load time and the problem disappears.

## Pitfalls

* `pd.to_datetime("01/02/2024")` is January 2nd; European files mean
  February 1st. Pass `format="%d/%m/%Y"` or `dayfirst=True`.
* Spreadsheet dates like `2024Q1`, `2024M01` or `2024-W05`: parse with
  `pd.PeriodIndex(..., freq="Q")`, or fix the string first
  (`s.str.replace("M", "-")`).
* Time zones only matter for intraday data. Localize once
  (`tz_localize("Europe/Lisbon")`) and convert (`tz_convert`); never mix
  naive and aware stamps.
