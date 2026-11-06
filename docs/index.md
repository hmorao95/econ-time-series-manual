# Economic Time Series in Python

A practical manual for economists who manipulate time series in Python:
the recipes you need between downloading a series and estimating a model.
Each chapter is short, starts with **quick rules**, and gives code that runs
as written on real US data.

| Chapter | What you will find |
|---|---|
| [1. Dates and frequencies](01-dates-frequencies.md) | `Period` vs `Timestamp`, frequency aliases, gaps, business days, fiscal years, merging sources |
| [2. Transformations](02-transformations.md) | growth rates (q/q, SAAR, y/y, Q4/Q4), carry-over, rebasing, deflating, splicing |
| [3. Temporal aggregation](03-aggregation.md) | flows vs stocks vs averages, incomplete periods, weekly → monthly, fiscal years |
| [4. Interpolation and disaggregation](04-interpolation.md) | filling gaps, Kalman smoother, Denton, Chow–Lin, Fernández |
| [5. Seasonal adjustment](05-seasonal-adjustment.md) | testing for seasonality, classical, STL, trading-day and Easter effects, X-13, MSTL |
| [6. Detrending and filtering](06-detrending.md) | polynomial trends, HP (two- and one-sided), Hamilton, BK, CF, Beveridge–Nelson, wavelets, UC models |
| [7. Rolling windows and smoothing](07-rolling-smoothing.md) | moving averages, EWMA, 3m/3m, rolling volatility and regressions, lags and leads |
| [8. Business cycles](08-business-cycles.md) | BBQ turning-point dating, phase statistics, stylized facts |
| [9. Diagnostics](09-diagnostics.md) | ADF/KPSS, structural breaks, ACF/PACF, outliers |
| [10. Crisis dummies and recession bands](10-crisis-dummies.md) | Portugal: reading a chronology, grey and hatched bands, episode and slope dummies, event studies, pulse/step dummies, Markov-switching probabilities |
| [Cheat sheet](cheatsheet.md) | one-line answers to the most common tasks |

## Setup

```bash
git clone https://github.com/hmorao95/econ-time-series-manual
cd econ-time-series-manual
uv sync                     # Python >= 3.10, pandas, statsmodels, scipy, matplotlib
uv run mkdocs serve         # browse the manual at http://127.0.0.1:8000
```

Without `uv`: `pip install -e .` and `pip install mkdocs-material`.

## Data

Every example uses a snapshot of FRED series bundled with the repository, so
the code runs offline and gives the numbers quoted in the text:

```python
from tsmanual import load, SERIES

gdp = load("GDPC1")  # bundled snapshot
cpi = load("CPIAUCSL", online=True)  # latest vintage from FRED
```

`SERIES` lists what is bundled: real and nominal GDP, the GDP deflator,
industrial production, retail sales (adjusted and not), CPI, unemployment
(adjusted and not), payrolls, weekly jobless claims, daily Brent and the
NBER recession indicator, plus Portuguese GDP, unemployment, the 10-year
bond yield and a quarterly Portuguese recession chronology
(`DATA_DIR / "pt_recessions.txt"`). `load(..., online=True)` fetches any FRED series.

## Conventions

* Series that grow are analysed in **logs × 100**, so differences are
  (approximately) percent changes.
* Code blocks within a chapter run in sequence and share variables.
* Every code block is executed in CI against the bundled data, so the code in
  this manual works with the pinned library versions.

## Credits

Several recipes in chapters 3, 6, 7 and 8 port functions of Filippo Ferroni's
[Empirical Macro Toolbox](https://naffe15.github.io/EMT/) (MATLAB) to
Python; its [function map](https://naffe15.github.io/_pages/function_map.html)
lists the originals. Data: FRED, Federal Reserve Bank of St. Louis.
