# Economic Time Series in Python

[![CI](https://github.com/hmorao95/econ-time-series-manual/actions/workflows/ci.yml/badge.svg)](https://github.com/hmorao95/econ-time-series-manual/actions/workflows/ci.yml)

A practical manual on manipulating economic time series in Python: dates and
frequencies, growth rates, temporal aggregation, interpolation and temporal
disaggregation, seasonal adjustment, detrending, business-cycle dating and
diagnostics. Each chapter opens with quick rules, then gives code that runs
as written on real US data (a bundled FRED snapshot).

![Business-cycle components of US GDP](docs/img/cycles.png)

## Contents

1. [Dates, periods and frequencies](docs/01-dates-frequencies.md)
2. [Transformations: logs, growth rates, indices, deflators](docs/02-transformations.md)
3. [Temporal aggregation](docs/03-aggregation.md)
4. [Interpolation and temporal disaggregation](docs/04-interpolation.md) (Denton, Chow–Lin, Fernández, Kalman smoother)
5. [Seasonal adjustment](docs/05-seasonal-adjustment.md) (classical, STL, trading-day and Easter effects, X-13, MSTL)
6. [Detrending and filtering](docs/06-detrending.md) (HP two- and one-sided, Hamilton, BK, CF, Beveridge–Nelson, Haar wavelets, UC models)
7. [Rolling windows, smoothing, lags and leads](docs/07-rolling-smoothing.md)
8. [Business cycles](docs/08-business-cycles.md) (BBQ turning points, phase statistics, stylized facts)
9. [Diagnostics](docs/09-diagnostics.md) (ADF/KPSS, Zivot–Andrews, ACF/PACF, outliers)
10. [Crisis dummies and recession bands](docs/10-crisis-dummies.md) (Portugal: grey bands, episode and slope dummies, event studies, Markov-switching probabilities)
11. [Paper-quality figures](docs/11-paper-figures.md) (SVAR impulse responses, comparisons, historical decompositions, drawn with [housestyle](https://github.com/hmorao95/housestyle))
12. [Cheat sheet](docs/cheatsheet.md)

The chapters read fine on GitHub. For the rendered site with search and
math, build it locally (below).

## Quickstart

```bash
git clone https://github.com/hmorao95/econ-time-series-manual
cd econ-time-series-manual
uv sync
uv run mkdocs serve          # http://127.0.0.1:8000
```

```python
from tsmanual import load

gdp = load("GDPC1")  # bundled FRED snapshot
cpi = load("CPIAUCSL", online=True)  # latest vintage from FRED
```

## Development

Every `python` block in every chapter is executed by the tests, with warnings
treated as errors, so the code in the manual cannot drift from working code.

```bash
uv run pytest                                    # run every chapter
uv run python -m tsmanual.runner --figures       # regenerate docs/img
uv run python scripts/fetch_data.py              # refresh the FRED snapshot
uv run mkdocs build --strict
```

Chapter conventions (see `src/tsmanual/runner.py`): blocks within a chapter
share one namespace; `# figure: name` saves the current figure to
`docs/img/name.png`; a first line `# requires: x13as` skips the block when
that executable is missing.

Refreshing the data can change numbers quoted in the text (FRED revises
history), so rerun the tests and reread the quoted figures after a refresh.

## Credits

Several recipes port functions of Filippo Ferroni's
[Empirical Macro Toolbox](https://naffe15.github.io/EMT/) (MATLAB) to Python.
Data: FRED, Federal Reserve Bank of St. Louis (series from BEA, BLS, Census,
the Federal Reserve Board, DOL and EIA).

## License

MIT.
