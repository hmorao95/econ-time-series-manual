# The `econts` toolkit

`econts` packages the recipes of this book as functions for your own
projects. All of them take and return pandas objects.

```bash
pip install git+https://github.com/hmorao95/econ-time-series-manual
```

Pin a commit or tag for reproducible projects, e.g.
`pip install "econts @ git+https://github.com/hmorao95/econ-time-series-manual@<commit>"`.
Each function has a docstring (`help(econts.filters.hamilton_filter)`).

## Data: `econts`

| Function | Does | Chapter |
|---|---|---|
| `load(series_id, online=False)` | bundled FRED snapshot, or any FRED series online | all |
| `SERIES`, `DATA_DIR` | what is bundled, and where | all |

## Transformations: `econts.transform`

| Function | Does | Chapter |
|---|---|---|
| `growth(s, kind)` | `"pop"`, `"yoy"`, `"annualized"`, `"log"`, `"log_annualized"`; never fills gaps | 2 |
| `carry_over(q, year)` | statistical overhang | 2 |
| `rebase(s, base)` | index with base-period average = 100 | 2 |
| `deflate(nominal, prices)` | real series | 2 |
| `ratio_splice(old, new, at)` | link series with a level break | 2 |
| `demean(s, by=None)` | full-sample or by-group demeaning (EMT `demean`) | 2 |
| `within(df, value, unit, time=None)` | panel within and two-way transformations | 2 |
| `zscore(s)`, `minmax(s)` | standardize, rescale to [0, 1] | 2 |

## Aggregation: `econts.aggregate`

| Function | Does | Chapter |
|---|---|---|
| `aggregate(s, freq, how, min_obs)` | resample with a minimum observation count | 3 |
| `weekly_to_monthly(w, how)` | split weeks across months by day | 3 |

## Disaggregation: `econts.disagg`

| Function | Does | Chapter |
|---|---|---|
| `denton(Y, indicator, m, how)` | proportional Denton benchmarking, with extrapolation | 4 |
| `chow_lin(Y, X, m, how)` | GLS on indicators, AR(1) errors | 4 |
| `fernandez(Y, X, m, how)` | GLS on indicators, random-walk errors | 4 |
| `agg_matrix(n_low, n_high, m, how)` | the aggregation matrix | 4 |

## Seasonal adjustment: `econts.seasonal`

| Function | Does | Chapter |
|---|---|---|
| `stl_adjust(s, calendar=False)` | robust STL in logs, optionally with trading-day and Easter effects | 5 |
| `calendar_regressors(index)` | weekday contrasts, month length, Easter | 5 |

## Trend and cycle: `econts.filters`

Each returns `(cycle, trend)`; apply to `100 * log(level)`.

| Function | Does | Chapter |
|---|---|---|
| `poly_detrend(y, order, break_at)` | polynomial trend, optional break (EMT `polydet`, `polydet_break`) | 6 |
| `hp(y, lamb)`, `hp_one_sided(y, lamb)` | two- and one-sided HP (EMT `one_sided_hpfilter_serial`) | 6 |
| `hamilton_filter(y, h, p)` | Hamilton (2018) regression filter (EMT `hamfilter`) | 6 |
| `bk(y)`, `cf(y)` | Baxter-King and Christiano-Fitzgerald band-pass | 6 |
| `beveridge_nelson(y, p)` | univariate BN (EMT `BNuniv`) | 6 |
| `bn_bq_bivariate(y, z, p)` | bivariate BN and Blanchard-Quah (EMT `BNBQbiv`) | 6 |
| `haar_components(y)`, `haar_cycle(y)` | Haar wavelet details and cycle (EMT `wavefilter`) | 6 |

## Business cycles: `econts.cycles`

| Function | Does | Chapter |
|---|---|---|
| `bbq(y, window, phase, cycle)` | Harding-Pagan turning points (EMT `date_`) | 8 |
| `phases(y, turning_points)`, `states(turning_points)` | phase durations and amplitudes, expansion indicator | 8 |
| `cycle_moments(cycles, reference)` | std, relative std, AR(1), cross-correlations (EMT `autocor`) | 8 |
| `robust_skewness(s)`, `robust_kurtosis(s)` | classical and quantile-based moments (EMT `skewness_`, `kurtosis_`) | 8 |
| `pc_factors(panel, n)` | principal-component factors, F'F/T = I (EMT `pc_T`) | 8 |
| `concordance(a, b)` | share of periods two chronologies agree | 10 |

## Dummies and bands: `econts.dummies`

| Function | Does | Chapter |
|---|---|---|
| `episodes(dummy)` | runs of ones as a table with start, end, length, peak | 10 |
| `dummy_from_episodes(index, spans)` | dummy from a list of dates | 10 |
| `episode_labels(dummy)` | one label per episode, for `get_dummies` or `groupby` | 10 |
| `pulse`, `step`, `ramp`, `temporary_change` | outlier and break dummies | 10 |
| `shade(ax, dummy, **style)` | grey or hatched bands covering whole periods (EMT `shade`) | 6, 10 |

## Diagnostics: `econts.diagnostics`

| Function | Does | Chapter |
|---|---|---|
| `unit_root_table(series)` | ADF and KPSS with a joint verdict | 9 |
| `robust_zscore`, `robust_outliers` | rolling median and MAD outlier flags | 9 |
| `winsorize(s, lower, upper)` | clip at quantiles | 9 |

## SVAR helpers: `econts.svar`

| Function | Does | Chapter |
|---|---|---|
| `irf_with_bands(var, shock, cumulate, impact)` | Cholesky responses, scaled shock, Monte Carlo bands | 11 |
| `historical_decomposition(var)` | contribution of each shock, period by period | 11 |
