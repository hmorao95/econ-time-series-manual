# Cheat sheet

One-line answers to common tasks. For the packaged versions of the longer
recipes, see the [econts toolkit](toolkit.md).

`s` is a pandas Series with a regular `DatetimeIndex` (or `PeriodIndex`);
`import numpy as np, pandas as pd, statsmodels.api as sm`.

## Dates and frequencies

| Task | Code |
|---|---|
| Timestamps → periods | `s.to_period("Q")` |
| Periods → timestamps (end) | `s.to_timestamp(how="end")` |
| Expose gaps as NaN | `s.asfreq("MS")` |
| Month of a fiscal year ending June | `pd.Period("2024-08", "M").asfreq("Y-JUN")` |
| Business days with US holidays | `pd.date_range(a, b, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))` |

## Transformations

| Task | Code |
|---|---|
| Growth, period on period (%) | `100 * s.pct_change()` |
| Year on year, monthly data (%) | `100 * s.pct_change(12)` |
| Annualized q/q (%) | `100 * ((1 + s.pct_change()) ** 4 - 1)` |
| Log difference × 100 | `100 * np.log(s).diff()` |
| 3m/3m annualized | `a = s.rolling(3).mean(); 100 * ((a / a.shift(3)) ** 4 - 1)` |
| Rebase to 2015 = 100 | `100 * s / s["2015"].mean()` |
| Deflate | `nominal / deflator * 100` |
| Demean | `s - s.mean()` |
| Demean by regime / by month | `s - s.groupby(key).transform("mean")` |
| Panel within transformation | `df.u - df.groupby("country").u.transform("mean")` |
| Z-score | `(s - s.mean()) / s.std()` |
| Min-max | `(s - s.min()) / (s.max() - s.min())` |

## Aggregation (high → low frequency)

| Series is a… | Code |
|---|---|
| flow | `s.resample("QS").sum(min_count=3)` |
| flow at annual rate (SAAR) | `s.resample("YS").mean()` |
| stock | `s.resample("QS").last()` |
| rate, price, index | `s.resample("QS").mean()` |
| daily OHLC | `s.resample("MS").ohlc()` |

## Interpolation (filling and disaggregation)

| Task | Code / method |
|---|---|
| Fill gaps proportionally to time | `s.interpolate(method="time")` |
| Carry last price forward | `s.ffill(limit=5)` |
| Shape-preserving | `s.interpolate(method="pchip")` |
| Gaps with uncertainty | `sm.tsa.UnobservedComponents(s, "local linear trend")`, `.level.smoothed` |
| Quarterly → monthly, adds up | Denton (chapter 4) |
| … with regression on indicators | Chow–Lin / Fernández (chapter 4) |

## Seasonal adjustment

| Task | Code |
|---|---|
| Seasonal means | `s.pct_change().groupby(s.index.month).mean()` |
| Classical | `seasonal_decompose(s, model="multiplicative", period=12)` |
| STL in logs | `STL(np.log(s), period=12, robust=True).fit().seasonal` |
| Several seasonal periods | `MSTL(s, periods=(24, 168)).fit()` |
| X-13ARIMA-SEATS | `x13_arima_analysis(s, x12path=...)` |

## Trend and cycle (quarterly, `y = 100 * np.log(s)`)

| Filter | Code |
|---|---|
| HP | `cycle, trend = sm.tsa.filters.hpfilter(y, lamb=1600)` |
| One-sided HP | Kalman filter of `UnobservedComponents(y, "smooth trend")` with variances `[1, 1/1600]` |
| Hamilton | OLS of `y` on `y.shift(8)` … `y.shift(11)` |
| Baxter–King | `sm.tsa.filters.bkfilter(y, low=6, high=32, K=12)` |
| Christiano–Fitzgerald | `sm.tsa.filters.cffilter(y, low=6, high=32, drift=True)` |
| Beveridge–Nelson, bivariate | VAR on growth and a cyclical variable, $-e_1'F(I-F)^{-1}Z_t$ (chapter 6) |
| Blanchard–Quah | VAR with long-run impact lower triangular (chapter 6) |
| HP λ | 6.25 annual, 1600 quarterly, 129600 monthly |

## Rolling windows

| Task | Code |
|---|---|
| Trailing / centered mean | `s.rolling(12).mean()` / `s.rolling(13, center=True).mean()` |
| EWMA | `s.ewm(halflife=3).mean()` |
| Annualized volatility, daily | `r.rolling(21).std() * np.sqrt(252)` |
| Calendar-time window | `s.rolling("30D").mean()` |
| Rolling regression | `RollingOLS(y, X, window=40).fit()` |

## Dummies and bands

| Task | Code |
|---|---|
| Parse day-first dates | `pd.to_datetime(s, format="%d/%m/%Y")` |
| Shade a quarterly dummy on any chart | `ax.axvspan(q.start_time, q.end_time, color="0.85", lw=0)` per episode |
| Hatched band | `ax.axvspan(a, b, fc="none", ec="C3", hatch="///")` |
| Pulse / step dummy | `(idx == pd.Period(at, "Q")).astype(float)` / `(idx >= ...)` |
| Slope dummy (interaction) | `smf.ols("du ~ g * rec", data=df)` |
| Episode dummies | `pd.get_dummies(episode_label, dtype=float)` |
| Recession probability | `sm.tsa.MarkovRegression(g, k_regimes=2).fit().smoothed_marginal_probabilities` |

## Paper figures (`import housestyle as hs`)

| Task | Code |
|---|---|
| Grid of panels | `fig, axes = hs.figure(2, 4)` |
| Impulse response panel | `hs.irf(ax, h, med, lo, hi, "Title", "%", "Months")` |
| Series k in house order | `hs.line(ax, x, y, k)` (1 black, 2 orange dashed, 3 green dash-dot) |
| Band | `hs.band(ax, x, lo, hi)` |
| Shaded episode / event | `hs.shade(ax, start, end)` / `hs.event(ax, x, y, "label")` |
| Stacked decomposition | `hs.bars(ax, dates, contributions)` |
| Labels, ticks, zero line | `hs.finish(ax, "Title", "Units", "Months")` |
| 300 dpi export | `hs.export(fig, "name")` |
| Cumulative IRF bands | `var.irf(24).cum_errband_mc(orth=True, signif=0.32)` |

## Cycle statistics

| Task | Code |
|---|---|
| Bowley skewness | `(q3 + q1 - 2 * q2) / (q3 - q1)` |
| Moors kurtosis (normal: 1.23) | octile ratio, chapter 8 |
| First principal component | `U, S, Vt = np.linalg.svd(Z)`; `U[:, 0] * np.sqrt(T)` |

## Diagnostics

| Task | Code |
|---|---|
| ADF p-value | `adfuller(s, regression="ct", result_object=True).pvalue` |
| KPSS p-value | `kpss(s, regression="c", result_object=True).pvalue` |
| Break + unit root | `zivot_andrews(s.to_numpy(), regression="ct")` |
| White noise | `sm.stats.acorr_ljungbox(s, lags=[4, 8])` |
