# Knowledge test

Two multiple-choice questions per chapter; open each answer box to check your choice.

## Chapter 1

**A monthly series has one month dropped from its index, so its `freq` is `None`. What does `s.pct_change()` report for the month after the gap?**
- A) `NaN`, because pandas detects the missing month
- B) The change over two months, computed against the last row present, with nothing to flag the gap
- C) An error, because the index is irregular
- D) The change against an interpolated value for the missing month

```{admonition} Answer
:class: note, dropdown
B
```

**You join `UNRATE` (months stamped on the first day) with a payrolls file whose months are stamped on the last day, and the inner join is empty. What is the robust fix?**
- A) Join with `how="outer"` and forward-fill the gaps
- B) Subtract `pd.offsets.MonthEnd(1)` from the second index
- C) Resample both series to daily frequency before joining
- D) Convert both indexes to monthly periods with `.to_period("M")` and join on those

```{admonition} Answer
:class: note, dropdown
D
```

## Chapter 2

**Which statement about annual GDP growth rates is correct?**
- A) Growth of the annual average (the press figure) and Q4-over-Q4 growth (used in FOMC projections) answer different questions and can differ by more than a percentage point
- B) Average-over-average and Q4/Q4 growth are always equal for data at a seasonally adjusted annual rate
- C) Q4/Q4 growth is the growth of the sum of the four quarters
- D) Average-over-average growth uses only the fourth quarter of each year

```{admonition} Answer
:class: note, dropdown
A
```

**The 2020Q2 fall in US real GDP is −28.0% at an annualized rate but −32.8 in annualized log points. What should you take from this?**
- A) One of the two numbers is a data error
- B) Log differences are always larger in absolute value than exact rates, so they are the safer choice
- C) The log approximation is poor for large changes: report exact rates and use log differences inside models
- D) Annualizing is not valid for quarters with negative growth

```{admonition} Answer
:class: note, dropdown
C
```

## Chapter 3

**You need quarterly values of a monthly stock, such as government debt outstanding. Which aggregation rule do the quick rules recommend?**
- A) `sum`
- B) `mean`
- C) `last` (the end-of-period value)
- D) `first`

```{admonition} Answer
:class: note, dropdown
C
```

**Why aggregate monthly retail sales with `resample("QS").sum(min_count=3)` rather than `.sum()`?**
- A) It converts the quarterly total to an annual rate
- B) Plain `sum` returns a total for a quarter with only one or two months of data (and 0 for an empty one); `min_count=3` returns `NaN` instead
- C) It drops quarters with fewer than three seasonal factors
- D) Plain `sum` propagates `NaN`, and `min_count` makes it skip missing values

```{admonition} Answer
:class: note, dropdown
B
```

## Chapter 4

**Why is `gdp_q.resample("MS").interpolate()` the wrong way to build monthly GDP from quarterly GDP?**
- A) It cannot handle data at a seasonally adjusted annual rate
- B) It only works on a `PeriodIndex`
- C) It extrapolates beyond the last published quarter
- D) The monthly values it produces do not average back to the published quarterly values

```{admonition} Answer
:class: note, dropdown
D
```

**You run Chow–Lin and the estimated AR(1) coefficient of the errors sits at its upper bound. What does the chapter suggest?**
- A) The residual behaves like a random walk: use Fernández (ρ = 1), which is built for target and indicators that do not cointegrate
- B) The indicators fit perfectly, so Denton with a constant indicator will do as well
- C) Widen the bound to allow ρ > 1 and re-estimate
- D) Disaggregate the growth rates instead of the levels

```{admonition} Answer
:class: note, dropdown
A
```

## Chapter 5

**Why does the chapter seasonally adjust retail sales in logs?**
- A) STL accepts only positive inputs
- B) The seasonal swing grows with the level of the series, so the seasonality is multiplicative
- C) Taking logs removes trading-day effects
- D) The official adjusted series is published in logs

```{admonition} Answer
:class: note, dropdown
B
```

**STL tracks official seasonally adjusted retail sales closely in levels, but month-on-month changes differ by 1.7 pp on average. What does the chapter identify as the main source of the gap?**
- A) STL loses observations at both ends of the sample
- B) STL keeps the seasonal pattern fixed over time
- C) Calendar effects, such as the number of each weekday in a month and the date of Easter, which STL cannot see
- D) The official series uses an additive model

```{admonition} Answer
:class: note, dropdown
C
```

## Chapter 6

**Why should a real-time output gap not be computed with the two-sided HP filter?**
- A) The trend at date t uses observations after t: at the end of the sample the trend is pulled toward the last observations, and the recent gap is revised as new data arrive
- B) λ = 1600 is only valid for annual data
- C) The two-sided HP filter loses 12 observations at each end
- D) The HP filter cannot be applied to logged data

```{admonition} Answer
:class: note, dropdown
A
```

**On US GDP, the univariate Beveridge–Nelson cycle from a low-order AR for growth is on average positive in NBER recessions. Why?**
- A) The AR is estimated on a sample that includes 2020
- B) The BN trend is a two-sided moving average that uses future data
- C) BN measures the growth cycle rather than the classical cycle
- D) Growth is positively autocorrelated, so a fall in growth lowers the long-run forecast (the trend) by more than it lowers output, leaving output above trend

```{admonition} Answer
:class: note, dropdown
D
```

## Chapter 7

**Which of these can safely be used as a regressor in a forecasting model?**
- A) `jobs.rolling(13, center=True).mean()`
- B) The Christiano–Fitzgerald cycle of the series, estimated on the full sample
- C) `jobs.rolling(12).mean()`, a trailing average
- D) Any of them, as long as `min_periods` is set

```{admonition} Answer
:class: note, dropdown
C
```

**In the 3m/3m annualized rate `100 * ((avg3 / avg3.shift(3)) ** 4 - 1)`, why is the exponent 4?**
- A) There are about four weeks in a month
- B) The ratio compares two averages three months apart, and a year has four such three-month spans
- C) It cumulates the last four monthly changes
- D) It converts a quarterly rate into a monthly one

```{admonition} Answer
:class: note, dropdown
B
```

## Chapter 8

**You run the BBQ algorithm on HP-filtered GDP instead of the level of log GDP. What are you dating?**
- A) The same classical cycle, with less noise
- B) Exactly the NBER recessions
- C) Expansions only, because detrending removes the troughs
- D) The growth cycle (deviations from trend), a different object with more turning points

```{admonition} Answer
:class: note, dropdown
D
```

**Adding the 2020 quarters to US GDP growth raises classical kurtosis from 4.7 to 20.7, while Moors' octile-based kurtosis does not move. What is the lesson?**
- A) Classical moments are dominated by a few extreme observations, so report a quantile-based measure next to the classical one
- B) Moors' kurtosis is biased toward its normal value of 3
- C) GDP growth has become normally distributed since 2020
- D) Quantile-based measures cannot detect fat tails

```{admonition} Answer
:class: note, dropdown
A
```

## Chapter 9

**For log GDP with a trend, ADF gives p = 0.60 and KPSS gives p = 0.01. What do you conclude?**
- A) Stationary: both tests reject
- B) Unclear: the two tests disagree
- C) Unit root: ADF does not reject its null of a unit root and KPSS rejects its null of stationarity
- D) Trend-stationary with a structural break

```{admonition} Answer
:class: note, dropdown
C
```

**Why does the outlier detector use the rolling median and the median absolute deviation (MAD) rather than the mean and standard deviation?**
- A) They are faster to compute on rolling windows
- B) Outliers inflate the mean and standard deviation and so hide themselves; the median and MAD are not dragged along by them
- C) The MAD equals the standard deviation for any distribution
- D) The median removes seasonality from the window

```{admonition} Answer
:class: note, dropdown
B
```

## Chapter 10

**Which dummy fits a one-quarter shock, such as a lockdown quarter?**
- A) A pulse dummy: 0 0 1 0 0
- B) A step dummy: 0 0 1 1 1
- C) A ramp: 0 0 1 2 3
- D) A slope dummy: the recession indicator interacted with GDP growth

```{admonition} Answer
:class: note, dropdown
A
```

**The Portuguese recession chronology ends in 2020Q4 and your data run to 2026Q2. How should the indicator be treated after 2020Q4?**
- A) Set it to 0, since no recession has been declared
- B) Forward-fill its last value
- C) Set it to 1 in quarters of negative GDP growth
- D) Leave it `NaN`: the state is unknown, so drop those quarters or extend the chronology explicitly

```{admonition} Answer
:class: note, dropdown
D
```

## Chapter 11

**In the oil SVAR, the real oil price, industrial production and CPI enter in log differences × 100 and unemployment in levels. To show effects on levels you should:**
- A) Cumulate all four responses
- B) Cumulate the responses of oil, industrial production and CPI, and plot unemployment's response as it is
- C) Cumulate only the unemployment response
- D) Cumulate nothing, since the responses are already in levels

```{admonition} Answer
:class: note, dropdown
B
```

**A recursive (Cholesky) SVAR gives a tight 68% band around an impulse response. What does the band tell you?**
- A) That the ordering of the variables is correct
- B) That 68% of future shocks will produce responses inside it
- C) The sampling uncertainty of the estimate given the identification, and nothing about whether the ordering is right
- D) That the response is significant at the 5% level

```{admonition} Answer
:class: note, dropdown
C
```
