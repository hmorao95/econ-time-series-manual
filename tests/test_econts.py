"""Tests for the econts toolkit, on the bundled data and on simulated series."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import econts
from econts import (
    aggregate,
    cycles,
    diagnostics,
    disagg,
    dummies,
    filters,
    load,
    seasonal,
    svar,
)
from econts import transform as tf


@pytest.fixture(scope="module")
def gdp():
    return load("GDPC1").asfreq("QS")


@pytest.fixture(scope="module")
def y(gdp):
    return 100 * np.log(gdp)


@pytest.fixture(scope="module")
def rec():
    return load("USRECQ").asfreq("QS")


# --- data -----------------------------------------------------------------------------


@pytest.mark.parametrize("series_id", econts.SERIES)
def test_bundled_series_load(series_id):
    s = load(series_id)
    assert s.name == series_id
    assert s.index.is_monotonic_increasing
    assert s.notna().sum() > 100


# --- transform ------------------------------------------------------------------------


def test_growth_kinds(gdp):
    q = gdp.to_period("Q")
    assert tf.periods_per_year(q.index) == 4
    assert tf.periods_per_year(gdp.index) == 4
    g = tf.growth(q, "pop")
    assert np.isclose(
        tf.growth(q, "annualized")["2024Q1"], 100 * ((1 + g["2024Q1"] / 100) ** 4 - 1)
    )
    assert np.isclose(
        tf.growth(q, "yoy")["2024Q1"], 100 * (q["2024Q1"] / q["2023Q1"] - 1)
    )
    assert np.isclose(
        tf.growth(q, "log_annualized")["2024Q1"], 4 * tf.growth(q, "log")["2024Q1"]
    )


def test_growth_does_not_fill_gaps():
    cpi = load("CPIAUCSL").to_period("M")
    assert np.isnan(tf.growth(cpi, "pop")["2025-11"])  # October 2025 is missing


def test_rebase_deflate_splice():
    ip = load("INDPRO")
    assert np.isclose(tf.rebase(ip, "2015").loc["2015"].mean(), 100)
    real = tf.deflate(load("GDP"), load("GDPDEF"))
    assert (real / load("GDPC1") - 1).abs().max() < 1e-3
    long = tf.ratio_splice(1.37 * ip[:"2012"], ip["2005":])
    assert (long / ip - 1).abs().max() < 1e-12


def test_carry_over_flat_year():
    idx = pd.period_range("2024Q1", "2025Q4", freq="Q")
    s = pd.Series([1, 2, 3, 4, 4, 4, 4, 4], index=idx, dtype=float)
    assert np.isclose(tf.carry_over(s, 2025), 100 * (4 / 2.5 - 1))


def test_demean_and_scalings():
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 10.0, 12.0])
    assert np.isclose(tf.demean(s).mean(), 0)
    by = np.array([0, 0, 0, 1, 1, 1])
    assert np.allclose(tf.demean(s, by).groupby(by).mean(), 0)
    assert np.isclose(tf.zscore(s).std(), 1)
    assert tf.minmax(s).min() == 0 and tf.minmax(s).max() == 1
    panel = pd.DataFrame(
        {"c": list("aabb"), "t": [1, 2, 1, 2], "v": [1.0, 3.0, 10.0, 14.0]}
    )
    assert np.allclose(tf.within(panel, "v", "c").groupby(panel["c"]).mean(), 0)
    assert np.allclose(tf.within(panel, "v", "c", "t").groupby(panel["t"]).mean(), 0)


# --- aggregate ------------------------------------------------------------------------


def test_aggregate_min_obs():
    brent = load("DCOILBRENTEU")
    m = aggregate.aggregate(brent, "MS", "mean", min_obs=15)
    assert np.isnan(m.iloc[-1])  # bundled data end early in a month
    assert np.isclose(m["2024-01-01"], brent["2024-01"].mean())


def test_weekly_to_monthly_splits_weeks_by_day():
    weeks = pd.date_range("2024-01-06", "2024-12-28", freq="W-SAT")
    monthly = aggregate.weekly_to_monthly(pd.Series(7.0, index=weeks))
    assert (monthly == monthly.index.days_in_month).all()  # 1 per day
    real = aggregate.weekly_to_monthly(load("ICSA")["2020":"2024"], how="mean")
    assert real.notna().all()


# --- disagg ---------------------------------------------------------------------------


def test_denton_adds_up_and_extrapolates(gdp):
    Y = gdp["2015":"2019"]
    ip = load("INDPRO")["2015":"2020-03"]  # one quarter beyond Y
    x = disagg.denton(Y, ip)
    assert len(x) == len(ip)
    assert (x[: 3 * len(Y)].resample("QS").mean() / Y - 1).abs().max() < 1e-12


@pytest.mark.parametrize("method", [disagg.chow_lin, disagg.fernandez])
def test_regression_disagg_adds_up(gdp, method):
    Y = gdp["2010":"2019"]
    X = pd.DataFrame({"ip": load("INDPRO"), "pay": load("PAYEMS")})["2010":"2019"]
    x, info = method(Y, X)
    assert (x.resample("QS").mean() / Y - 1).abs().max() < 1e-10
    assert 0 <= info["rho"] <= 1


def test_agg_matrix_sum():
    C = disagg.agg_matrix(2, 7, m=3, how="sum")
    assert C.shape == (2, 7) and C.sum() == 6 and C[:, 6].sum() == 0


# --- seasonal -------------------------------------------------------------------------


def test_calendar_regressors_columns():
    X = seasonal.calendar_regressors(
        pd.date_range("2024-01-01", "2024-12-01", freq="MS")
    )
    assert list(X.columns) == [*seasonal.WEEKDAYS, "length", "easter"]
    assert X.loc["2024-03-01", "easter"] == 1.0  # Easter 2024: 31 March


def test_stl_calendar_closer_to_official():
    nsa, sa = load("RSXFSN").asfreq("MS"), load("RSXFS").asfreq("MS")
    plain = seasonal.stl_adjust(nsa)["sa"]
    cal = seasonal.stl_adjust(nsa, calendar=True)["sa"]

    def gap(a):
        return (100 * np.log(a).diff() - 100 * np.log(sa).diff()).pow(2).mean() ** 0.5

    assert gap(cal) < 0.7 * gap(plain)


# --- filters --------------------------------------------------------------------------


def test_filters_shapes_and_identities(y):
    for c, t in [
        filters.poly_detrend(y, 2, "1973-10-01"),
        filters.hp(y),
        filters.cf(y),
        filters.hamilton_filter(y),
        filters.beveridge_nelson(y),
    ]:
        assert c.index.equals(y.index)
        assert np.allclose((c + t).dropna(), y[(c + t).notna()])
    assert filters.bk(y).isna().sum() == 24
    c, t = filters.haar_cycle(y)  # trend also drops D1 and D2
    D = filters.haar_components(y, levels=2)
    assert np.allclose((c + t + D.sum(axis=1)).dropna(), y[15:])


def test_hp_one_sided_equals_kalman_filter(y):
    c1, _ = filters.hp_one_sided(y)
    res = sm.tsa.UnobservedComponents(y, level="smooth trend").smooth([1.0, 1 / 1600])
    kf_cycle = y - res.level.filtered
    assert (c1 - kf_cycle).iloc[20:].abs().max() < 1e-3


def test_bivariate_bn_bq_negative_in_recessions(y, rec):
    u = load("UNRATE").resample("QS").mean()
    out = filters.bn_bq_bivariate(y["1948":"2019"], u["1948":"2019"])
    means = out.groupby(rec.reindex(out.index)).mean()
    assert (means.loc[1.0] < 0).all() and (means.loc[0.0] > 0).all()


def test_haar_detail_of_linear_trend():
    s = pd.Series(0.5 * np.arange(100.0))
    D = filters.haar_components(s)
    assert np.allclose(D["D3"].dropna(), 0.5 * 2 ** (3 - 2))  # g * 2^(j-2)


# --- cycles ---------------------------------------------------------------------------


def test_bbq_matches_most_nber_peaks(y):
    tp = cycles.bbq(y.to_period("Q"))
    peaks = set(tp.index[tp == 1].astype(str))
    assert {
        "1953Q2",
        "1957Q3",
        "1973Q4",
        "1980Q1",
        "1981Q3",
        "1990Q3",
        "2019Q4",
    } <= peaks
    ph = cycles.phases(y.to_period("Q"), tp)
    assert set(ph["phase"]) == {"expansion", "contraction"}
    assert (ph.loc[ph.phase == "contraction", "amplitude"] < 0).all()
    st = cycles.states(tp)
    assert set(st.unique()) <= {0, 1}


def test_concordance():
    a = pd.Series([0, 1, 1, 0])
    assert cycles.concordance(a, pd.Series([0, 1, 0, 0])) == 0.75


def test_robust_moments_normal():
    s = pd.Series(np.random.default_rng(0).normal(size=200_000))
    assert cycles.robust_skewness(s).abs().max() < 0.02
    assert (cycles.robust_kurtosis(s) - cycles.KURTOSIS_NORMAL).abs().max() < 0.03


def test_pc_factors_normalization():
    rng = np.random.default_rng(1)
    f = rng.normal(size=300)
    panel = pd.DataFrame(
        {f"x{i}": f * (i + 1) + rng.normal(size=300) for i in range(5)}
    )
    F, L, share = cycles.pc_factors(panel, n=2)
    assert np.allclose(F.T @ F / len(F), np.eye(2))
    assert (L["F1"] > 0).all() and share[0] > 0.6
    moments = cycles.cycle_moments(panel, "x0")
    assert np.isclose(moments.loc["x0", "corr(t+0)"], 1)


# --- dummies --------------------------------------------------------------------------


def test_episodes_round_trip():
    idx = pd.period_range("2000Q1", periods=12, freq="Q")
    d = dummies.dummy_from_episodes(idx, [("2000Q3", "2001Q1"), ("2002Q2", "2002Q2")])
    ep = dummies.episodes(d)
    assert ep["length"].tolist() == [3, 1]
    assert str(ep["peak"].iloc[0]) == "2000Q2"
    assert dummies.episode_labels(d).dropna().unique().tolist() == ["2000-01", "2002"]


def test_dummy_shapes():
    idx = pd.period_range("2020Q1", periods=6, freq="Q")
    assert dummies.pulse(idx, "2020Q3").tolist() == [0, 0, 1, 0, 0, 0]
    assert dummies.step(idx, "2020Q3").tolist() == [0, 0, 1, 1, 1, 1]
    assert dummies.ramp(idx, "2020Q3").tolist() == [0, 0, 1, 2, 3, 4]
    assert np.allclose(
        dummies.temporary_change(idx, "2020Q3", 0.5), [0, 0, 1, 0.5, 0.25, 0.125]
    )


def test_shade_draws_whole_periods():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = pd.period_range("2000Q1", periods=8, freq="Q")
    fig, ax = plt.subplots()
    dummies.shade(
        ax, dummies.dummy_from_episodes(idx, [("2000Q2", "2000Q3")]), label="rec"
    )
    assert len(ax.patches) == 1
    plt.close(fig)


# --- diagnostics ----------------------------------------------------------------------


def test_unit_root_table(y):
    tab = diagnostics.unit_root_table(
        {"log gdp": (y[:"2019"], "ct"), "growth": (y.diff()[:"2019"], "c")}
    )
    assert tab.loc["log gdp", "verdict"] == "unit root"
    assert tab.loc["growth", "verdict"] == "stationary"


def test_outliers_and_winsorize():
    flags = diagnostics.robust_outliers(np.log(load("ICSA")))
    assert flags["2020-03-28"]
    s = pd.Series(np.arange(100.0))
    assert diagnostics.winsorize(s).max() < 99


# --- svar -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def var_results():
    data = (
        pd.DataFrame(
            {
                "ip": 100 * np.log(load("INDPRO")).diff(),
                "u": load("UNRATE"),
            }
        )
        .loc["1990":"2019"]
        .asfreq("MS")
    )
    return sm.tsa.VAR(data).fit(4)


def test_historical_decomposition_adds_up(var_results):
    contrib, det = svar.historical_decomposition(var_results)
    data = pd.DataFrame(
        var_results.endog[-len(det) :], index=det.index, columns=det.columns
    )
    total = contrib.T.groupby(level="variable").sum().T[det.columns]
    assert np.allclose(det + total, data)


def test_irf_with_bands_scaling(var_results):
    out = svar.irf_with_bands(
        var_results, shock=0, horizon=12, cumulate=[True, False], impact=1.0, repl=50
    )
    assert np.isclose(out["median"].iloc[0, 0], 1.0)
    assert out["lower"].shape == (13, 2)
    assert (out["lower"] <= out["upper"] + 1e-12).all().all()
