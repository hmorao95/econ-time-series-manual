"""Command-line interface: ``econts <command> <source> [flags]``.

Built with `python-fire <https://github.com/google/python-fire>`_: every
method of :class:`_Cli` is a subcommand and every parameter a flag, so the
CLI mirrors the library. Examples::

    econts series
    econts load GDPC1 --out gdp.csv
    econts growth CPIAUCSL --kind yoy
    econts aggregate PAYEMS --freq QS --how mean
    econts filter GDPC1 --log --method hamilton --out cycle.csv
    econts disagg GDPC1 --indicators INDPRO,PAYEMS --method chow-lin --out gdp_m.csv
    econts seasonal RSXFSN --calendar --out sales_sa.csv
    econts bbq GDPC1 --log
    econts unitroot UNRATE
    econts outliers ICSA --log

A ``source`` is a CSV file (first column dates, then one or more value
columns; pick one with ``--column``) or a FRED series id (the bundled
snapshot when available, otherwise fetched from FRED). Results are printed;
``--out FILE.csv`` writes them instead.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from econts import aggregate as agg
from econts import cycles, diagnostics, disagg, filters, seasonal
from econts import transform as tf
from econts.data import SERIES, load

#: Boolean flags; a bare ``--flag`` is rewritten to ``--flag=True`` so Fire
#: does not take the next word (often the source) as its value.
BOOL_FLAGS = ("log", "online", "calendar", "monthly", "trend")
SHOW_ROWS = 24


def _read(source: str, column: str | None = None) -> pd.Series:
    """Load a series from a CSV file or a FRED id."""
    path = Path(source)
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if column is not None:
            s = df[column]
        else:
            numeric = df.select_dtypes("number")
            if numeric.empty:
                raise ValueError(f"{source}: no numeric column")
            s = numeric.iloc[:, 0]
        return s.astype(float).rename(s.name or path.stem)
    if path.suffix or any(sep in source for sep in "/\\"):
        raise FileNotFoundError(f"no such file: {source}")
    sid = source.upper()
    return load(sid, online=sid not in SERIES)


def _emit(obj: pd.Series | pd.DataFrame, out: str | None) -> str:
    """Write ``obj`` to ``out`` as CSV, or render it as text."""
    if out is not None:
        path = Path(out)
        obj.to_csv(path)
        return f"wrote {path} ({len(obj)} rows)"
    text = obj.to_frame() if isinstance(obj, pd.Series) else obj
    if len(text) <= SHOW_ROWS:
        return text.to_string()
    shown = text.tail(SHOW_ROWS).to_string()
    return (
        f"{shown}\n(last {SHOW_ROWS} of {len(text)} rows; use --out FILE.csv for all)"
    )


def _log100(s: pd.Series, log: bool) -> pd.Series:
    if log:
        if (s <= 0).any():
            raise ValueError("--log needs a strictly positive series")
        return 100 * np.log(s)
    return s


class _Cli:
    """Manipulate economic time series from the shell.

    A SOURCE is a CSV file (first column dates; choose a value column with
    --column) or a FRED series id (bundled snapshot, else fetched online).
    Results print as text, or go to a CSV file with --out FILE.csv.
    Run `econts COMMAND --help` for a command's flags.
    """

    # Methods return text, not DataFrames: Fire would print a returned
    # DataFrame's attributes instead of its data.

    def series(self) -> str:
        """List the FRED series bundled with the toolkit.

        Returns:
            The series ids and descriptions.

        """
        width = max(map(len, SERIES))
        return "\n".join(f"{k:<{width}}  {v}" for k, v in SERIES.items())

    def load(self, series_id: str, online: bool = False, out: str | None = None) -> str:
        """Print or save a FRED series.

        Args:
            series_id: FRED id, e.g. GDPC1.
            online: fetch the latest vintage instead of the bundled snapshot.
            out: CSV file to write.

        Returns:
            The series as text, or a confirmation.

        """
        sid = str(series_id).upper()
        return _emit(load(sid, online=online or sid not in SERIES), out)

    def growth(
        self,
        source: str,
        kind: str = "yoy",
        column: str | None = None,
        out: str | None = None,
    ) -> str:
        """Growth rate in percent.

        Args:
            source: CSV file or FRED id.
            kind: pop, yoy, annualized, log or log_annualized.
            column: CSV column to use.
            out: CSV file to write.

        Returns:
            The growth rate as text, or a confirmation.

        """
        s = _read(source, column)
        return _emit(tf.growth(s, kind).rename(f"{s.name}_{kind}"), out)  # type: ignore[arg-type]

    def aggregate(
        self,
        source: str,
        freq: str = "QS",
        how: str = "mean",
        min_obs: int = 1,
        column: str | None = None,
        out: str | None = None,
    ) -> str:
        """Aggregate to a lower frequency.

        Args:
            source: CSV file or FRED id.
            freq: target frequency alias (MS, QS, YS, YE-SEP, ...).
            how: sum (flows), mean (rates, prices, SAAR flows), last (stocks), ...
            min_obs: minimum observations per period, else NaN.
            column: CSV column to use.
            out: CSV file to write.

        Returns:
            The aggregated series as text, or a confirmation.

        """
        s = _read(source, column)
        return _emit(agg.aggregate(s, freq, how, min_obs), out)

    def filter(
        self,
        source: str,
        method: str = "hp",
        log: bool = False,
        lamb: float = 1600.0,
        h: int = 8,
        p: int = 4,
        order: int = 1,
        break_at: str | None = None,
        low: int = 6,
        high: int = 32,
        column: str | None = None,
        out: str | None = None,
    ) -> str:
        """Split a series into cycle and trend.

        Args:
            source: CSV file or FRED id.
            method: hp, hp1 (one-sided HP), hamilton, bk, cf, bn, poly or haar.
            log: work with 100 x log(series), so the cycle is in percent.
            lamb: HP smoothing (1600 quarterly, 129600 monthly, 6.25 annual).
            h: Hamilton horizon (8 quarterly, 24 monthly).
            p: Hamilton lags (4 quarterly, 12 monthly); BN AR order.
            order: polynomial order for poly.
            break_at: break date for poly, e.g. 1973-10-01.
            low: band-pass lower period (bk, cf).
            high: band-pass upper period (bk, cf).
            column: CSV column to use.
            out: CSV file to write.

        Returns:
            The series, cycle and trend as text, or a confirmation.

        """
        y = _log100(_read(source, column), log)
        method = method.lower()
        if method == "hp":
            cycle, trend = filters.hp(y, lamb)
        elif method in ("hp1", "hp-one-sided", "one-sided"):
            cycle, trend = filters.hp_one_sided(y, lamb)
        elif method == "hamilton":
            cycle, trend = filters.hamilton_filter(y, h, p)
        elif method == "bk":
            cycle = filters.bk(y, low, high)
            trend = y - cycle
        elif method == "cf":
            cycle, trend = filters.cf(y, low, high)
        elif method == "bn":
            cycle, trend = filters.beveridge_nelson(y, p)
        elif method == "poly":
            cycle, trend = filters.poly_detrend(y, order, break_at)
        elif method == "haar":
            cycle, trend = filters.haar_cycle(y)
        else:
            raise ValueError(f"unknown method {method!r}")
        return _emit(pd.DataFrame({"series": y, "cycle": cycle, "trend": trend}), out)

    def disagg(
        self,
        source: str,
        indicators: str | Sequence[str],
        method: str = "denton",
        how: str = "mean",
        m: int = 3,
        column: str | None = None,
        out: str | None = None,
    ) -> str:
        """Disaggregate a low-frequency series using high-frequency indicators.

        The indicators must start in the first sub-period of the source's first
        period (dates mark period starts, as in FRED); they may run past its
        end, and those sub-periods are extrapolated.

        Args:
            source: low-frequency CSV file or FRED id (e.g. quarterly GDP).
            indicators: comma-separated CSV files or FRED ids (e.g. INDPRO,PAYEMS);
                denton uses the first.
            method: denton, chow-lin or fernandez.
            how: mean (rates, SAAR flows) or sum (flows) aggregation.
            m: sub-periods per period (3 for quarters to months, 4 or 12 for years).
            column: CSV column of the source.
            out: CSV file to write.

        Returns:
            The high-frequency series as text, or a confirmation, plus rho for chow-lin.

        """
        Y = _read(source, column)
        names = (
            indicators.split(",") if isinstance(indicators, str) else list(indicators)
        )
        X = pd.concat([_read(str(n).strip()) for n in names], axis=1)
        X = X.loc[Y.index[0] :].dropna()
        Y = Y.dropna().iloc[: len(X) // m]
        method = method.lower()
        if method == "denton":
            result, note = disagg.denton(Y, X.iloc[:, 0], m, how), ""  # type: ignore[arg-type]
        elif method in ("chow-lin", "chowlin", "fernandez"):
            fn = disagg.fernandez if method == "fernandez" else disagg.chow_lin
            result, info = fn(Y, X, m, how)  # type: ignore[arg-type]
            note = f"\nrho = {info['rho']:.3f}" if method != "fernandez" else ""
        else:
            raise ValueError(f"unknown method {method!r}")
        return _emit(result, out) + note

    def seasonal(
        self,
        source: str,
        period: int | None = None,
        calendar: bool = False,
        additive: bool = False,
        column: str | None = None,
        out: str | None = None,
    ) -> str:
        """Seasonally adjust with robust STL, optionally with calendar effects.

        Args:
            source: monthly or quarterly CSV file or FRED id (not adjusted).
            period: seasonal period; inferred (12 monthly, 4 quarterly) when omitted.
            calendar: also remove trading-day and Easter effects (monthly only).
            additive: adjust levels instead of logs.
            column: CSV column to use.
            out: CSV file to write.

        Returns:
            The adjusted series and seasonal factors as text, or a confirmation.

        """
        s = _read(source, column)
        period = period or tf.periods_per_year(s.index)
        if period not in (4, 12):
            raise ValueError("seasonal adjustment needs monthly or quarterly data")
        s = s.asfreq("MS" if period == 12 else "QS")
        if calendar and period != 12:
            raise ValueError("--calendar needs monthly data")
        res = seasonal.stl_adjust(
            s,
            period=period,
            seasonal=13 if period == 12 else 7,
            log=not additive,
            calendar=calendar,
        )
        return _emit(res, out)

    def bbq(
        self,
        source: str,
        log: bool = False,
        monthly: bool = False,
        column: str | None = None,
    ) -> str:
        """Date business-cycle peaks and troughs (Harding-Pagan BBQ).

        Args:
            source: CSV file or FRED id, in levels (e.g. GDPC1).
            log: date 100 x log(series), so amplitudes are in percent.
            monthly: use the monthly rules (window 5, phase 6, cycle 15).
            column: CSV column to use.

        Returns:
            The turning points and the average duration and amplitude of each phase.

        """
        y = _log100(_read(source, column), log).dropna()
        rules = {"window": 5, "phase": 6, "cycle": 15} if monthly else {}
        tp = cycles.bbq(y, **rules)
        points = tp[tp != 0].map({1: "peak", -1: "trough"})
        ph = cycles.phases(y, tp)
        summary = ph.groupby("phase")[["duration", "amplitude"]].mean().round(2)
        return f"{points.to_frame('turning point').to_string()}\n\naverage by phase\n{summary.to_string()}"

    def unitroot(
        self, source: str, trend: bool = False, column: str | None = None
    ) -> str:
        """ADF and KPSS tests on the level and the first difference.

        Args:
            source: CSV file or FRED id.
            trend: include a trend in the test on the level (for trending series).
            column: CSV column to use.

        Returns:
            The p-values and a verdict for the level and the difference.

        """
        s = _read(source, column).dropna()
        table = diagnostics.unit_root_table(
            {
                "level": (s, "ct" if trend else "c"),
                "first difference": (s.diff(), "c"),
            }
        )
        return table.round(3).to_string()

    def outliers(
        self,
        source: str,
        log: bool = False,
        window: int = 21,
        threshold: float = 4.0,
        column: str | None = None,
    ) -> str:
        """List observations far from the rolling median (robust z-score).

        Args:
            source: CSV file or FRED id.
            log: test log(series), for series that grow.
            window: rolling window (centered).
            threshold: robust standard deviations to flag.
            column: CSV column to use.

        Returns:
            The flagged observations with their robust z-scores.

        """
        s = _read(source, column).dropna()
        x = np.log(s) if log else s
        z = diagnostics.robust_zscore(x, window)
        flagged = pd.DataFrame({"value": s, "robust z": z.round(1)})[
            z.abs() > threshold
        ]
        return flagged.to_string() if len(flagged) else "no outliers"


def _bind_bool_flags(argv: list[str]) -> list[str]:
    """Rewrite bare boolean flags as ``--flag=True`` (arguments after ``--`` untouched)."""
    out, passthrough = [], False
    for arg in argv:
        if arg == "--":
            passthrough = True
        name = (
            arg[2:].replace("-", "_") if arg.startswith("--") and "=" not in arg else ""
        )
        out.append(f"--{name}=True" if not passthrough and name in BOOL_FLAGS else arg)
    return out


def main(argv: list[str] | None = None) -> None:
    """Run the Fire-generated CLI; expected errors print one line and exit 1."""
    import fire

    args = _bind_bool_flags(sys.argv[1:] if argv is None else list(argv))
    try:
        fire.Fire(_Cli(), command=args, name="econts")
    except (
        ValueError,
        KeyError,
        OSError,
    ) as err:  # OSError covers missing files and HTTP errors
        print(f"econts: error: {err}", file=sys.stderr)
        raise SystemExit(1) from None
