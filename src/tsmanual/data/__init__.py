"""Bundled FRED snapshot used by every chapter of the manual."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

SERIES: dict[str, str] = {
    "GDPC1": "Real GDP, quarterly, SAAR, chained 2017 dollars (BEA)",
    "GDP": "Nominal GDP, quarterly, SAAR, billions of dollars (BEA)",
    "GDPDEF": "GDP implicit price deflator, quarterly, SA, 2017=100 (BEA)",
    "INDPRO": "Industrial production index, monthly, SA, 2017=100 (Fed)",
    "RSXFSN": "Retail sales, monthly, not seasonally adjusted, $ millions (Census)",
    "RSXFS": "Retail sales, monthly, seasonally adjusted, $ millions (Census)",
    "CPIAUCSL": "CPI all items, monthly, SA, 1982-84=100 (BLS)",
    "UNRATENSA": "Unemployment rate, monthly, NSA, percent (BLS)",
    "UNRATE": "Unemployment rate, monthly, SA, percent (BLS)",
    "PAYEMS": "Nonfarm payroll employment, monthly, SA, thousands (BLS)",
    "ICSA": "Initial jobless claims, weekly (ending Saturday), SA (DOL)",
    "DCOILBRENTEU": "Brent crude oil spot price, daily, $/barrel (EIA)",
    "USRECQ": "NBER recession indicator, quarterly, 1 = recession (FRED)",
    "CLVMNACSCAB1GQPT": "Portugal real GDP, quarterly, SA, 2010 EUR mn (Eurostat)",
    "LRHUTTTTPTQ156S": "Portugal unemployment rate, quarterly, SA, percent (OECD)",
    "IRLTLT01PTM156N": "Portugal 10-year government bond yield, monthly, % (OECD)",
}


def load(series_id: str, online: bool = False) -> pd.Series:
    """Return a FRED series as a float Series with a DatetimeIndex.

    Args:
        series_id: FRED identifier, e.g. ``"GDPC1"``.
        online: fetch the latest vintage from FRED instead of the bundled
            snapshot. Any FRED series works online, bundled or not.

    Returns:
        The series, named ``series_id``. FRED dates mark the start of each
        period (``1990-04-01`` is 1990Q2 for quarterly data). Missing
        observations are NaN.

    """
    source = FRED_CSV.format(series_id) if online else DATA_DIR / f"{series_id}.csv"
    df = pd.read_csv(source, index_col=0, parse_dates=True, na_values=".")
    s = df.iloc[:, 0].astype(float).rename(series_id)
    s.index.name = "date"
    return s
