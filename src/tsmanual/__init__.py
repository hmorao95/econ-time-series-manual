"""Helpers for the Economic Time Series in Python manual.

Exports:
    load: read a bundled FRED series (or fetch it fresh) as a pandas Series.
    SERIES: the series bundled with the manual, with short descriptions.
"""

from tsmanual.data import SERIES, load

__all__ = ["SERIES", "load"]
