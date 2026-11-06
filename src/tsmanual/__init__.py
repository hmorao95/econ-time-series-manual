"""Helpers for the Economic Time Series in Python manual.

Exports:
    load: read a bundled FRED series (or fetch it fresh) as a pandas Series.
    SERIES: the series bundled with the manual, with short descriptions.
    DATA_DIR: folder of the bundled files (also holds pt_recessions.txt).
"""

from tsmanual.data import DATA_DIR, SERIES, load

__all__ = ["DATA_DIR", "SERIES", "load"]
