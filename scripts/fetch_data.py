"""Refresh the bundled FRED snapshot in src/tsmanual/data/.

Run with ``uv run python scripts/fetch_data.py``. FRED revises data, so
numbers quoted in the chapters can move slightly after a refresh.
"""

from tsmanual.data import DATA_DIR, SERIES, load

for sid in SERIES:
    s = load(sid, online=True)
    s.to_csv(DATA_DIR / f"{sid}.csv")
    print(f"{sid:14s} {s.index[0].date()} to {s.index[-1].date()}  n={len(s)}")
