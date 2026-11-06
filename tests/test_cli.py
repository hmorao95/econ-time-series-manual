"""Tests for the econts command-line interface (offline: bundled data only)."""

import pandas as pd
import pytest

from econts import load
from econts.cli import _bind_bool_flags, _Cli, main


@pytest.fixture
def cli():
    return _Cli()


def test_series_lists_bundled_ids(cli):
    text = cli.series()
    assert "GDPC1" in text and "USRECQ" in text


def test_load_writes_csv_and_reads_it_back(cli, tmp_path):
    out = tmp_path / "gdp.csv"
    assert "wrote" in cli.load("gdpc1", out=str(out))
    text = cli.growth(str(out), kind="yoy")
    assert "use --out" in text  # long output is truncated


def test_growth_infers_frequency_from_csv(cli, tmp_path):
    idx = pd.date_range("2020-01-01", periods=30, freq="MS")
    path = tmp_path / "s.csv"
    pd.DataFrame({"x": range(100, 130)}, index=idx).to_csv(path)
    out = tmp_path / "g.csv"
    cli.growth(str(path), kind="yoy", out=str(out))
    g = pd.read_csv(out, index_col=0)
    assert g.iloc[12, 0] == pytest.approx(100 * (112 / 100 - 1))


@pytest.mark.parametrize(
    "method", ["hp", "hp1", "hamilton", "bk", "cf", "bn", "poly", "haar"]
)
def test_filter_methods(cli, tmp_path, method):
    out = tmp_path / "f.csv"
    cli.filter("GDPC1", method=method, log=True, out=str(out))
    df = pd.read_csv(out, index_col=0)
    assert list(df.columns) == ["series", "cycle", "trend"]
    assert df["cycle"].notna().sum() > 200


def test_disagg_methods_add_up(cli, tmp_path):
    for method in ("denton", "chow-lin", "fernandez"):
        out = tmp_path / f"{method}.csv"
        text = cli.disagg("GDPC1", "INDPRO,PAYEMS", method=method, out=str(out))
        assert ("rho" in text) == (method == "chow-lin")
        x = pd.read_csv(out, index_col=0, parse_dates=True).iloc[:, 0]
        q = x.resample("QS").mean()
        Y = load("GDPC1").reindex(q.index).dropna()
        assert (q.reindex(Y.index) / Y - 1).abs().max() < 1e-8


def test_aggregate_seasonal_bbq_unitroot_outliers(cli):
    assert "use --out" in cli.aggregate("PAYEMS", freq="QS", how="mean")
    assert "sa" in cli.seasonal("RSXFSN", calendar=True)
    text = cli.bbq("GDPC1", log=True)
    assert "1990-07-01" in text and "average by phase" in text
    assert "verdict" in cli.unitroot("GDPC1", trend=True)
    assert "2020-03-28" in cli.outliers("ICSA", log=True)


def test_bad_inputs_raise(cli):
    with pytest.raises(ValueError):
        cli.filter("GDPC1", method="nope")
    with pytest.raises(ValueError):
        cli.seasonal("GDPC1", calendar=True)  # quarterly data
    with pytest.raises(FileNotFoundError):
        cli.growth("missing.csv")


def test_bind_bool_flags():
    assert _bind_bool_flags(["filter", "--log", "GDPC1"]) == [
        "filter",
        "--log=True",
        "GDPC1",
    ]
    assert _bind_bool_flags(["x", "--", "--log"]) == ["x", "--", "--log"]
    assert _bind_bool_flags(["--log=False"]) == ["--log=False"]


def test_main_runs_and_exits_1_on_expected_errors(capsys):
    main(["filter", "--log", "GDPC1", "--method", "hamilton"])
    assert "cycle" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["growth", "missing.csv"])
    assert exc.value.code == 1
    assert "no such file" in capsys.readouterr().err
