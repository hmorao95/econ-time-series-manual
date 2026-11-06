# Economic Time Series in Python

A practical handbook for economists who manipulate time series in Python:
the steps between downloading a series and estimating a model. Every chapter
is a notebook that runs as written on real data (a bundled snapshot of
FRED), and every recipe is also packaged in **`econts`**, a toolkit you can
install and import in your own projects.

```{image} media/cycles.png
:alt: Business-cycle components of US GDP from several filters
:width: 700px
:align: center
```

## 📖 Chapters

```{tableofcontents}
```

Each chapter ends with a **Summary** of its rules of thumb and a set of
**Exercises** on the bundled data. The [knowledge test](knowledge_test.md)
checks the main ideas, and the [cheat sheet](cheatsheet.md) gives one-line
answers to the most common tasks.

## 🚀 Getting started

Every chapter opens in Google Colab or nbviewer from the
[repository README](https://github.com/hmorao95/econ-time-series-manual);
the first cell installs the toolkit when it runs on Colab.

To run the notebooks locally:

```bash
git clone https://github.com/hmorao95/econ-time-series-manual
cd econ-time-series-manual
uv sync                    # or: pip install -e ".[book]" jupyterlab
uv run --with jupyterlab jupyter lab notebooks
```

## 🧰 The `econts` toolkit

The functions written out in the chapters are collected in a package, so a
project can use them without copying code:

```bash
pip install git+https://github.com/hmorao95/econ-time-series-manual
```

```python
import numpy as np
from econts import load
from econts.filters import hamilton_filter
from econts.disagg import denton

y = 100 * np.log(load("GDPC1"))
cycle, trend = hamilton_filter(y)
```

The [toolkit page](toolkit.md) lists every function and the chapter that
explains it.

## 📊 Data

`econts.load` reads a snapshot of FRED series bundled with the package, so
the notebooks run offline and reproduce the numbers quoted in the text.
`load(series_id, online=True)` fetches the latest vintage of any FRED
series. Bundled: US real and nominal GDP, the GDP deflator, industrial
production, retail sales (adjusted and not), CPI, unemployment (adjusted and
not), payrolls, weekly jobless claims, daily Brent, the NBER recession
indicator, Portuguese GDP, unemployment and 10-year yield, and a quarterly
Portuguese recession chronology.

## Conventions

* Series that grow are analysed in **logs × 100**, so differences are
  (approximately) percent changes.
* Cells within a chapter run in order and share variables.
* Every notebook is executed in CI with warnings treated as errors, so the
  code works with the pinned library versions.

## Credits

The structure of this book follows Filippo Maria Bianchi's
[Time series analysis with Python](https://github.com/FilippoMB/python-time-series-handbook).
Several recipes port functions of Filippo Ferroni's
[Empirical Macro Toolbox](https://naffe15.github.io/EMT/) (MATLAB), and
chapter 11 draws its figures with
[housestyle](https://github.com/hmorao95/housestyle). Data: FRED, Federal
Reserve Bank of St. Louis.
