# Agent notes

Two deliverables in one repo, modelled on FilippoMB/python-time-series-handbook:

- **The book**: a Jupyter Book (`_config.yml`, `_toc.yml`) of one notebook
  per chapter in `notebooks/NN/<topic>.ipynb`, front and back matter in
  `notebooks/00/` (intro, toolkit, cheat sheet, knowledge test, resources).
  Each chapter: setup cell (installs `econts[book]` on Colab) → sections →
  Summary (rules of thumb) → Exercises. Published to GitHub Pages by
  `.github/workflows/deploy-book.yml` (gh-pages branch).
- **The toolkit**: the `econts` package in `src/econts/`, one module per
  topic, typed, with docstrings and tests in `tests/test_econts.py`.
  Notebooks keep their recipes inline for teaching, followed by an
  "In your projects" note naming the toolkit function. When a recipe changes
  in a notebook, change the toolkit function and its test too.

- **The CLI**: `econts.cli` (python-fire `_Cli`, methods return text,
  `main()` exits 1 on expected errors), entry point `econts`, tested in
  `tests/test_cli.py` with bundled data only. New toolkit functions that make
  sense from a shell get a subcommand.

Conventions:

- Notebooks are stored **with outputs**. After editing one, re-run it with
  `uv run python scripts/execute_notebooks.py notebooks/NN/<file>.ipynb`.
  `tests/test_notebooks.py` re-runs every notebook with warnings as errors.
  Tag a cell `skip-execution` when it needs an external program (X-13).
- Numbers quoted in the prose come from the bundled FRED snapshot in
  `src/econts/data/`. If you change code or refresh data, re-run the notebook
  and update the quoted numbers.
- Keep chapters short: rules, recipe, pitfalls. New methods must earn their
  place.
- `housestyle` (public, pinned by commit) is an optional extra (`book`) and a
  dev dependency; the toolkit itself must not import it.
- Supported: Python >= 3.10, pandas 2.2+ and 3.x, statsmodels >= 0.15. Write
  code that behaves the same on pandas 2 and 3 (e.g. `pct_change(fill_method=None)`).
- Workflow: branch, PR, squash-merge when CI is green.
