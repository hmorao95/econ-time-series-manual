# Agent notes

A manual, not a library. The deliverable is `docs/*.md`; `src/tsmanual` only
loads the bundled data and runs the chapters.

- Every ```python block in `docs/NN-*.md` runs in CI (`uv run pytest`) with
  warnings as errors, sharing one namespace per chapter. Keep blocks runnable
  in order; use ```text for output or pseudo-code.
- Numbers quoted in the prose come from the bundled FRED snapshot in
  `src/tsmanual/data/`. If you change code or refresh data, rerun the chapter
  (`uv run python -m tsmanual.runner docs/NN-*.md`) and update the quoted numbers.
- Figures: a `# figure: name` line saves `docs/img/name.png`; regenerate with
  `uv run python -m tsmanual.runner --figures`.
- Keep chapters short: quick rules, recipe, pitfalls. New methods must earn
  their place.
- Supported: Python >= 3.10, pandas 2.2+ and 3.x, statsmodels >= 0.15. Write
  code that behaves the same on pandas 2 and 3 (e.g. `pct_change(fill_method=None)`).
- Workflow: branch, PR, squash-merge when CI is green.
