# figures/

Generated output. Every file here is produced by the analysis pipeline, so only this README
is tracked in git.

`notebooks/reproduce_all_figures.py` populates the directory:

- **`*.png`** are symlinks into `results/`, rebuilt from the notebook's figure manifest on
  every run. Each link is named after its target, so there is exactly one name per figure and
  a producer that changes its output path cannot leave a dangling link behind.
- **`*.tex`** tables are written here directly by
  `analysis/generate_training_table.py` and `analysis/extract_variant_db_stats.py`.

Which manuscript label corresponds to which file is tabulated in the figure index in
[`docs/REPRODUCING_ANALYSES.md`](../docs/REPRODUCING_ANALYSES.md).

`enrichment_scatter.png` is a presentation figure rather than a manuscript one, so it
carries no figure number. It appears only when `PRESENTATION_FIGURES = True`.

`ls -l` shows the targets; a broken link means its producer has not been run yet.
