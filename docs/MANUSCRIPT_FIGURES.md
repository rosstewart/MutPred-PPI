# Manuscript figures and tables

Every figure and table in `main_091026.tex` and `supplement_091026.tex`, the
command that produces it, and what it depends on. This is the checklist for a
full regeneration: if an entry here has no output, the paper cannot be rebuilt.

Figure files are symlinks from `figures/` into the results tree, so a figure is
"present" only when its target exists. `ls -l figures/` shows the targets.

Two labels are **not** generated: `fig:pipeline` and `fig:architecture` are
hand-drawn schematics, and `fig:cdc42_example` is a ChimeraX rendering. They are
committed artwork, not pipeline outputs.

## Main text

| Label | File | Produced by | Depends on |
|---|---|---|---|
| `fig:pipeline` | `MutPred-PPI_pipeline.png` | *hand-drawn schematic* | — |
| `fig:architecture` | `MutPred-PPI_architecture.png` | *hand-drawn schematic* | — |
| `tab:datasets` | `training_data_table.tex` | `analysis/generate_training_table.py` | mapped source CSVs (`af3_failed` column) |
| `fig:sahni_fragoza_cv` | `roc_sahni_fragoza_with_variance.png` | `analysis/run_roc_comparison.py` → `roc_plots.py` | GCV, `sahni_fragoza`, all methods |
| `fig:blind_test_roc` | `varchamp_cava_combined_roc_plots_roc_comparison_to_swing.png` | `analysis/blind_test_figures.py` | VarChAMP blind test, all methods |
| `fig:enrichment` | `enrichment_bootstrap_analysis_sufficient_partners.png` | `analysis/variant_db_charts.py --edgotype-bootstrap` | all 6 variant DBs classified |
| `fig:cdc42_example` | `CDC42_WASP_Y64.png` | *ChimeraX rendering* | — |

`fig:brca1_example` (`brca1_bard1_C61.png`) is commented out in the current
manuscript and is not required.

## Supplement

| Label | File | Produced by | Depends on |
|---|---|---|---|
| `tab:variant_dbs_table` | Table S1 | `analysis/extract_variant_db_stats.py` | all 6 variant DBs |
| `fig:roc_sahni_fragoza_biclass` | `roc_sahni_fragoza_biclass_with_variance.png` | `analysis/biclass_sf_gcv.py` | GCV, `sahni_fragoza` |
| `fig:sahni_cv` | `roc_sahni_with_variance.png` | `analysis/run_roc_comparison.py` | GCV, `sahni_only`, all methods |
| `fig:varchamp_training_set_comparison` | `varchamp_cava_combined_roc_plots_roc_comparison_training_sets.png` | `analysis/blind_test_figures.py` | blind test, both MutPred-PPI training sets |
| `fig:ablation` | `ablation_bar_sahni_fragoza.png` | `analysis/run_roc_ablation.py` → `roc_plots.plot_ablation_bars` | GCV ablations, `sahni_fragoza` |
| `fig:combined_robustness_by_class` | `combined_robustness_by_class.png` | `analysis/combined_robustness_figure.py` | interface / pLDDT / protein-class stratifications |
| `fig:sahni_varchamp_cava_cv` | `roc_sahni_fragoza_varchamp_full_pooled_with_variance.png` | `analysis/run_roc_comparison.py` | GCV, `sahni_fragoza_varchamp_all`, all methods |
| `fig:enrichment_k3` | `enrichment_bootstrap_analysis_sufficient_partners_k3.png` | `analysis/variant_db_charts.py --controlled-bootstrap --k3-only` | all 6 variant DBs classified |
| `fig:threshold_sensitivity` | `threshold_sensitivity.png` | `analysis/threshold_sensitivity.py` | all 6 variant DBs classified |
| `fig:protein_class_enrichment` | `pathogenic_by_class.png` | `analysis/protein_class_enrichment.py` | ClinVar + gnomAD predictions, GO annotations |
| `fig:stability_mechanism` | `scatter_per_variant_kde.png` | `analysis/stability_interaction_scatter.py` | PPI + stability predictions for 6 groups |

## How manuscript filenames map to pipeline outputs

The `.tex` files reference names that the pipeline no longer writes. This is not
a defect: `figures/` holds a symlink per manuscript name pointing at the current
output, so LaTeX resolves and the scripts keep clean names. `ls -l figures/`
shows the mapping.

| Manuscript name (symlink) | Pipeline output (target) |
|---|---|
| `varchamp_cava_combined_roc_plots_roc_comparison_to_swing.png` | `roc_varchamp_blind_test.png` |
| `varchamp_cava_combined_roc_plots_roc_comparison_training_sets.png` | `roc_varchamp_blind_test_training_comparison.png` |
| `roc_sahni_fragoza_varchamp_full_pooled_with_variance.png` | `roc_sahni_fragoza_varchamp_all_with_variance.png` |
| `enrichment_bootstrap_analysis_sufficient_partners.png` | `enrichment_bootstrap_sufficient_partners.png` |
| `enrichment_bootstrap_analysis_sufficient_partners_k3.png` | `enrichment_bootstrap_sufficient_partners_k3.png` |
| `ablation_bar_sahni_fragoza.png` | `ablation_bar_sahni_fragoza_with_variance.png` |

Worth tidying before release: three of these aliases carry dataset tokens the
090826 rebaseline retired (`cava`, `varchamp_full_pooled`), which `legacy_guard`
rejects everywhere else. Renaming them in the `.tex` to the pipeline's own names
would let the aliases be deleted; nothing in the code needs to change.

## A dependency that is easy to miss

Every variant-database figure depends on **ProtVar's precomputed AlphaFold3
interfaces**, which are a separate download and are not part of the Zenodo
bundle. ClinVar, COSMIC, gnomAD and HGMD take roughly 40-50% of their contact
graphs from it; neurodev and asd take none. A reproduction that skips it still
runs, and silently produces those four databases at about half their true
coverage. See [`SETUP.md`](SETUP.md) for the download and the `external/protvar_pdb`
symlink.

## Regenerating

The suites that feed the figures, in order:

```bash
conda run -n ppi python src/run_benchmarks.py --suite gcv --gpus 0,1,2,3
conda run -n ppi python src/run_benchmarks.py --suite blind-test --gpus 0,1
conda run -n ppi python src/run_benchmarks.py --suite variant-db --gpus 0,1,2,3
conda run -n ppi python src/analysis/classify_variant_dbs.py \
    --output-dir results/variant_dbs_all_data
```

then the figure scripts in the table above. `notebooks/reproduce_all_figures.py`
runs the whole chain end to end and is the supported path;
[`REPRODUCING_ANALYSES.md`](REPRODUCING_ANALYSES.md) documents each step
individually.
