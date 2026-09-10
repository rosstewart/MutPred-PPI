# Figure Inventory and Dependency Map

**Generated 2026-09-09, before the pending AF3 re-run.** This is a working document: every
row is a thing you must rebuild or knowingly abandon. It answers three questions —
*what artifacts exist*, *what produces each*, and *which inputs are already invalid*.

> **Superseded in several places as of 2026-09-10 — read this note before trusting a path
> below.** `results_revisions/` no longer exists; its subdirectories moved under `results/`
> (`macro_aucs/` -> `results/gcv/`, `robustness_analyses/` -> `results/robustness/`,
> `protein_class_enrichment/` -> `results/protein_class/`, `variant_dbs_sfvfp/` ->
> `results/variant_dbs_all_data/`). DDMutPPI is no longer evaluated at all — it is not a
> "still wired in but excluded" case any more; every reference to it below is historical.
> `src/evaluation/run_vcfp_blind_test.py` is retired (renamed and rewritten as
> `run_varchamp_blind_test.py`, train=`sahni_fragoza_mapped090826` /
> test=`varchamp_all_mapped090826`, no more `datasets/sfvcfp_rows.csv.gz`). SAAMBE-3D/MutPPI/
> MutPPI+ have an in-repo producer now (**O5** is resolved: `src/evaluation/mutppi_cv.py`).
> `src/utils/legacy_guard.py` now raises on every retired path/dataset-name pattern this
> document names, instead of the six pre-flight checks in §8 being the only signal.
> **`notebooks/reproduce_all_figures.py`** runs the entire regeneration order in §7 as one
> cached, resumable script/notebook — read that first for a live run, and treat this document
> as the historical record of what was broken and why.

Companion documents: `docs/REPRODUCING_ANALYSES.md` (the commands), `REPRODUCIBILITY.md`
(phase-by-phase runbook + known gaps), `archive/pre_090826/README.md` (what was retired).

Path shorthand used throughout:

| symbol | resolves to |
|---|---|
| `P` | `/data/ross/ppi_lossgain/interaction_loss/publication` |
| `B` | `/data/ross/ppi_lossgain/interaction_loss` (`MUTPRED_DATA_ROOT`) |
| `CV` | `P/datasets/cv_reference` |
| `GCV` | `P/results_revisions/macro_aucs` |
| `VCFP` | `P/results/varchamp_seqcnf_newvar_eval` |

---

## 0. Headline numbers

| | count |
|---|---|
| Manuscript figures (`main.tex` + `supplement.tex`) | **14** (5 main, 9 supplement) |
| Manuscript tables | **3** |
| Generated figure/table artifacts on disk (`figures/`, `results_revisions/`, `results/`) | **69** |
| Figure-producing scripts in `src/` | **13** |
| Manuscript figures whose producer is **missing or archived** | **6** |
| Figure scripts that **DIE** today (all verified by execution) | **6** |
| Figure scripts that **run but silently use stale inputs** | **10** |
| Distinct input files classified **pre-090826** or **position-keyed** | **~40** |
| Orphans (§4) | **13** |

**Everything in `GCV/` and every `.npy` in `VCFP/` is position-keyed against a retired
row ordering.** `sahni_fragoza` went 5,894 → 6,219 rows; SFVCFP blind-test classes went
C1=1,867 → C1=9,236. Nothing keyed by position survives that.

---

## 1. Input classification legend

| class | meaning |
|---|---|
| **canonical** | derived from `P/datasets/{cv_reference,variant_dbs}/*rows.csv.gz`, `P/datasets/sfvcfp_rows.csv.gz`, `P/datasets/training_eval/*`, a contact-graph `.h5` store, or `P/datasets/af3_structures*_canonical/` |
| **position-keyed** | a per-row array/pickle aligned to a row ORDERING. Everything under `GCV/`, every `VCFP/*.npy`, every `*_predictions.tsv` prediction cache. **These are the dangerous ones.** |
| **pre-090826** | built under the old mapping (mtime < 2026-09-08 *and* keyed to old row set) |
| **external** | third-party, not generated here (BioGRID, UniProt, ClinVar, COSMIC, HGMD, gnomAD, ClinGen, SKEMPI, `datasets/mega_splits.pkl`) |
| **unknown/missing** | no producer found, or the file does not exist |

---

## 2. Manuscript figures — producer and status

`figures/*.png` are almost all symlinks. The **Real path** column is what is actually
written; the **Producer** column is what you must run.

### 2.1 `main.tex`

| # | `\label` | `figures/` entry | Real path | Producer | Status |
|---|---|---|---|---|---|
| 1 | `fig:pipeline` | `MutPred-PPI_pipeline.png` | — | **NONE** | **ORPHAN — file does not exist, no script, no editable source** |
| 2 | `fig:architecture` | `MutPred-PPI_architecture.png` | — | **NONE** | **ORPHAN — file does not exist, no script, no editable source** |
| 3 | `fig:sahni_fragoza_cv` | `roc_sahni_fragoza_with_variance.png` | `GCV/roc_plots_with_variance/roc_sahni_fragoza_with_variance.png` | `src/analysis/run_roc_comparison.py` → `roc_plots.py` | **DIES** (verified) |
| 4 | `fig:blind_test_roc` | `varchamp_cava_combined_roc_plots_roc_comparison_to_swing.png` | `VCFP/roc_plots/roc_varchamp_full_pooled.png` | `src/analysis/varchamp_blind_test.py` | **RUNS, SILENTLY STALE** |
| 5 | `fig:enrichment` | `enrichment_bootstrap_analysis_sufficient_partners.png` | `results_revisions/variant_dbs_sfvfp/enrichment_bootstrap_sufficient_partners.png` | `src/analysis/variant_db_charts.py` | **RUNS, SILENTLY STALE** |
| (—) | `fig:brca1_example` | `brca1_bard1_C61.png` | — | **NONE** | commented out in `main.tex:266`; PNG never existed |

### 2.2 `supplement.tex`

| # | `\label` | `figures/` entry | Real path | Producer | Status |
|---|---|---|---|---|---|
| S1 | `fig:sahni_cv` | `roc_sahni_with_variance.png` | `GCV/roc_plots_with_variance/roc_sahni_with_variance.png` | `run_roc_comparison.py` → `roc_plots.py` | **DIES** (module-level exception before this dataset is plotted) |
| S2 | `fig:varchamp_training_set_comparison` | `varchamp_cava_combined_roc_plots_roc_comparison_training_sets.png` | `VCFP/roc_plots/roc_varchamp_full_pooled_training_comparison.png` | `varchamp_blind_test.py` | **RUNS, SILENTLY STALE** |
| S3 | `fig:ablation` | `ablation_bar_sahni_fragoza.png` | `GCV/roc_plots_with_variance/ablation_bar_sahni_fragoza_with_variance.png` | `src/analysis/run_roc_ablation.py` → `roc_plots.py` | **DIES** (verified) |
| S4 | `fig:interface_performance` | `interface_auroc_by_class.png` | `results_revisions/robustness_analyses/interface_auroc_by_class.png` | `src/analysis/interface_analysis.py` | **DIES** — `IndexError` (verified) |
| S5 | `fig:plddt_performance` | `plddt_auroc_by_class.png` | `results_revisions/robustness_analyses/plddt_auroc_by_class.png` | `src/analysis/plddt_stratification.py` | **DIES** — `FileNotFoundError` (**O11**), then `IndexError` (verified) |
| S6 | `fig:single_multi_domain_performance` | `protein_class_auroc_by_class.png` | `results_revisions/robustness_analyses/protein_class_auroc_by_class.png` | `src/analysis/protein_class_stratification.py` | **DIES** — `IndexError` (verified) |
| S7 | `fig:sahni_varchamp_cava_cv` | `roc_sahni_fragoza_varchamp_full_pooled_with_variance.png` | `GCV/roc_plots_with_variance/...` | `run_roc_comparison.py` → `roc_plots.py` | **DIES** |
| S8 | `fig:enrichment_k3` | `enrichment_bootstrap_analysis_sufficient_partners_k3.png` | `results_revisions/variant_dbs_sfvfp/enrichment_bootstrap_sufficient_partners_k3.png` | `variant_db_charts.py --controlled-k 3` | **RUNS, SILENTLY STALE** |
| S9 | `fig:threshold_sensitivity` | `threshold_sensitivity.png` | `results_revisions/robustness_analyses/threshold_sensitivity.png` | `src/analysis/threshold_sensitivity.py` | **RUNS, SILENTLY STALE** |

### 2.3 Tables

All three are **pasted inline** into the `.tex` — there is no `\input{}`. Regenerating the
`.tex` fragment is not enough; someone must re-paste it.

| `\label` | source fragment | Producer | Status |
|---|---|---|---|
| `tab:datasets` (`main.tex:132`) | `figures/training_data_table.tex` | `src/analysis/generate_training_table.py` | **RUNS, SILENTLY STALE** (reads two pre-090826 label `.txt` files) |
| `tab:variant_dbs_table` (`supplement.tex:134`) | `figures/variant_db_stats_table.tex` | `src/analysis/extract_variant_db_stats.py` | **RUNS, SILENTLY STALE** |
| `tab:cosmic_onco_tsg_stat_test` (`supplement.tex:274`) | `figures/cosmic_onco_tsg_qn_vs_edgetic.tex` → `results_revisions/cosmic_stat_test/` | `src/analysis/cosmic_onco_tsg_stat_test.py` | **RUNS, SILENTLY STALE** |

---

## 3. Generated but NOT in the manuscript

These exist and are maintained; decide whether they are supplement candidates or dead
weight before the AF3 run.

| artifact | producer | notes |
|---|---|---|
| `figures/combined_robustness_by_class.png` → `results_revisions/robustness_analyses/` | `src/analysis/combined_robustness_figure.py` | 3-row A/B/C merge of S4+S5+S6; **DIES** (imports the three that die) |
| `figures/pathogenic_by_class.png` → `results_revisions/protein_class_enrichment/` | `src/analysis/protein_class_enrichment.py` | runs; also writes `pathogenic_by_class.tsv` |
| `figures/scatter_per_variant_kde.png` → `results_revisions/stability_interaction/` | `src/analysis/stability_interaction_scatter.py` | also `scatter_per_variant.png`, `per_variant_summary.tsv` |
| `figures/roc_sahni_fragoza_biclass_with_variance.png` → `results_revisions/biclass_gcv/` | `src/analysis/biclass_sf_gcv.py` | **DIES** — imports `roc_plots` |
| `figures/ablation_boxplot_sahni_fragoza.png` | `run_roc_ablation.py` (BOXPLOT variant) | **DIES** |
| `figures/roc_sahni_fragoza_varchamp1p_cava_with_variance.png` | `run_roc_comparison.py` | **DIES**; `varchamp1p_cava` is an obsolete dataset (see §5.4) |
| `results_revisions/stability_interaction/clustering_{kmeans,gmm}.png`, `clustering_summary.tsv` | `src/analysis/stability_interaction_clustering.py` | runs; reads `variant_dbs/` (SF model), not `variant_dbs_sfvfp/` |
| `results_revisions/stability_interaction/interaction_vs_stability_{scatter,quadrants}.png` | `src/analysis/stability_interaction_comparison.py` | runs; reads `variant_dbs/` (SF model) |
| `GCV/roc_plots_with_variance/spearman_box_sahni_fragoza.png` | `src/analysis/iptm_analysis.py` | runs; consumes `GCV/*detailed_results.pkl` (position-keyed) |
| `results_revisions/variant_dbs_sfvfp/enrichment_bootstrap_sufficient_partners_{clinvar_hgmd,k5,k7}.png` | `variant_db_charts.py` | runs, stale |
| `results_revisions/robustness_analyses/*.tsv` (3) | S4/S5/S6 scripts | **DIES** with their figures |
| `datasets/reconstruction_tables/*.csv` (~40) | `src/analysis/export_reconstruction_tables.py` | runs, but silently blanks id columns on count mismatch — see §5.3 |

---

## 4. THE ORPHAN LIST

> An orphan is an artifact whose producing script is missing/archived, **or** whose inputs
> cannot be regenerated. Ordered by severity.

### O1 — `CV/*_all_vt_ids_and_labels.txt` :: ~~NO PRODUCER ANYWHERE~~ **RESOLVED 2026-09-09, not an input**

**Severity: was highest, now none.** The files are archived to
`archive/pre_090826/cv_reference/` and nothing reads them. The finding below
stands — they really do have no producer — but the conclusion drawn from it was
wrong: they were never a *source* of labels, only a redundant copy of them.

Measured against the canonical table `sahni_fragoza_mapped090826`:

| | rows |
|---|---|
| `sahni_fragoza_all_vt_ids_and_labels.txt` | 5,894 |
| ↳ matched to a canonical row | 2,874 |
| ↳ **of which the label disagrees** | **0** |
| ↳ pair absent from the canonical table | 3,014 |
| ↳ pair present, mutation absent | 6 |

Where the two describe the same row they agree exactly; the 3,020 unmatched rows
are pre-090826 mappings the re-baselining dropped deliberately. The label is the
`perturbed` column of the canonical tables, so "no producer" costs nothing — and
re-emitting these files (the fix originally proposed here) would have been
actively wrong, because it would have re-minted the retired row set.

Readers migrated, rather than the file preserved:

| reader | now reads |
|---|---|
| `roc_plots.py` | `perturbed` via `gcv_common.load_data`, checked against `{prefix}rows.csv.gz` |
| `generate_training_table.py` | five canonical row tables |
| `biclass_sf_gcv.py` | `perturbed` for the biclass set; canonical `row_index` order for the positional baselines |
| `saambe3d_cv.py`, `ddmutppi_cv.py` | no longer reference them at all |

`grep -rn "and_labels" --include=*.py src/` now returns only docstrings.

**Historical detail, retained.** Ten files in `P/datasets/cv_reference/`. Verified by
`git rev-list --all | git grep` across the *entire* history: six scripts **read** them,
**zero** scripts **write** them. `export_cv_reference.py` — the canonical exporter — emits
`rows.csv.gz`, `clusters.pkl`, `fold_splits_{seed}.pkl`, `pair_test_classes_{seed}.npy`
and nothing else. `sahni_fragoza_all_vt_ids_and_labels.txt` is dated **2025-08-25**.

| file | lines | canonical equivalent | mismatch |
|---|---|---|---|
| `all_vt_ids_and_labels.txt` | 1,487 | `sahni_only_train_rows.csv.gz` = 1,595 | **−108** |
| `sahni_fragoza_all_vt_ids_and_labels.txt` | 5,894 | `sahni_fragoza_train_rows.csv.gz` = 6,219 | **−325** |
| `combined_sahni_varchamp1p_cava_seq_confirmed_all_vt_ids_and_labels.txt` | 4,423 | none exists | — |
| `combined_sahni_fragoza_varchamp1p_cava_seq_confirmed_...txt` | 8,827 | none exists | — |
| (6 more) | — | none | — |

Readers: `roc_plots.py`, `biclass_sf_gcv.py`, `generate_training_table.py`,
`saambe3d_cv.py`, `ddmutppi_cv.py`, (archived) `saambe3d_varchamp2026_gcv.py`.

**Fix applied:** the readers were migrated to the canonical row tables. The
alternative — adding a labels-file emitter to `export_cv_reference.py` — was
rejected: it would have re-created a second copy of the labels, which is the
duplication this refactor removes, and pinned it to the retired row ordering.

### O2 — `CV/*_all_vt_ids*.pkl` :: DELIBERATELY DEPRECATED, still read by 5 scripts (blocks S4, S5, S6, `tab:datasets`)

`export_cv_reference.py` states in its own docstring: *"`vt_id` pickles are NO LONGER
written."* But `sahni_fragoza_train_all_vt_ids_{0..29}.pkl` (**5,894** entries,
2026-09-05) are still read alongside `sahni_fragoza_train_fold_splits_{seed}.pkl`
(**6,219** rows, 2026-09-09). Verified failure:

```
protein_class_stratification.py:131  fold_vt_ids = [vt_ids_seed[idx] for idx in test_idx]
IndexError: list index out of range
```

Readers: `interface_analysis.py`, `plddt_stratification.py`,
`protein_class_stratification.py`, `combined_robustness_figure.py` (transitively),
`generate_training_table.py`, `export_reconstruction_tables.py`.

**Fix:** these scripts want `(interactor, partner, mutation)` per row — that is exactly
`CV/sahni_fragoza_train_rows.csv.gz`. Migrate the join key; do not resurrect the pickles.

### O3 — `figures/MutPred-PPI_pipeline.png` and `figures/MutPred-PPI_architecture.png` :: LOST ASSETS (Fig 1, Fig 2)

Referenced at `main.tex:100` and `main.tex:117`. **Neither file exists.** Exhaustive search
across `/data/ross/ppi_lossgain` and `/home/rcstewart` for those names, for any
`*pipeline*` / `*architecture*` raster, and for any editable source (`.pptx`, `.drawio`,
`.svg`, `.ai`, `.key`) found nothing. Not in git history, not in
`results_archive_052726.tar.gz`, not in `archive/mutpred_ppi_original_submission/`.
**They must be redrawn from scratch.** `main.tex` will not compile until they are.

### O4 — `VCFP/DDMutPPI (varchamp_full_pooled)_c{1,2,3}_*.npy` :: EMPTY, no producer (Fig 4, S2)

All nine arrays are 128 bytes = **zero rows**. `DDMutPPI` is listed in
`varchamp_blind_test.METHODS_TO_COMPARE`, so it is *supposed* to be in Fig 4 and is
silently dropped. `VCFP/DDMutPPI_vcfp_cache.pkl` (2026-09-08) holds 2,970 entries — all
`nan`. No script in `src/` produces DDMutPPI VCFP predictions; `ddmutppi_cv.py` covers GCV
only. Either restore the DDMutPPI submission path or remove it from `METHODS_TO_COMPARE`.

### O5 — SAAMBE-3D / MutPPI / MutPPI+ VCFP arrays :: no producer, and short by 2,936 rows

`VCFP/{SAAMBE-3D,MutPPI,MutPPIPlus} (Sahni+Fragoza train) (varchamp_full_pooled)_c*` total
**14,116 / 14,101** rows while every other method totals **17,052** — they missed the
`vc1pcava` supplement (17,052 − 2,936 = 14,116). `restratify_skempi_methods.py` only
*re-stratifies* existing arrays (in place, destructively); nothing generates the
predictions. `run_vcfp_blind_test.py` supports only `mutpredppi, esignet, mint, pplm,
swing`. C1 is empty (n=0) for all three, which is correct under SKEMPI stratification.

### O6 — `results_revisions/protein_class_enrichment/enrichment_by_protein_class.png`, `quasinull_fold_by_protein_class.png`, `.tsv` :: producer was rewritten away

Dated 2026-08-29. The current `protein_class_enrichment.py` emits only
`pathogenic_by_class.png/.tsv`. No writer for these names exists in `src/`, `archive/`, or
git history. The only surviving source is a Claude Code editor snapshot at
`/home/rcstewart/.claude/file-history/a865809c-.../6c5e23bdb2b55073@v2` (2026-08-28).
Not manuscript figures — **recommend deleting** rather than resurrecting.

### O7 — `results/*/roc_plots/roc_comparison*.png` (8 files, 2025-08 → 2026-05) :: superseded, unreproducible

`results/{sahni_cv,sahni_fragoza_cv,sahni_varchamp1p_cava_cv}/roc_plots/roc_comparison.png`,
`results/{cava,varchamp1p}_seqcnf_newvar_eval/roc_plots/roc_comparison_{to_swing,training_sets}.png`,
`results/varchamp_cava_combined/roc_plots/roc_comparison_*.png`. Written by pre-refactor
ancestors of `roc_plots.py` / `varchamp_blind_test.py`; the naming scheme no longer exists.
Superseded by `GCV/roc_plots_with_variance/` and `VCFP/roc_plots/`. **Recommend archiving.**

### O8 — `results_revisions/dataset_comparison/chart_*.png` (5) :: producer is archived

Written by `archive/dead_scripts_20260906/varchamp_dataset_comparison.py`. No longer runnable:
its inputs (`archive/pre_090826/data_caches/training_data_internal.csv`, moved 2026-09-10;
`B/2026/sfvc2026_labeled_data.csv`) are legacy CSVs, not canonical row tables. Not manuscript
figures.

### O9 — `figures/brca1/` and `figures/brca1_extra/` :: empty stubs

Each contains only an empty `results/` subdirectory. Real BRCA1 output lives in
`P/brca1/results/` and `P/brca1_extra/results/` as TSVs — **no figure was ever generated**.
`main.tex:266` has the BRCA1 figure commented out. Delete the stubs or build the figure.

### O11 — `datasets/annotations/plddt_pair_cache.pkl` :: DOES NOT EXIST; builder defaults to `/tmp` (blocks S5)

```
plddt_stratification.py:155  with open(PLDDT_CACHE, "rb") as f:
FileNotFoundError: .../datasets/annotations/plddt_pair_cache.pkl
```

`plddt_stratification.py:53` reads `ANNOTATIONS_DIR/plddt_pair_cache.pkl`. Its documented
builder `build_plddt_cache.py` has `DEFAULT_OUTPUT = tempfile.gettempdir()/"plddt_pair_cache.pkl"`
and **refuses to write into `datasets/annotations/` without `--force`**. The pair-keyed cache
was evidently built to `/tmp` and lost. This fires *before* the O2 `IndexError`, so S5 has
two independent blockers.

**Fix:** `python src/analysis/build_plddt_cache.py --output datasets/annotations/plddt_pair_cache.pkl --force`
(inputs are canonical: `datasets/af3_structures{,_variant_dbs}_canonical/manifest.csv`, both 2026-09-09).

Separately, `datasets/annotations/plddt_cache.pkl` (2026-09-09, 20 MB) is the **legacy
accession-keyed AFDB-monomer cache**, has **no producer in the repo**, and is now read only
under `build_plddt_cache.py --compare-legacy`. Per `archive/pre_090826/README.md` it is
superseded. Archive it.

### O12 — `variant_db_charts.py` input families with no producer (silent no-ops)

| input pattern | consumer flag | state |
|---|---|---|
| `results_revisions/variant_dbs_sfvfp/{base}/{ds}_posts.npy` | `--score-histograms` | **MISSING everywhere**; only stale copies in `archive/results_stale/variant_dbs/*/`. Emits **empty histograms** rather than failing |
| `results_revisions/variant_dbs_sfvfp/{base}/mean_n_partners{,_af,_onco,_tsg}.npy` | (loaded, never used) | **MISSING**; dead code |
| `results_revisions/variant_dbs_sfvfp/cosmic/cosmic_{4+,single}_tumor_site_to_edgotypes.pkl` | `--tumor-sites` | **MISSING**; warns and returns, so `cosmic_*_tumor_site_edgotypes.png`, `permutation_histogram.png`, `cv_comparison.png` **cannot be produced** |

`variant_db_charts.py` is the only file in the repo that mentions any of these names — no
writer exists in `src/`, `archive/`, or `repro_test/`. The main enrichment figure (Fig 5)
does **not** depend on them, so this is contained, but three chart types are dead.

### O13 — `datasets/annotations/pfam_domains_cache.pkl` :: no producer (S6)

2026-07-26, external InterPro/Pfam derived. `protein_class_stratification.py` is the only
file that references it; nothing builds it. Treat as **external** and preserve — a cold
rebuild has no path.

### O10 — inputs that cannot be regenerated at all

| input | consumer | why |
|---|---|---|
| `B/{clinvar,cosmic,gnomad}/prott5_embeddings.h5` | `compress_to_subgraphs.py` | deleted to reclaim disk; `prott5_subgraphs.h5` cannot be rebuilt (gap 2) |
| `--csv` MutPred2 output | `import_mutpred2_vcfp_scores.py` | required CLI arg, file not in tree |
| `B/gnomad/id_to_seq.pkl` | `variant_rows.py` | missing (code guards with `.exists()`) |
| `B/{hgmd,autism}/prott5_subgraphs.h5` | `run_variant_db_inference.py` | missing by design; falls back to **pre-090826 `af3_graphs/*.mat`** (2025-06) |
| `B/{db}/af3_graphs/*.mat` for 3,456 pairs | variant-DB inference | only local record of those complexes (gap 4) — **do not delete** |

---

## 5. Input classification by chain

### 5.1 GCV / ROC chain — Fig 3, S1, S3, S7 (`roc_plots.py`)

| input | class | state |
|---|---|---|
| `CV/{,sahni_fragoza_,combined_sahni_varchamp1p_cava_seq_confirmed_}all_vt_ids_and_labels.txt` | **pre-090826** | 1,487 / 5,894 / 4,423 — **O1** |
| `CV/{,sahni_fragoza_train_,sahni_varchamp1p_cava_train_}fold_splits_{0..29}.pkl` | canonical (only `sahni_fragoza_train_`, 6,219, 2026-09-09) | **mismatch vs labels** |
| `CV/{,swing_train_,combined_..._concat_clust_}pair_test_classes_{0..29}.npy` | canonical (`swing_train_` = 6,219) | mismatch |
| `CV/{prefix}fold_splits.pkl` (unsuffixed) | **pre-090826** | 2025-07/08, 1,487 / 5,894 / 4,423 |
| `GCV/*_detailed_results.pkl` (60 files) | **position-keyed** | 5,894-row ordering, per `archive/pre_090826/README.md` |
| `GCV/{dataset}_mutpred2_standalone_preds.npy` | **position-keyed** | 5,894 |
| `GCV/{dataset}_SAAMBE-3D_preds.npy` | **position-keyed** | 5,894 |
| `GCV/SAAMBE_train_uniprots.npy` | external (SKEMPI, 258 proteins) | fine |
| `datasets/annotations/all_to_uniprot.pkl` | external | 2025-09-11 |
| `datasets/annotations/confidence_scores.pkl` | canonical-buildable — `build_confidence_cache.py` | on disk **2026-05-31**, very stale |

**Verified failure** (`python -c "import roc_plots"`, module-level code):

```
roc_plots.py:197  ValueError: sahni_fragoza: fold_splits_0 indexes 6219 rows but the
cached comparison-method predictions cover 5894. The CV reference has been regenerated
on the canonical tables, so the MutPred2 / SAAMBE-3D / iptm caches must be recomputed
against the same row set before this figure can be rebuilt.
```

The `sahni` pass (`''` prefix, 1,487) completes first and **writes** `GCV/sahni_SAAMBE-3D_test_classes.npy`
and `GCV/iptm_sahni_gcv_splits.pkl` as an import side effect — be aware that merely
importing this module mutates `GCV/`.

### 5.2 Blind test chain — Fig 4, S2 (`varchamp_blind_test.py`)

| input | class | state |
|---|---|---|
| `VCFP/{method}_c{1,2,3}_{preds,labels,vt_ids}.npy` | **position-keyed** | 2026-08-27 → 09-07 |
| `datasets/sfvcfp_rows.csv.gz` (via `run_vcfp_blind_test.py`) | **canonical** | 2026-09-08, 22,338 rows |

**Silent staleness, quantified.** The canonical SFVCFP table carries `blind_test_class`
directly:

| stratum | canonical (`sfvcfp_rows.csv.gz`) | on disk (`VCFP/*.npy`) | delta |
|---|---|---|---|
| C1 | 9,236 | 1,867 | **−80%** |
| C2 | 7,077 | 7,248 | +2% |
| C3 | 6,025 | 7,937 | +32% |
| total | 22,338 | 17,052 | **−5,286** |

`varchamp_blind_test.py` reads whatever `.npy` are present and plots them. It cannot
detect this. **Fig 4 and S2 as published report a C1 AUROC computed on 20% of the C1
rows, under a completely different stratification rule.**

### 5.3 Robustness chain — S4, S5, S6, combined

| input | class | state |
|---|---|---|
| `CV/sahni_fragoza_train_all_vt_ids{,_0..29}.pkl` | **pre-090826** | 5,894 — **O2** |
| `CV/sahni_fragoza_train_fold_splits_{seed}.pkl` | canonical | 6,219 |
| `CV/swing_train_pair_test_classes_{seed}.npy` | canonical | 6,219 |
| `GCV/MutPredPPI_sahni_fragoza_megascale_all_detailed_results.pkl` | **position-keyed** | 5,894 |
| `datasets/annotations/plddt_pair_cache.pkl` | **unknown/missing** | **DOES NOT EXIST** — **O11** |
| `datasets/annotations/plddt_cache.pkl` | **unknown/missing producer** (legacy AFDB monomer) | 2026-09-09; read only via `--compare-legacy` |
| `datasets/annotations/pfam_domains_cache.pkl` | external (InterPro/Pfam), **no producer** — **O13** | 2026-07-26 |
| `datasets/training_eval/contact_graphs.h5` (interface residues) | **canonical** | 2026-09-09 20:41 |
| `datasets/training_eval/{sahni_fragoza_mapped090826_rows,sequences}.csv.gz` | **canonical** | 2026-09-08 |

**Verified by execution — all three die:**

```
interface_analysis.py:218            IndexError: list index out of range
protein_class_stratification.py:131  IndexError: list index out of range
plddt_stratification.py:155          FileNotFoundError: .../plddt_pair_cache.pkl
```

`combined_robustness_figure.py` calls `ia.compute_curves()`, `ps.compute_curves()`,
`pc.compute_curves()` in that order and therefore dies on the first.

`interface_analysis.py` is the worst mixture in the repo: **canonical** 2026-09-09 contact
graphs and sequences joined against a **pre-090826** vt_id ordering and **pre-090826**
predictions, in the same function.

`export_reconstruction_tables.py` reads the *same* mixed-vintage pair but has a
count-consistency guard — it does not crash, it **silently blanks** the
`vt_id/interactor/partner/variant` columns. That is worse: `datasets/reconstruction_tables/*.csv`
look fine and are unjoinable.

### 5.4 Variant-DB chain — Fig 5, S8, S9, `tab:variant_dbs_table`, cosmic table

| input | class | state |
|---|---|---|
| `datasets/variant_dbs/{clinvar,cosmic,gnomad,hgmd,autism}_rows.csv.gz` | **canonical** | **2026-09-09 18:50** |
| `datasets/variant_dbs/contact_graphs.h5`, `aliases.csv` | **canonical** | 2026-09-09 17:20 |
| `results_revisions/variant_dbs_sfvfp/{db}_mutpred_ppi_predictions.tsv` | **position-keyed** | **2026-08-19** — predates the canonical tables by 3 weeks |
| `results_revisions/variant_dbs_sfvfp/{db}/{db}_posterior_ls.pkl` | **position-keyed** | 2026-08-19/20 |
| `results_revisions/variant_dbs_sfvfp/all_bootstrap_results.pkl` (88 MB) | **position-keyed** | 2026-09-05, built on the 08-19 predictions |
| `results_revisions/variant_dbs_sfvfp/bootstrap_results_controlled_k{3,5,7}.pkl` | **position-keyed** | 2026-09-04 |
| `results_revisions/protein_class_annotations.csv` | external (UniProt/GO) | 2026-08-28 |
| `results_revisions/{,master_}variant_db_predictions.csv.gz` | **position-keyed** | 2026-08-24 (copy in `datasets/` is 09-04, byte-identical) |
| `datasets/annotations/clinvar/*_dirbind_variant_subset.pkl` | external (ClinVar) | 2025-06-13 |
| `datasets/annotations_licensed/{onco_tsg_dict,vt_to_tumor_site,hgmd_variant_subset}.pkl` | external (COSMIC/HGMD) | 2025 |
| `datasets/annotations/autism/variant_{subset,label_dict}.pkl` | external (Pejaver/Fu) | 2025-05-01 |
| `datasets/annotations/clingen_ar_ad_uniprot_sets.pkl` | external (ClinGen) — `build_ar_ad_gene_sets.py` | 2026-09-07 |
| `datasets/annotations/{gnomad,benign}_allele_frequencies.tsv` | external (gnomAD) | 2025 |

Every one of these scripts **runs to completion**. None of them checks that the prediction
TSVs match the canonical row tables. **The whole Fig 5 / S8 / S9 / table block is a silent
staleness, not a crash.**

**Model-directory mismatches — fix these while regenerating.** Two prediction trees exist:
`results_revisions/variant_dbs/` (SF model, 2026-08-10/11) and
`results_revisions/variant_dbs_sfvfp/` (SFVCFP model, 2026-08-19). Per
`project_active_datasets`, **SFVCFP is the only live set**, yet four scripts still read the
SF tree:

| script | reads | should read |
|---|---|---|
| `stability_interaction_clustering.py` | `variant_dbs/` | `variant_dbs_sfvfp/` — and it imports its loaders from `stability_interaction_scatter.py`, which reads `variant_dbs_sfvfp/` |
| `stability_interaction_comparison.py` | `variant_dbs/` | `variant_dbs_sfvfp/` |
| `fetch_protein_class_annotations.py` | `variant_dbs/*.tsv` (glob) | `variant_dbs_sfvfp/` — its output `protein_class_annotations.csv` is consumed by `protein_class_enrichment.py`, which reads the *sfvfp* tree. Affects which accessions get GO-annotated, not the scores. |
| `make_master_variant_db_csv.py` | `SF_TSV` dict → `variant_dbs/` | dead code — the dict is defined but never used in `main()` |

### 5.5 Obsolete dataset configurations still wired into figures

`gcv_common.DATASET_CONFIGS` and `export_cv_reference.NAMING` cover **five** datasets, all
`*_mapped090826`: `sahni_only`, `fragoza_only`, `sahni_fragoza`, `varchamp_all`,
`sahni_fragoza_varchamp_all`. Anything else in `CV/` has **no regeneration path**:

| prefix | rows | last built | canonical? |
|---|---|---|---|
| `sahni_fragoza_varchamp1p_cava_train_` | 8,827 | 2026-09-05 | **no** |
| `sahni_varchamp1p_cava_train_` | 4,423 | 2026-09-05 | **no** — consumed by `roc_plots.py` dataset 3 |
| `sahni_fragoza_varchamp2026_train_` | 7,369 | 2026-08-17 | **no** |
| `sahni_fragoza_varchamp_{full,pooled,full_pooled}_train_` | 10,305 / 17,910 / 22,338 | 2026-09-05/08 | **no** (`full_pooled` is SFVCFP; use `datasets/sfvcfp_rows.csv.gz`) |
| `sahni_fragoza_{af2,inter}_train_` | 4,892 / 5,894 | 2026-08 | **no** |
| `varchamp_pooled_only_train_` | 12,210 | 2026-08-18 | **no** |

Per `project_active_datasets`: SFVCFP is the only live SF+VarChAMP set. `roc_plots.datasets`
still hardcodes `sahni_varchamp1p_cava` as its third dataset — that figure
(`roc_sahni_fragoza_varchamp1p_cava_with_variance.png`) is **unregenerable** and should be
dropped or repointed at `sahni_fragoza_varchamp_all`.

---

## 6. Runnability matrix — tested vs inferred

| script | verdict | how established |
|---|---|---|
| `roc_plots.py` (+ `run_roc_comparison.py`, `run_roc_ablation.py`) | **DIES** — `ValueError` at line 197 | **TESTED** — executed, traceback captured verbatim |
| `protein_class_stratification.py` | **DIES** — `IndexError` at line 131 | **TESTED** — executed to failure |
| `interface_analysis.py` | **DIES** — `IndexError` at line 218 | **TESTED** — executed to failure |
| `plddt_stratification.py` | **DIES** — `FileNotFoundError` at line 155, then `IndexError` | **TESTED** — executed to failure |
| `combined_robustness_figure.py` | **DIES** | inferred — calls `compute_curves()` on all three above |
| `biclass_sf_gcv.py` | **DIES** | **TESTED** — `import biclass_sf_gcv` triggers `roc_plots` module-level exception |
| `varchamp_blind_test.py` | RUNS, silently stale | **TESTED** import; staleness proven by array-length/class-count comparison |
| `variant_db_charts.py` | RUNS, silently stale | **TESTED** import; all inputs exist, all dated 2026-08-19/09-05 |
| `protein_class_enrichment.py` | RUNS, silently stale | **TESTED** import; inputs exist |
| `threshold_sensitivity.py` | RUNS, silently stale | **TESTED** import; inputs exist |
| `cosmic_onco_tsg_stat_test.py` | RUNS, silently stale | **TESTED** import; inputs exist |
| `stability_interaction_{scatter,clustering,comparison}.py` | RUNS, silently stale | **TESTED** import; inputs exist |
| `iptm_analysis.py` | RUNS, silently stale | **TESTED** import |
| `generate_training_table.py` | RUNS, silently stale | **TESTED** import; reads 2025-dated label `.txt` |
| `extract_variant_db_stats.py` | RUNS, silently stale | **TESTED** import |
| `export_reconstruction_tables.py` | RUNS, **silently blanks id columns** | inferred from the count-consistency guard + verified 5,894 vs 6,219 inputs |
| `export_cv_reference.py` | RUNS, canonical | inputs are `training_eval/*` — the only purely canonical figure-chain script |
| `run_vcfp_blind_test.py` | RUNS, canonical | **TESTED** import; reads `sfvcfp_rows.csv.gz` |

> Import note: `stability_interaction_clustering.py`, `cosmic_onco_tsg_stat_test.py` and
> `export_reconstruction_tables.py` need `src/analysis` on `sys.path` (they are
> `python src/analysis/X.py` entry points). Not a defect.

---

## 7. Regeneration order

Each stage lists what unblocks. **Do not skip a stage** — every later stage is
position-keyed against an earlier one.

### Stage 0 — finish the AF3 run *(in progress)*
- Complete `af3_missing_090826.log` (900 lines of driver output; inputs at
  `B/MAPPING_090826/af3_inputs_small/`). Driver `run_af3_ross4.sh` lives at
  `/data/ross/af3_mmseqs2/` — **not in the repo** (`REPRODUCIBILITY.md` gap 6).
- `python src/data_processing/canonicalize_structures.py` → `datasets/af3_structures_canonical/`
  (currently 3,855 of 4,301) and `datasets/af3_structures_variant_dbs_canonical/` (22,240).
- **Unblocks:** nothing directly.

### Stage 1 — contact graphs + confidence caches
```
python src/data_processing/rebuild_graphs_from_structures.py      # -> datasets/training_eval/contact_graphs.h5
python repro_test/build_af3_index.py                     # -> datasets/training_eval/af3_index.csv.gz
# NOTE both builders default to /tmp and refuse to write datasets/annotations/ without --force
python src/analysis/build_plddt_cache.py \
    --output datasets/annotations/plddt_pair_cache.pkl --force        # O11 -- currently MISSING
python src/analysis/build_confidence_cache.py \
    --output datasets/annotations/confidence_scores.pkl --force       # on disk: 2026-05-31, from "old_train_confidences"
```
- **Unblocks:** nothing yet. `confidence_scores.pkl` is the oldest live input in the tree
  (2026-05-31, sourced from `B/three_datasets_af3_models/old_train_confidences/json/`) and
  gates the iptm strata inside `roc_plots.py`. Repoint `--json-dir` at the new AF3 output.
- `plddt_pair_cache.pkl` **does not exist at all** — S5 cannot run until this step.

### Stage 2 — canonical row tables
```
python repro_test/build_canonical_tables.py              # -> datasets/training_eval/*_rows.csv.gz, sequences.csv.gz
python repro_test/build_sfvcfp_table.py                  # -> datasets/sfvcfp_rows.csv.gz            (22,338)
python src/variant_db_inference/build_variant_db_tables.py --db all   # -> datasets/variant_dbs/*_rows.csv.gz
python src/analysis/export_cv_reference.py --dataset all # -> CV/*rows.csv.gz, fold_splits, pair_test_classes, clusters
```
- **⚠ MUST ALSO FIX HERE:** teach `export_cv_reference.py` to emit
  `{prefix}all_vt_ids_and_labels.txt` (**O1**) or migrate its six readers, and delete
  `CV/*all_vt_ids*.pkl` after migrating its five readers (**O2**). *Nothing downstream can
  be trusted until this is done — see §8.*
- **Unblocks:** nothing yet.

### Stage 3 — embeddings
```
python src/data_processing/precompute_prott5_datasets.py
python src/data_processing/precompute_esm2_datasets.py
python src/evaluation/precompute_mint_embeddings.py       # CUDA_VISIBLE_DEVICES required
python src/evaluation/precompute_pplm_embeddings.py       # CUDA_VISIBLE_DEVICES required
python src/variant_db_inference/precompute_prott5.py
python src/variant_db_inference/compress_to_subgraphs.py  # clinvar/cosmic NOT re-runnable (O10)
```
- **Unblocks:** nothing. Note `datasets/training_eval/*_{prott5,esm2,mint,pplm}.pkl` were
  already rebuilt 2026-09-08/09 — verify against the new graphs before trusting them.

### Stage 4 — training
```
python src/training/preprocess_stability_data.py
python src/training/pretrain_stability.py                 # -> weights/MutPred-PPI_stability_pretrain.pt
python src/training/train_final_model.py --dataset ...    # -> weights/MutPred-PPI*.pt
```
- **Unblocks:** nothing. `datasets/mega_splits.pkl` is **external** (SPURS train/val/test
  protein splits) — do not attempt to regenerate it. `megascale_preprocessed/preprocessed.pkl`
  is **not** external, despite an earlier version of this doc: `preprocess_stability_data.py`
  above generates it directly, and it is the byte-identical gate for the §A/§B contact-graph
  builder unification (2026-09-10) — a corrected run must reproduce it exactly.

### Stage 5 — GCV
```
python src/evaluation/mutpred_ppi_gcv.py --dataset sahni_fragoza_mapped090826 [--ablation ...]
python src/evaluation/esignet_cv.py / mint_cv.py / pplm_cv.py / swing_gcv.py / saambe3d_cv.py / ddmutppi_cv.py
```
- Writes `GCV/*_detailed_results.pkl` + `*_macro_aucs.npy` on the **6,219-row** ordering.
- **⚠ Also regenerate here:** `GCV/{dataset}_mutpred2_standalone_preds.npy` and
  `GCV/{dataset}_SAAMBE-3D_preds.npy` — `roc_plots.py`'s own error message names these as
  the blockers. `saambe3d_cv.py` and `ddmutppi_cv.py` read the O1 label `.txt` files, so
  Stage 2's fix must land first.
- **UNBLOCKS: Fig 3, S1, S3, S7** (`run_roc_comparison.py`, `run_roc_ablation.py`) and
  the non-manuscript `biclass_gcv` + `spearman_box` figures.
- **UNBLOCKS: S4, S5, S6 + combined robustness**, *provided* O2 is fixed.

### Stage 6 — VCFP blind test
```
for m in mutpredppi esignet mint pplm swing; do
  python src/evaluation/run_vcfp_blind_test.py --method $m [--predictor seq_diff|site_diff] [--test-pretrain]
done
python src/analysis/import_mutpred2_vcfp_scores.py --csv <MutPred2 output>   # external input (O10)
python src/analysis/restratify_skempi_methods.py                            # SAAMBE-3D/MutPPI/MutPPI+ — in place!
```
- All arrays must come out at **22,338** rows (C1 9,236 / C2 7,077 / C3 6,025).
- **⚠ Unresolved:** SAAMBE-3D / MutPPI / MutPPI+ / DDMutPPI have **no prediction producer**
  (**O4**, **O5**). Decide now: restore them or drop them from `METHODS_TO_COMPARE`.
- **UNBLOCKS: Fig 4, S2.**

### Stage 7 — variant-DB inference
```
python src/variant_db_inference/run_variant_db_inference.py --db {clinvar,cosmic,gnomad,hgmd,autism} \
    --store datasets/variant_dbs/contact_graphs.h5 --out results_revisions/variant_dbs_sfvfp/{db}_mutpred_ppi_predictions.tsv
python src/variant_db_inference/run_stability_inference.py --dataset {…}
```
- Replaces the **2026-08-19** TSVs. hgmd/autism fall back to pre-090826 `.mat` graphs —
  see O10.
- **UNBLOCKS:** nothing yet.

### Stage 8 — classification + bootstrap caches
```
python src/analysis/fetch_protein_class_annotations.py     # -> results_revisions/protein_class_annotations.csv
python src/analysis/build_ar_ad_gene_sets.py               # -> datasets/annotations/clingen_ar_ad_uniprot_sets.pkl
python src/analysis/classify_variant_dbs.py --pred-dir results_revisions/variant_dbs_sfvfp
python src/analysis/make_master_variant_db_csv.py          # -> results_revisions/master_variant_db_predictions.csv.gz
python src/analysis/variant_db_charts.py --bootstrap       # -> all_bootstrap_results.pkl, bootstrap_results_controlled_k{3,5,7}.pkl
```
- **UNBLOCKS:** nothing yet (these are the caches Stage 9 reads).

### Stage 9 — figures and tables
```
# Fig 5, S8
python src/analysis/variant_db_charts.py --output-dir results_revisions/variant_dbs_sfvfp
python src/analysis/variant_db_charts.py --output-dir results_revisions/variant_dbs_sfvfp --controlled-k 3
# S9
python src/analysis/threshold_sensitivity.py
# non-manuscript
python src/analysis/protein_class_enrichment.py
python src/analysis/stability_interaction_scatter.py
python src/analysis/stability_interaction_clustering.py
python src/analysis/stability_interaction_comparison.py
# tables  (then RE-PASTE into main.tex / supplement.tex by hand — there is no \input)
python src/analysis/generate_training_table.py             # -> figures/training_data_table.tex
python src/analysis/extract_variant_db_stats.py            # -> figures/variant_db_stats_table.tex
python src/analysis/cosmic_onco_tsg_stat_test.py           # -> figures/cosmic_onco_tsg_qn_vs_edgetic.tex
# reconstruction tables (only after O2 is fixed)
python src/analysis/export_reconstruction_tables.py
```
- **UNBLOCKS: Fig 5, S8, S9, all 3 tables.**

### Stage 10 — manual, no script
- **Redraw Fig 1 (`MutPred-PPI_pipeline.png`) and Fig 2 (`MutPred-PPI_architecture.png`)** —
  **O3**. Nothing else in this document can be automated away; these two cannot be
  automated at all. Start now; they do not depend on the AF3 run.
- Decide the fate of `figures/brca1*` (**O9**) and the commented-out `fig:brca1_example`.
- Re-paste the three regenerated `.tex` table fragments.

---

## 8. Pre-flight checklist

Run these **before** Stage 5. Each is a one-liner that catches a silent misalignment that
would otherwise survive into a published number.

```bash
P=/data/ross/ppi_lossgain/interaction_loss/publication

# 1. Label file must match canonical rows  (currently 5894 vs 6219 -- FAILS)
wc -l $P/datasets/cv_reference/sahni_fragoza_all_vt_ids_and_labels.txt
python -c "import pandas;print(len(pandas.read_csv('$P/datasets/cv_reference/sahni_fragoza_train_rows.csv.gz')))"

# 2. fold_splits, pair_test_classes and vt_ids must agree  (currently 6219/6219/5894 -- FAILS)
python - <<'EOF'
import pickle, numpy as np
cv='/data/ross/ppi_lossgain/interaction_loss/publication/datasets/cv_reference/'
fs=pickle.load(open(cv+'sahni_fragoza_train_fold_splits_0.pkl','rb'))
print('fold_splits      ', max(max(t) for _,_,t in fs)+1)
print('pair_test_classes', len(np.load(cv+'swing_train_pair_test_classes_0.npy')))
print('all_vt_ids       ', len(pickle.load(open(cv+'sahni_fragoza_train_all_vt_ids_0.pkl','rb'))))
EOF

# 3. detailed_results must cover the canonical row count  (currently 5894 -- FAILS)
python -c "
import pickle
it=pickle.load(open('$P/results_revisions/macro_aucs/ESigNet_sahni_fragoza_detailed_results.pkl','rb'))['iterations'][0]
print(sum(len(f[c]['preds']) for f in it['folds'].values() for c in ('class_1','class_2','class_3')))"

# 4. Blind-test arrays must match sfvcfp_rows.csv.gz  (currently 17052 vs 22338 -- FAILS)
python -c "
import pandas,numpy as np
d=pandas.read_csv('$P/datasets/sfvcfp_rows.csv.gz'); print('canonical', d.blind_test_class.value_counts().sort_index().tolist())
m='$P/results/varchamp_seqcnf_newvar_eval/MutPred-PPI (megascale_all, all-data) (varchamp_full_pooled)'
print('on disk  ', [len(np.load(f'{m}_c{c}_preds.npy')) for c in (1,2,3)])"

# 5. Prediction TSVs must be newer than the canonical variant-DB tables  (currently 08-19 < 09-09 -- FAILS)
ls -l --time-style=+%F $P/datasets/variant_dbs/clinvar_rows.csv.gz \
                       $P/results_revisions/variant_dbs_sfvfp/clinvar_mutpred_ppi_predictions.tsv

# 6. Caches that must simply EXIST  (plddt_pair_cache.pkl currently absent -- FAILS)
ls -l $P/datasets/annotations/plddt_pair_cache.pkl \
      $P/datasets/annotations/confidence_scores.pkl \
      $P/datasets/annotations/pfam_domains_cache.pkl
```

**All six currently fail.** They should all pass before any figure is regenerated.

---

## 9. The single biggest risk

**`datasets/cv_reference/*_all_vt_ids_and_labels.txt` has no producer in the entire git
history, and no canonical equivalent.**

Six scripts read these files. `sahni_fragoza_all_vt_ids_and_labels.txt` is dated
**2025-08-25**, holds the retired **5,894** ordering, and is the sole source of the
per-row labels that every baseline curve (MutPred2, SAAMBE-3D, iptm) in Fig 3 / S1 / S3 /
S7 is computed against. `export_cv_reference.py` — the script that regenerated everything
else in that directory on 2026-09-09 — does not write it and never did.

Until an emitter exists (or the six readers move to `*_rows.csv.gz`), the GCV figures
cannot be regenerated *at all*, and the row count they would need does not exist in any
producible artifact. This is a hard blocker that sits **upstream of the AF3 run's benefit**:
finishing AF3 will not fix it, and discovering it after AF3 completes costs a full
re-run of Stage 5.

Second place: **Fig 4 / S2 do not fail — they lie.** The C1 stratum shrank from 9,236
canonical rows to 1,867 on disk (−80%) under a changed stratification rule, and
`varchamp_blind_test.py` has no guard that can notice.
