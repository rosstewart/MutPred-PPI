# Reproducing Every Figure and Table

Cross-validation benchmarking, the VarChAMP blind test, variant-repository inference/classification/
charts, and supplementary analyses. For training the model from scratch, see
[`docs/TRAINING.md`](TRAINING.md). Pre-computed prediction/label tables that reconstruct every figure's curves without rerunning
anything are in `datasets/reconstruction_tables/`. `datasets/` is gitignored and delivered via the
Zenodo bundle -- see [`docs/DATA_SOURCES.md`](DATA_SOURCES.md).

## The canonical data layer

Everything reads two tables per dataset from `datasets/mapped090826/`, built from the 090826
mapping by `repro_test/build_canonical_tables.py`:

```
<dataset>_rows.csv.gz    row_index, interactor, partner, mutation, position, wt_aa, mut_aa,
                         perturbed, dataset, dataset_tier, fragoza_source, source_row_id, cluster
<dataset>_splits.csv.gz  seed, row_index, test_fold, test_class
sequences.csv.gz         accession, sequence
af3_index.csv.gz         seq_a_sha, seq_b_sha, len_a, len_b, path, chain_a_is_first
```

Guarantees, asserted at build time: every `mutation` is 1-based and validated against its
sequence, accessions are UniProt (isoform suffix only where the sequence differs from canonical),
no duplicate `(interactor, partner, mutation)`, no null labels, `row_index` contiguous and never
renumbered. **The pipeline is 1-based end to end** -- mutation strings, embedding-cache keys and
the tables all agree, so nothing converts between conventions. Node indices (`mutation_idx`) stay
0-based because they address a graph row, not a residue in a mutation string.

There is no `--data-root`: the tables locate themselves.

The five live datasets:

| `--dataset` | rows |
|---|---|
| `sahni_fragoza_varchamp_all_mapped090826` | 23,320 |
| `varchamp_all_mapped090826` | 17,376 |
| `sahni_fragoza_mapped090826` | 6,219 |
| `fragoza_only_mapped090826` | 4,729 |
| `sahni_only_mapped090826` | 1,595 |

Rebuild them with:

```bash
conda run -n ppi python repro_test/build_canonical_tables.py          # rows + splits + sequences
conda run -n ppi python repro_test/build_af3_index.py                 # structure index, by sequence
```

## Grouped Cross-Validation (Fig 3, S1)

Every trained method runs through one shared runner (`gcv_common.run_gcv`); only the training loop
differs per method.

```bash
DS=sahni_fragoza_varchamp_all_mapped090826

# MutPred-PPI (graph + ProtT5). --ablation selects the freeze strategy;
# megascale_all (default) freezes nothing.
conda run -n ppi python src/evaluation/mutpred_ppi_gcv.py --dataset $DS --device cuda:0

# Comparator methods
conda run -n ppi python src/evaluation/swing_gcv.py   --dataset $DS                  # blind test
conda run -n ppi python src/evaluation/swing_gcv.py   --dataset $DS --test-pretrain  # leaky variant
conda run -n ppi python src/evaluation/esignet_cv.py  --dataset $DS --device cuda:1
# MINT/PPLM each have two predictor variants; roc_plots expects BOTH.
for pred in seq_diff site_diff; do
  conda run -n ppi python src/evaluation/mint_cv.py --dataset $DS --predictor $pred
  conda run -n ppi python src/evaluation/pplm_cv.py --dataset $DS --predictor $pred
done
```

Embedding caches must be precomputed first. All caches are keyed on the **1-based** mutation
exactly as the tables store it:

```bash
conda run -n ppi python src/data_processing/precompute_prott5_datasets.py --dataset $DS --device cuda:0
conda run -n ppi python src/evaluation/precompute_mint_embeddings.py      --dataset $DS --compute-residue
conda run -n ppi python src/evaluation/precompute_pplm_embeddings.py      --dataset $DS
# eSIG-Net: use eSIG-Net's own precompute script (see its repository)
```

### Biclass SF GCV (S-biclass)

Restricts Fig 3's cross-validation to ordered protein pairs (A, B) where mutations in A include
both disruptive and non-disruptive labels.

```bash
conda run -n ppi python src/analysis/biclass_sf_gcv.py
```

Output: `results_revisions/biclass_gcv/roc_sahni_fragoza_biclass_with_variance.png` → **S-biclass**

## VarChAMP Blind Test (Fig 4, S2)

VarChAMP data is unpublished IGVF consortium data — cross-reference [data.igvf.org](https://data.igvf.org).

```bash
# Main blind test per method
conda run -n ppi python src/evaluation/mutpred_ppi_cv.py \
    --dataset sahni_fragoza_varchamp_full_pooled --device cuda:0
# (repeat for eSIG-Net, SWING, MINT, PPLM — see their respective cv scripts)

# One command per method: trains/predicts, then merges and restratifies automatically
conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method mutpredppi
conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method esignet
conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method mint --predictor seq_diff
conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method mint --predictor site_diff
conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method pplm --predictor seq_diff
conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method pplm --predictor site_diff
conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method swing
conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method swing --test-pretrain

# MutPred2 (parse-only; no model to train)
conda run -n ppi python src/analysis/import_mutpred2_vcfp_scores.py --csv /path/to/mutpred2_output.csv

# Generate Fig 4 + S2
conda run -n ppi python src/analysis/varchamp_blind_test.py
```

Methods that require PDB structures (SAAMBE-3D, MutPPI, MutPPI+, DDMutPPI) cannot predict
gene-symbol-keyed entries and are evaluated on a smaller subset; MutPred2 and the GNN-based
methods cover the full test set.

## Variant Database Inference (Fig 5–7)

ProtT5 embeddings must be precomputed before inference (large, resume-safe):

```bash
nohup conda run -n ppi python src/variant_db_inference/precompute_prott5.py \
    --fasta /path/to/gnomad_wt_and_vt.fasta --out prott5_embeddings.h5 \
    --device cuda:0 > precompute.log 2>&1 &

conda run -n ppi python src/variant_db_inference/run_variant_db_inference.py \
    --dataset gnomad --device cuda:0
```

HGMD and COSMIC require licensed access. COSMIC can be enabled with `--include-cosmic`; HGMD is
excluded from all distributed files.

## Variant Database Classification and Chart Generation

```bash
conda run -n ppi python src/analysis/classify_variant_dbs.py \
    --output-dir results_revisions/variant_dbs_sfvfp

conda run -n ppi python src/analysis/variant_db_charts.py \
    --data-dir results_revisions/variant_dbs_sfvfp \
    --edgotype-bootstrap --controlled-bootstrap --k3-only
```

Output:
- `enrichment_bootstrap_sufficient_partners.png` → **Fig 5** (ClinVar row includes Rare Benign/Benign/Pathogenic/VUS/Pathogenic AR/Pathogenic AD; HGMD row includes HGMD/AR/AD)
- `enrichment_bootstrap_sufficient_partners_k3.png` → **S4** (same grouping, partner-controlled)

### Gene inheritance-mode (AR/AD) mapping

Required once, before Fig 5/S4/S-stability:

```bash
conda run -n ppi python src/analysis/build_ar_ad_gene_sets.py
```

Produces a gene→UniProt AR/AD mapping (mutually exclusive sets) from ClinGen MOI curations — see
[`docs/DATA_SOURCES.md`](DATA_SOURCES.md). `classify_variant_dbs.py` consumes this directly; no
separate re-run step is needed.

### COSMIC Onco/TSG QN vs. Edgetic stat test (S-cosmic-stat)

```bash
conda run -n ppi python src/analysis/cosmic_onco_tsg_stat_test.py
```

Output: `results_revisions/cosmic_stat_test/cosmic_onco_tsg_qn_vs_edgetic.tex` → **S-cosmic-stat**

### Protein class enrichment (S-protclass)

```bash
conda run -n ppi python src/analysis/protein_class_enrichment.py
```

Output: `results_revisions/protein_class_enrichment/pathogenic_by_class.png` → **S-protclass**

### Stability vs. interaction per-variant scatter (S-stability, 6 panels)

```bash
conda run -n ppi python src/analysis/stability_interaction_scatter.py --cosmic-min-recurrence 32
```

Output: `results_revisions/stability_interaction/scatter_per_variant_kde.png` → **S-stability**
(ClinVar Pathogenic, ClinVar Benign, ClinVar VUS, gnomAD, HGMD, COSMIC recurrence≥32)

### Robustness analyses

```bash
conda run -n ppi python src/analysis/interface_analysis.py             # interface vs. non-interface
conda run -n ppi python src/analysis/plddt_stratification.py           # AF3 pLDDT quality
conda run -n ppi python src/analysis/threshold_sensitivity.py
conda run -n ppi python src/analysis/protein_class_stratification.py   # single- vs. multi-domain

# Combined 3-row figure (panels A/B/C)
conda run -n ppi python src/analysis/combined_robustness_figure.py
```

Output in `results_revisions/robustness_analyses/`: each script's own `*_auroc_by_class.png` +
`.tsv`, plus `combined_robustness_by_class.png`.

## Table Generation

```bash
conda run -n ppi python src/analysis/generate_training_table.py    # figures/training_data_table.tex
conda run -n ppi python src/analysis/extract_variant_db_stats.py   # figures/variant_db_stats_table.tex
```

## ROC/AUC Figures

`roc_plots.py` is a library plus a notebook-style driver, not the entry point --
importing it executes both the ROC generation and the separate ipTM analysis.
Use the wrappers:

```bash
conda run -n ppi python src/analysis/run_roc_comparison.py   # Fig 3, S1, S3, S-new
conda run -n ppi python src/analysis/run_roc_ablation.py     # S-abl
```

## Ablation figure (S-abl)

```bash
conda run -n ppi python src/analysis/run_roc_ablation.py
```

The `full`/`full_all` ablation ("Prior Best" bar) uses `weights/v1_0/MutPred-PPI_v1_0_stability_pretrain.pt`,
a pre-MegaScale (FoldX/RaSP-based) checkpoint kept for this one comparison; all other ablations use
`weights/MutPred-PPI_stability_pretrain.pt`. If `weights/v1_0/` is unavailable, skip `full`/`full_all`.

Output: `results_revisions/macro_aucs/roc_plots_with_variance/ablation_bar_sahni_fragoza_with_variance.png` → **S-abl**
