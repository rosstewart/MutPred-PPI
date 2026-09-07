# Reproducing Every Figure and Table

Cross-validation benchmarking, the VarChAMP blind test, variant-repository inference/classification/
charts, and supplementary analyses. For training the model from scratch, see
[`docs/TRAINING.md`](TRAINING.md). Pre-computed prediction/label tables that reconstruct every
figure's curves without rerunning anything are in
[`datasets/reconstruction_tables/`](../datasets/reconstruction_tables/README.md).

## Grouped Cross-Validation (Fig 3, S1)

```bash
# MutPred-PPI GCV (30 seeds, Sahni+Fragoza dataset)
conda run -n ppi python src/evaluation/mutpred_ppi_cv.py \
    --dataset sahni_fragoza --ablation megascale_all --device cuda:0

# Comparator methods (require external installations)
conda run -n ppi python src/evaluation/swing_cv.py    --dataset sahni_fragoza
conda run -n ppi python src/evaluation/esignet_cv.py  --dataset sahni_fragoza
conda run -n ppi python src/evaluation/mint_cv.py     --dataset sahni_fragoza
conda run -n ppi python src/evaluation/pplm_cv.py     --dataset sahni_fragoza
conda run -n ppi python src/evaluation/saambe3d_cv.py --dataset sahni_fragoza
```

MINT/PPLM sequence embeddings must be precomputed first:

```bash
conda run -n ppi python src/evaluation/precompute_mint_embeddings.py
conda run -n ppi python src/evaluation/precompute_pplm_embeddings.py
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

```bash
conda run -n ppi python src/analysis/roc_plots.py
```

## Ablation figure (S-abl)

```bash
conda run -n ppi python src/analysis/run_roc_ablation.py
```

The `full`/`full_all` ablation ("Prior Best" bar) uses `weights/v1_0/MutPred-PPI_v1_0_stability_pretrain.pt`,
a pre-MegaScale (FoldX/RaSP-based) checkpoint kept for this one comparison; all other ablations use
`weights/MutPred-PPI_stability_pretrain.pt`. If `weights/v1_0/` is unavailable, skip `full`/`full_all`.

Output: `results_revisions/macro_aucs/roc_plots_with_variance/ablation_bar_sahni_fragoza_with_variance.png` → **S-abl**
