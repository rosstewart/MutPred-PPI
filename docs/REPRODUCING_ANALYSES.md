# Reproducing Every Figure and Table

Cross-validation benchmarking, the VarChAMP blind test, variant-repository inference/classification/
charts, and supplementary analyses. For training the model from scratch, see
[`docs/TRAINING.md`](TRAINING.md). Pre-computed prediction/label tables that reconstruct every figure's curves without rerunning
anything are in `datasets/reconstruction_tables/`. `datasets/` is gitignored and delivered via the
Zenodo bundle -- see [`docs/DATA_SOURCES.md`](DATA_SOURCES.md).

**One command runs everything below in order:** `notebooks/reproduce_all_figures.py`
(jupytext percent format -- `jupytext --to notebook` for a `.ipynb`, or run it directly as a
script) caches every step, generates missing embeddings on first use, displays each figure
inline, and writes to the exact paths this document describes. Set `QUICK = True` in its
header for a fast (hours, not days) smoke test that never touches the canonical `results/`
tree -- see the notebook's own docstring cell.

## The canonical data layer

Everything reads two tables per dataset from `datasets/training_eval/`, built from the 090826
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

### Contact graphs

Graphs are not files on disk any more. Each tier has one HDF5 `ContactGraphStore`
([`src/contact_graphs.py`](../src/contact_graphs.py)):

| Store | Used by |
|---|---|
| `datasets/training_eval/contact_graphs.h5` | GCV, blind test, final training |
| `datasets/variant_dbs/contact_graphs.h5` | variant-database inference |

Keys are the sorted pair of `sha256(chain_sequence)[:16]`, so a pair of sequences has one key no
matter which accessions or chain order produced it. `load_dense(interactor=..., partner=...)` and
`load_edge_index(interactor=..., partner=...)` are keyword-only, take **sequences**, and hand back
the graph already oriented so that the interactor occupies nodes `[0, len(interactor))`. Self-loops
are added on read and cannot be disabled. That orientation-on-read is what retired the old `NRR`
chain-split integer, and with it the 121 clinvar / 121 cosmic / 116 gnomad / 29 hgmd complexes
whose filenames claimed the wrong chain order.

Rebuild a store from the canonical structures:

```bash
conda run -n ppi python src/data_processing/rebuild_graphs_from_structures.py \
    --structures datasets/af3_structures_canonical \
    --out datasets/training_eval/contact_graphs.h5 \
    [--compare-to datasets/training_eval/contact_graphs.h5] [--n-jobs 16]
```

### AF3 structures

`datasets/af3_structures_canonical/` (3,854 pairs) and
`datasets/af3_structures_variant_dbs_canonical/` (22,239 pairs) hold one gzipped mmCIF per pair:

```
{ACC_LO}__{ACC_HI}.cif.gz        e.g.  O14787-2__Q13207.cif.gz
manifest.csv                     filename, chain_a_accession, chain_b_accession, chain_a_id,
                                 chain_b_id, len_a, len_b, seq_a_sha, seq_b_sha,
                                 old_name_first_token, mean_plddt, source_format,
                                 n_candidates, source
```

Accessions are uppercase and **sorted**, joined by `__` (no accession contains an underscore), so
the name survives isoform and RefSeq-style ids and **encodes no orientation**. Where AF3 produced
several models of a pair, the one with the highest mean pLDDT wins, with `.cif` breaking exact
ties. Built by `src/data_processing/canonicalize_structures.py`; names are derived from chain *contents*,
never from the old `fold_a_b_model_0` filenames.

### Variant-database tables

One self-contained table per database, replacing the scattered annotation pickles at the point of
use:

```
datasets/variant_dbs/{clinvar,cosmic,gnomad,hgmd,autism}_rows.csv.gz
```

Shared columns: `interactor`, `partner`, `mutation` (**1-based**), `pair_key` (the contact-graph
content address — sequences are not inlined), `clingen_moi`, `in_embedding_store`. Per-DB
annotations follow: `clinical_significance`/`allele_frequency` (clinvar), `allele_frequency`
(gnomad), `recurrence`/`tumor_sites`/`onco_tsg` (cosmic), `neurodev_label` (autism); hgmd carries
the shared columns only. Row counts: clinvar 949,065, cosmic 1,447,917, gnomad 10,529,577,
hgmd 56,266, autism 19,148.

```bash
conda run -n ppi python src/variant_db_inference/build_variant_db_tables.py --db all
```

The pickles remain the source; these are a derived view. Zero-based conversion (FASTA headers,
ProtT5 keys, subgraph H5 variant keys) happens only through
`src/variant_db_inference/variant_rows.py::to_zero_based`, never inline.

## Grouped Cross-Validation (Fig 3, S1)

Every trained method runs through one shared runner
([`src/utils/gcv_common.py`](../src/utils/gcv_common.py)`::run_gcv`, which also holds
`DATASET_CONFIGS`, `load_data` and the split loading); only the training loop differs per method.
MutPred-PPI's loop is `src/training/train_fold.py::train_fold`, imported by both
`mutpred_ppi_gcv.py` and `train_final_model.py` — it has no CLI of its own.

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
# Pretrained (SKEMPI), not retrained -- SAAMBE-3D was already canonical;
# MutPPI/MutPPI+ are migrated in from an external, unversioned script.
conda run -n ppi python src/evaluation/saambe3d_cv.py --dataset $DS --outdir results/gcv/
conda run -n ppi python src/evaluation/mutppi_cv.py   --dataset $DS --model 0 --outdir results/gcv/  # MutPPI
conda run -n ppi python src/evaluation/mutppi_cv.py   --dataset $DS --model 1 --outdir results/gcv/  # MutPPI+
```

DDMutPPI is not benchmarked at all (excluded outright: an 87% job-timeout rate on its
public API made a complete scoring run unattainable), not
merely dropped from these commands.

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

Output: `results/biclass_gcv/roc_sahni_fragoza_biclass_with_variance.png` → **S-biclass**

## VarChAMP Blind Test (Fig 4, S2)

VarChAMP data is unpublished IGVF consortium data — cross-reference [data.igvf.org](https://data.igvf.org).

Train on `sahni_fragoza_mapped090826`, predict on all of `varchamp_all_mapped090826` — the
two canonical GCV datasets, nothing else. This replaced a retired, separately-built table
(`datasets/sfvcfp_rows.csv.gz`, via a `vcfp_common.py` helper archived on 2026-09-07) that
mixed pre-090826 sources; see `src/evaluation/run_varchamp_blind_test.py`'s module docstring.
DDMutPPI is excluded outright (not evaluated at all: 87% job-timeout rate on its public API).

```bash
conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method mutpredppi
conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method esignet
conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method mint --predictor seq_diff
conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method mint --predictor site_diff
conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method pplm --predictor seq_diff
conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method pplm --predictor site_diff
conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method swing
conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method saambe3d
conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method mutppi
conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method mutppiplus

# MutPred2 (parse-only; no model to train)
conda run -n ppi python src/analysis/import_mutpred2_varchamp_scores.py --csv /path/to/mutpred2_output.csv

# Generate Fig 4 + S2
conda run -n ppi python src/analysis/varchamp_blind_test.py
```

`--method swing --test-pretrain` (the leaky Doc2Vec-on-everything variant) is not yet wired
up for this single train/test split — it raises `NotImplementedError` with an explanation;
use the default blind-test mode. SAAMBE-3D/MutPPI/MutPPI+ are pretrained on SKEMPI, not
retrained here, and are classed by SKEMPI training-protein overlap
(`utils.gcv_common.skempi_test_class`), not Sahni+Fragoza overlap — same rule as their GCV
stratification.

## Variant Database Inference (Fig 5–7)

ProtT5 embeddings must be precomputed before inference (large, resume-safe):

```bash
nohup conda run -n ppi python src/variant_db_inference/precompute_prott5.py \
    --fasta /path/to/gnomad_wt_and_vt.fasta --out prott5_embeddings.h5 \
    --device cuda:0 > precompute.log 2>&1 &

conda run -n ppi python src/variant_db_inference/run_variant_db_inference.py \
    --dataset gnomad --device cuda:0
```

Rows come from `datasets/variant_dbs/{db}_rows.csv.gz` and graphs from
`datasets/variant_dbs/contact_graphs.h5`, so nothing is parsed out of a filename.

**Output schema.** The prediction TSV carries explicit columns:

```
interactor	partner	mutation	score
```

`mutation` is 1-based. The old composite `complex_id` = `{interactor}_{partner}` column is gone —
splitting it on `_` mis-assigned both proteins whenever an accession itself contained the
separator. The resume path still recognises a legacy `complex_id/variant/score` header so an
interrupted older run can be continued, but new runs never write it. (The standalone
`src/inference/` pipeline is the one place that still emits `complex_id`; see
[`docs/INFERENCE.md`](INFERENCE.md).)

HGMD and COSMIC require licensed access. HGMD is excluded from all distributed files. COSMIC
columns are opt-in when the master CSV is assembled:
`src/analysis/make_master_variant_db_csv.py --include-cosmic`.

### Variant-database source mapping

The per-database mapping steps that produce the annotation pickles the tables are built from.
All of these take licensed or bulk downloads as required arguments — run each with `--help` for
the full list, since the inputs differ per database:

| Script | Required inputs |
|---|---|
| `src/data_processing/variant_databases/map_clinvar.py` | `--stage {variants,interactors} --output-dir` (plus `--variant-summary`, `--hgnc`) |
| `src/data_processing/variant_databases/get_cosmic_annotations.py` | `--cmc-file --gene-symbol-to-uniprot` |
| `src/data_processing/variant_databases/map_tulika_autism.py` | `--variant-dir --biogrid-dir` (ASD/NDD) |
| `src/data_processing/variant_databases/map_cosmic.py` | `--cmc-file --biogrid-dir --output-dir` (licensed) |
| `src/data_processing/variant_databases/map_hgmd.py` | `--hgmd-file --hgmd-dm-wts --hgmd-dm-vts --refseq-to-uniprot --biogrid-dir --output-dir` (licensed) |

### Annotation caches

Two caches under `datasets/annotations/` have explicit rebuilders rather than being opaque
Zenodo blobs:

```bash
# plddt_cache.pkl — per-residue pLDDT from AlphaFold DB MONOMER models
# (not the AF3 complexes: their chains are trimmed to the assayed constructs).
# Consumer: src/analysis/plddt_stratification.py
conda run -n ppi python src/analysis/build_plddt_cache.py --compare-to datasets/annotations/plddt_cache.pkl

# confidence_scores.pkl — {complex_key: {'iptm','ptm'}} from AF3 *_summary_confidences.json.
# Consumer: src/analysis/roc_plots.py
conda run -n ppi python src/analysis/build_confidence_cache.py --json-dir <af3_output_dir> --recursive
```

Both default to comparing against the shipped file and require `--force` to overwrite anything
under `datasets/annotations/`.

## Variant Database Classification and Chart Generation

```bash
conda run -n ppi python src/analysis/classify_variant_dbs.py \
    --output-dir results/variant_dbs_all_data

conda run -n ppi python src/analysis/variant_db_charts.py \
    --data-dir results/variant_dbs_all_data \
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

Output: `results/cosmic_stat_test/cosmic_onco_tsg_qn_vs_edgetic.tex` → **S-cosmic-stat**

### Protein class enrichment (S-protclass)

```bash
conda run -n ppi python src/analysis/protein_class_enrichment.py
```

Output: `results/protein_class/pathogenic_by_class.png` → **S-protclass**

### Stability vs. interaction per-variant scatter (S-stability, 6 panels)

```bash
conda run -n ppi python src/analysis/stability_interaction_scatter.py --cosmic-min-recurrence 32
```

Output: `results/stability_interaction/scatter_per_variant_kde.png` → **S-stability**
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

Output in `results/robustness/`: each script's own `*_auroc_by_class.png` +
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

Output: `results/gcv/roc_plots_with_variance/ablation_bar_sahni_fragoza_with_variance.png` → **S-abl**
