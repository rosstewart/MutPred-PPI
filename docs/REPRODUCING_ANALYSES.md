# Reproducing Every Figure and Table

Cross-validation benchmarking, the VarChAMP blind test, variant-repository
inference/classification/charts, and supplementary analyses. For training the model from
scratch, see [`docs/TRAINING.md`](TRAINING.md). `datasets/` and `results/` are gitignored and
come from the Zenodo deposit -- see [`docs/ZENODO.md`](ZENODO.md).

Every ROC and PR curve can be recomputed without rerunning anything from
`results/gcv/*_detailed_results.pkl`, which store the raw predictions and labels per seed,
fold and test class. For flat CSVs instead of pickles:
`python src/analysis/export_reconstruction_tables.py --figure all`.

## Figure index

Figure files under `figures/` are symlinks into `results/`, so a figure is "present" only when
its target exists; `ls -l figures/` shows the targets.

| Label | Output | Produced by | Needs |
|---|---|---|---|
| Table 1 | `training_data_table.tex` | `analysis/generate_training_table.py` | mapped source CSVs | 
| Fig 3 | `roc_sahni_fragoza_with_variance.png` | `analysis/run_roc_comparison.py` | GCV, `sahni_fragoza`, all methods |
| Fig 4 | `roc_varchamp_blind_test.png` | `analysis/blind_test_figures.py` | VarChAMP blind test |
| Fig 5 | `enrichment_bootstrap_sufficient_partners.png` | `analysis/variant_db_charts.py --edgotype-bootstrap` | all six repositories classified |
| Table S1 | `variant_db_stats_table.tex` | `analysis/extract_variant_db_stats.py` | all six repositories |
| S1 | `roc_sahni_with_variance.png` | `analysis/run_roc_comparison.py` | GCV, `sahni_only` |
| S2 | `roc_varchamp_blind_test_training_comparison.png` | `analysis/blind_test_figures.py` | blind test, both training sets |
| S3 | `ablation_bar_sahni_fragoza_with_variance.png` | `analysis/run_roc_ablation.py` | GCV ablations, `sahni_fragoza` |
| S7 | `roc_sahni_fragoza_varchamp_all_with_variance.png` | `analysis/run_roc_comparison.py` | GCV, `sahni_fragoza_varchamp_all` |
| S8 | `enrichment_bootstrap_sufficient_partners_k3.png` | `analysis/variant_db_charts.py --controlled-bootstrap --k3-only` | all six repositories classified |
| S9 | `threshold_sensitivity.png` | `analysis/threshold_sensitivity.py` | all six repositories classified |
| S-biclass | `roc_sahni_fragoza_biclass_with_variance.png` | `analysis/biclass_sf_gcv.py` | GCV, `sahni_fragoza` |
| S-protclass | `pathogenic_by_class.png` | `analysis/protein_class_enrichment.py` | ClinVar + gnomAD predictions, GO annotations |
| S-robustness | `combined_robustness_by_class.png` | `analysis/combined_robustness_figure.py` | interface / pLDDT / protein-class stratifications |
| S-stability | `stability_interaction_scatter.png` | `analysis/stability_interaction_scatter.py` | interaction + stability predictions |

These labels are the ones `notebooks/reproduce_all_figures.py` prints, so the notebook's
output and this table agree.

Figures 1 and 2 (pipeline and architecture schematics) and the CDC42 example panel are
hand-drawn or rendered in ChimeraX; nothing in this repository regenerates them.

## Excluding variants the model was trained on

Every figure that measures **enrichment over a variant repository** — Fig 5, S8, S9,
S-protclass and the stability figure — excludes variants the model was trained on, and so
does **Table S1**, which describes those same groups. A
variant in the training set has a fitted score rather than a predicted one, and the
disease sets overlap training far more than the gnomAD background does, so leaving them
in inflates precisely the contrast being measured.

Overlap is matched on **(interactor, variant), ignoring the partner**: a variant seen
against any partner during training has had its mutation representation fitted, and the
model reads the same mutated-site features whichever partner it is scored against.
Matching the full triple instead would keep those rows.

The definition lives in `src/analysis/training_overlap.py` and is applied in one place
per input path — `classify_variant_dbs.py` for the stratum tables that Fig 5/S8/S9 and
the stability figure read, and `protein_class_enrichment.py` and
`extract_variant_db_stats.py`, which read the raw prediction TSVs. Table S1's variant
counts therefore match the sample sizes printed on the figures; `pytest tests/ --run-data`
asserts that they do. Pass `--keep-training-overlap` to `classify_variant_dbs.py` to
reproduce the unfiltered view; those are not the published numbers.

The deposited master tables are **annotated rather than filtered**: a `training_overlap`
boolean column marks the affected rows, so either view can be reproduced from the
deposit without needing the training set, which is not redistributable.

## Verification utilities

Standalone checks over the built artifacts. None writes a figure or table; they exist so a
rebuilt structure set, graph store or cross-validation run can be shown to satisfy the
invariants the analysis code assumes.

| Command | Checks |
|---|---|
| `python src/variant_db_inference/audit_caches.py` | every variant has an embedding, a contact graph and a prediction, or a stated reason why not |
| `python src/verification/check_structure_superset.py --new <dir> --live <dir>` | a rebuilt structure set still covers every pair the live one did |
| `python src/verification/verify_structure_chains.py --dataset sahni_fragoza` | the chain handed to each structure-based comparator really holds the mutated protein |
| `python src/verification/check_gcv_consistency.py` | per-seed AUCs of a run in progress sit within the spread of the earlier seeds |

`pytest tests/ --run-data` asserts the same properties non-interactively.

**One command runs everything below in order:** `notebooks/reproduce_all_figures.py`
(jupytext percent format -- `jupytext --to notebook` for a `.ipynb`, or run it directly as a
script) caches every step, generates missing embeddings on first use, displays each figure
inline, and writes to the exact paths this document describes.

**It ships with `QUICK = True`** (line 48): a fast smoke test that uses 1 cross-validation
seed instead of 30 and subsampled variant databases, writing to `results_quick/` so it never
touches the canonical `results/` tree. Those are **not** the published numbers -- set
`QUICK = False` to reproduce them, which takes days rather than hours.

## The canonical data layer

Everything reads two tables per dataset from `datasets/training_eval/`, built by
`src/data_processing/training_sets/prepare_gcv_tables.py`
(see [`DATA_PREPARATION.md`](DATA_PREPARATION.md) for the chain that produces the
mapping itself):

```
<dataset>_rows.csv.gz    row_index, interactor, partner, mutation, position, wt_aa, mut_aa,
                         perturbed, dataset, dataset_tier, fragoza_source, source_row_id, cluster
<dataset>_splits.csv.gz  seed, row_index, test_fold, test_class
sequences.csv.gz         accession, sequence
```

Guarantees, asserted at build time: every `mutation` is 1-based and validated against its
sequence, accessions are UniProt (isoform suffix only where the sequence differs from canonical),
no duplicate `(interactor, partner, mutation)`, no null labels, `row_index` contiguous and never
renumbered. **The pipeline is 1-based end to end** -- mutation strings, embedding-cache keys and
the tables all agree, so nothing converts between conventions. Node indices (`mutation_idx`) stay
0-based because they address a graph row, not a residue in a mutation string.

There is no `--data-root`: the tables locate themselves.

The five live datasets (short names are the canonical `--dataset` values; the full
full stamped form (`sahni_fragoza_mapped090826`, and so on) is also accepted):

| `--dataset` | rows |
|---|---|
| `sahni_fragoza_varchamp_all` | 23,254 |
| `varchamp_all` | 17,317 |
| `sahni_fragoza` | 6,212 |
| `fragoza_only` | 4,726 |
| `sahni_only` | 1,591 |

Rebuild them with:

```bash
conda run -n ppi python src/data_processing/training_sets/prepare_gcv_tables.py \
    --dataset all --n-seeds 30                                # rows + splits + sequences
```

### Contact graphs

Graphs are not files on disk any more. Each tier has one HDF5 `ContactGraphStore`
([`src/contact_graphs.py`](../src/contact_graphs.py)):

| Store | Used by |
|---|---|
| `datasets/training_eval/contact_graphs.h5` | GCV, blind test, final training |
| `datasets/variant_dbs/contact_graphs.h5` | variant-database inference |

Both names refer to the same file. Keying, accessors and orientation are documented once, in
[`docs/INFERENCE.md`](INFERENCE.md#the-contact-graph-store); the short version is that graphs are looked up by sequence, never by filename.

Rebuild a store from the canonical structures:

```bash
conda run -n ppi python src/data_processing/rebuild_graphs_from_structures.py \
    --structures datasets/af3_structures_canonical \
    --out datasets/training_eval/contact_graphs.h5 \
    [--compare-to datasets/training_eval/contact_graphs.h5] [--n-jobs 16]
```

### AF3 structures

`datasets/af3_structures_canonical/` holds one gzipped mmCIF per pair, 100,739 in total —
training/evaluation complexes and variant-repository complexes were merged into this single
tree:

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

### Structure sources for the variant databases

Variant-database complexes come from two places, and both are needed:

1. **Folded in-house** -- the per-database `af3_models*/` trees (neurodev, gnomad,
   clinvar, asd, hgmd, cosmic).
2. **ProtVar precomputed AlphaFold3 interfaces** -- 126,118 high-confidence
   complexes, downloaded separately:

   ```bash
   curl -O https://ftp.ebi.ac.uk/pub/databases/ProtVar/predictions/interfaces/2024.05.28_interface_models_high_confidence.tar
   tar xf 2024.05.28_interface_models_high_confidence.tar
   ln -s $PWD/pdb external/protvar_pdb
   ```

ClinVar, COSMIC, gnomAD and HGMD draw most of their partner structures from ProtVar:
**76,023 of the 100,739 canonical structures (75.5%) come from it**, against 24,716 folded
in-house. The `provenance` column of `datasets/af3_structures_canonical/manifest.csv` records
which is which for every structure. neurodev and asd need none of it.

Omitting ProtVar does not affect the figures if you use the deposited contact-graph store,
which already contains all 100,739 graphs — it matters only when rebuilding the store from
structures.

`canonicalize_structures.py` resolves every chain by SEQUENCE against the
canonical tables, so a structure whose chains are unknown is skipped -- the whole
ProtVar directory can be passed as a source without pre-filtering, and PDB input
is auto-detected and normalised to gzipped mmCIF alongside the mmCIF sources.

### Variant-database tables

One self-contained table per database, replacing the scattered annotation pickles at the point of
use:

```
datasets/variant_dbs/{clinvar,cosmic,gnomad,hgmd,neurodev}_rows.csv.gz
```

Shared columns: `interactor`, `partner`, `mutation` (**1-based**), `pair_key` (the contact-graph
content address — sequences are not inlined), `clingen_moi`, `in_embedding_store`. Per-DB
annotations follow: `clinical_significance`/`allele_frequency` (clinvar), `allele_frequency`
(gnomad), `recurrence`/`tumor_sites`/`onco_tsg` (cosmic), `neurodev_label` (neurodev); hgmd carries
the shared columns only. Row counts: clinvar 949,065, cosmic 1,447,917, gnomad 10,529,577,
hgmd 56,266, neurodev 19,148.

```bash
conda run -n ppi python src/variant_db_inference/build_variant_db_tables.py --db all
```

The pickles remain the source; these are a derived view. Zero-based conversion (FASTA headers,
ProtT5 keys, subgraph H5 variant keys) happens only through
`src/variant_db_inference/variant_rows.py::to_zero_based`, never inline.

## Comparison methods: clone each upstream repository

Every method we benchmark against is run from **its own upstream checkout**, not
from a copy inside this repository. Clone them into `external_methods/`
(gitignored), one directory per method:

```bash
mkdir -p external_methods
# SAAMBE-3D: download and unpack the archive from
#   http://compbio.clemson.edu/SAAMBE-3D/   (a project page, not a git remote)
# into external_methods/saambe3d/
git clone https://github.com/VarunUllanat/mint       external_methods/mint
git clone https://github.com/ChengfeiYan/PPLM        external_methods/PPLM
git clone https://github.com/Liu-Jing/eSIG-Net       external_methods/esignet
git clone https://github.com/Wang-Lin-boop/MutPPI    external_methods/MutPPI

# Commits used for the published results:
#   saambe3d  182a2746c8adb7434f1ac28c111e6b3f031c59e7
#   mint      12946127faeba20698e83bfc040913ebc993a3c7
#   PPLM      c2a4d5d1f9a433dddc65b5b11908ba3ea1970a51
#   esignet   cd36a4a058125910d3ff0ef9b5cc717960fdbc78
#   MutPPI    7a5c6f764818a7346c1d52977af607e81f2eaf10
```

Set `MUTPRED_PPI_METHODS_DIR` to put them elsewhere. Any script that needs a
method it cannot find fails immediately with the exact `git clone` command
rather than an `ImportError` from inside a `sys.path` insert.

| Method | Directory | Additional files it needs |
|---|---|---|
| SAAMBE-3D | `saambe3d/` | Ships its own SKEMPI-trained `*_v01.model` boosters. Requires `prody` (not in the `ppi` env). It therefore needs its own interpreter: set `SAAMBE3D_PYTHON=/path/to/python` to an environment that has `prody` installed. (`saambe3d_cv.py` also looks for a `py311_saambe3d` conda env next to your Miniconda install, which is a convenience, not a requirement.) |
| MINT | `mint/` | `mint.ckpt` and `esm2_t33_650M_UR50D.json` from the MINT release page. |
| PPLM | `PPLM/` | `weights/pplm_t33_650M.pt` from the PPLM release page. |
| eSIG-Net | `esignet/` | Uses `backbones/sdnn/sdnn_model.py` from the checkout. Publishes no feature-extraction code, so ours is reconstructed and validated -- see `src/evaluation/predictors/validate_esignet_features.py`. |
| MutPPI / MutPPI+ | `MutPPI/` | Per-fold checkpoints under `output/checkpoint/`; `mutppi_cv.py` prints the training command if they are absent. |
| SWING | *(in-repo)* | `src/evaluation/swing_common.py`; no external checkout. |
| DDMut-PPI | *(not benchmarked)* | Excluded outright: an 87% job-timeout rate on its public API made a complete scoring run unattainable. |

Only SWING is implemented in this repository. Nothing under `src/` is
third-party source.

Which figure comes from which command is tabulated in the
[figure index](#figure-index) at the top of this document.

## Running a whole suite

The per-method commands below are the ground truth for what each script does, and
are the right thing to run when reproducing one number. To run a *suite* to
completion, use the job runner rather than looping over them by hand:

```bash
conda run -n ppi python src/run_benchmarks.py --status                  # what is done
conda run -n ppi python src/run_benchmarks.py --suite gcv --gpus 0,1,2,3
conda run -n ppi python src/run_benchmarks.py --suite all --dry-run     # print, run nothing
```

It skips jobs whose outputs are already complete, so an interrupted run is
resumed by re-issuing the same command. Jobs are split into a CPU pool
(`--jobs`, default 12) and a GPU pool (one job per id in `--gpus`).

`--threads` caps BLAS/OpenMP threads per job and defaults to 1. The libraries otherwise start
one thread per core, and on a many-core host a single small MLP fit spends most of its wall
time in OpenMP barriers — raising the thread count can make a fit several times *slower*.

Thread count is also not numerically neutral: OpenBLAS partitions reductions by team size,
which moves cross-validation AUCs in the 4th decimal. Use one value for a whole suite rather
than mixing.

## Grouped Cross-Validation (Fig 3, S1)

Every trained method runs through one shared runner
([`src/utils/gcv_common.py`](../src/utils/gcv_common.py)`::run_gcv`, which also holds
`DATASET_CONFIGS`, `load_data` and the split loading); only the training loop differs per method.
MutPred-PPI's loop is `src/training/train_fold.py::train_fold`, imported by both
`mutpred_ppi_gcv.py` and `train_final_model.py` — it has no CLI of its own.

```bash
DS=sahni_fragoza_varchamp_all

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
# Pretrained on SKEMPI, not retrained here (see the stratification note below).
conda run -n ppi python src/evaluation/saambe3d_cv.py --dataset $DS --outdir results/gcv/
conda run -n ppi python src/evaluation/mutppi_cv.py   --dataset $DS --model 0 --outdir results/gcv/  # MutPPI
conda run -n ppi python src/evaluation/mutppi_cv.py   --dataset $DS --model 1 --outdir results/gcv/  # MutPPI+
```

DDMut-PPI is not benchmarked at all; see the comparator table above. It was excluded
outright, not merely dropped from these commands.

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

Train on `sahni_fragoza`, predict on all of `varchamp_all` — the two canonical datasets,
nothing else. The trainable methods are retrained here rather than loading a checkpoint, so
the blind test always reflects the current tables.
DDMut-PPI is excluded; see the comparator table above.

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
conda run -n ppi python src/analysis/blind_test_figures.py
```

`--method swing --test-pretrain` fits Doc2Vec over the merged train+test corpus, which is
SWING's own default configuration; the plain `--method swing` arm fits it on the training
rows only. Both are reported. SAAMBE-3D/MutPPI/MutPPI+ are pretrained on SKEMPI, not
retrained here, and are classed by SKEMPI training-protein overlap
(`utils.gcv_common.skempi_test_class`), not Sahni+Fragoza overlap — same rule as their GCV
stratification.

## Variant Repository Inference (Fig 5, S8, S9)

Scored with the single all-data model (`weights/MutPred-PPI.pt`);
`assert_all_data_model` refuses to start with anything else, so there is only ever one
published results tree.

**Without the unpublished VarChAMP data** that model cannot be trained. Pass
`--model-tier sahni_fragoza` to use the demonstration model from the Zenodo weights bundle
instead. It writes to `results/variant_dbs_sahni_fragoza/` and to
`{db}_mutpred_ppi_predictions_sahni_fragoza.tsv`, never over the published tree, and the
figure scripts take `--demo-tier` to stamp their output accordingly. Its scores are **not**
the published numbers. `notebooks/reproduce_all_figures.py` selects this automatically when
the VarChAMP table is missing.

ΔΔG predictions are unaffected: `run_stability_inference.py` uses the MegaScale-pretrained
stability model, which never saw VarChAMP.

The script auto-detects a compact subgraph H5 (preferred, ~120-165 GB) or falls
back to a full ProtT5 embeddings H5. If neither is present, build the subgraph H5:

Per-database caches live under `$MUTPRED_DATA_ROOT/<db>/`, which defaults to the directory
*above* the repository (`python -c "from paths import describe; describe()"` prints the
resolved value). Substitute your own path for `$DB` below.

```bash
DB="$MUTPRED_DATA_ROOT/gnomad"          # or wherever you keep the per-database caches

# Step 1 (once): precompute per-protein embeddings (~100-170 GB):
nohup conda run -n ppi python src/variant_db_inference/precompute_prott5.py \
    --fasta "$DB/gnomad_interaction_loss_wt_and_vt.fasta" \
    --out   "$DB/prott5_embeddings.h5" \
    --device cuda:0 > precompute_gnomad.log 2>&1 &

# Step 2 (once, optional but recommended): compress to 2-hop subgraphs (~120-165 GB):
conda run -n ppi python src/variant_db_inference/compress_to_subgraphs.py \
    --dataset gnomad

# Step 3: run inference (uses subgraph H5 if present, else full embeddings):
conda run -n ppi python src/variant_db_inference/run_variant_db_inference.py \
    --dataset gnomad --device cuda:0
```

Rows come from `datasets/variant_dbs/{db}_rows.csv.gz` and graphs from
`datasets/variant_dbs/contact_graphs.h5`, so nothing is parsed out of a filename.

**Output schema.** The prediction TSV carries explicit columns:

```
interactor	partner	mutation	score
```

`mutation` is 1-based. Both pipelines — this one and the standalone `src/inference/`
three-step pipeline — write these same four columns. A composite
`complex_id` = `{interactor}_{partner}` column was used previously; splitting it back on
`_` mis-assigned both proteins whenever an accession itself contained the separator, so it
was replaced by explicit columns. The resume path still recognises the old header, so an
interrupted older run can be continued.

HGMD and COSMIC require licensed access. HGMD is excluded from all distributed files. COSMIC
columns are opt-in when the master CSV is assembled:
`src/analysis/make_master_variant_db_csv.py --include-cosmic`.

### Variant-database source mapping

Both this step and the per-database `map_*.py` scripts below are **optional**.
The Zenodo bundle ships what they produce (`datasets/variant_dbs/*_rows.csv.gz`),
and their inputs are licensed (COSMIC, HGMD) or many gigabytes (ClinVar, gnomAD,
BioGRID). `notebooks/reproduce_all_figures.py` gates them behind
`RUN_VARIANT_DB_MAPPING = False`; run them only to rederive the interactome from
source.

Every `map_*.py` below consumes the BioGRID pickles, so that step runs first:

```bash
conda run -n ppi python src/data_processing/variant_databases/get_biogrid_interactors.py \
    --biogrid-tsv biogrid/biogrid_ppi.tsv \
    --uniprot-fasta biogrid/all_uniprot_ids.fasta \
    --output-dir $MUTPRED_DATA_ROOT/biogrid
```

This defines "physical binding evidence only": an edge is kept when BioGRID
records it under one of five experimental systems evidencing a **direct**
contact — Co-crystal Structure, Cross-Linking-MS (XL-MS), Far Western,
Reconstituted Complex, Protein-Peptide. Systems that only establish co-complex
membership (Affinity Capture-MS and similar) are excluded, because an edgotype
is a claim about a specific binding interface. The set is
`get_biogrid_interactors.BINDING_TECHNIQUES`, and every variant-database pair in
`datasets/variant_dbs/{db}_rows.csv.gz` satisfies it.

The per-database mapping steps that produce the annotation pickles the tables are built from.
All of these take licensed or bulk downloads as required arguments — run each with `--help` for
the full list, since the inputs differ per database:

| Script | Required inputs |
|---|---|
| `src/data_processing/variant_databases/get_biogrid_interactors.py` | `--biogrid-tsv --uniprot-fasta --output-dir` (stage 0; all rows below need its output) |
| `src/data_processing/variant_databases/map_clinvar.py` | `--stage {variants,interactors} --output-dir` (plus `--variant-summary`, `--hgnc`) |
| `src/data_processing/variant_databases/get_cosmic_annotations.py` | `--cmc-file --gene-symbol-to-uniprot` |
| `src/data_processing/variant_databases/map_neurodev.py` | `--mode neurodev --neurodev-case --neurodev-control --biogrid-dir --output-dir` (builds the `neurodev` database's case/control labels) |
| `src/data_processing/variant_databases/map_asd_ndd.py` | `--variant-dir --biogrid-dir` (Fu et al. de novo ASD; unlabelled) |
| `src/data_processing/variant_databases/map_cosmic.py` | `--cmc-file --biogrid-dir --output-dir` (licensed) |
| `src/data_processing/variant_databases/map_hgmd.py` | `--hgmd-file --hgmd-dm-wts --hgmd-dm-vts --refseq-to-uniprot --biogrid-dir --output-dir` (licensed) |

### Annotation caches

Two caches under `datasets/annotations/` have explicit rebuilders rather than being opaque
Zenodo blobs:

```bash
# plddt_cache.pkl — per-residue pLDDT from AlphaFold DB MONOMER models
# (not the AF3 complexes: their chains are trimmed to the assayed constructs).
# Consumer: src/analysis/plddt_stratification.py
conda run -n ppi python src/analysis/build_plddt_cache.py --compare-legacy

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
- `enrichment_bootstrap_sufficient_partners_k3.png` → **S8** (same grouping, partner-controlled)

### Gene inheritance-mode (AR/AD) mapping

Required once, before Fig 5/S8/S-stability:

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
conda run -n ppi python src/analysis/run_roc_comparison.py   # Fig 3, S1, S7
conda run -n ppi python src/analysis/run_roc_ablation.py     # S3
```

## Ablation figure (S3)

```bash
conda run -n ppi python src/analysis/run_roc_ablation.py
```

The **"Prior Best" bar is omitted by default.** It is the previously published model rather
than an ablation of the current architecture, and its checkpoint is distributed separately:

> RECOMB model (bioRxiv v2): MutPred-PPI v1.0 —
> <https://github.com/rosstewart/MutPred-PPI/releases/tag/v1.0.0>

Unpack that release into `weights/v1_0/` and pass `--include-prior-best` to draw it:

```bash
conda run -n ppi python src/analysis/run_roc_ablation.py --include-prior-best
```

Every other ablation arm uses `weights/MutPred-PPI_stability_pretrain.pt`, which is in the
Zenodo weights bundle.

Output: `results/gcv/roc_plots_with_variance/ablation_bar_sahni_fragoza_with_variance.png` → **S3**
