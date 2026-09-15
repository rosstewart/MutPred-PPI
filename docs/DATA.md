# Data: what to download and where it goes

The repository holds code. Datasets, structures, contact graphs, model weights and
precomputed predictions are deposited on Zenodo.

<!-- DOI: pending -->

## Download and unpack

The deposit is four archives. Each is rooted at the repository, so unpack them from the
repository root and every file lands where the code expects it.

```bash
git clone https://github.com/rosstewart/mutpred-ppi.git
cd mutpred-ppi

tar xzf mutpred-ppi-datasets.tar.gz -C .     # 1.4 GB  tables, annotations, graph store
tar xzf mutpred-ppi-results.tar.gz  -C .     # 324 MB  cross-validation and predictions
tar xzf mutpred-ppi-weights.tar.gz  -C .     #  13 MB  model checkpoints

sha256sum -c --ignore-missing MANIFEST.sha256
```

Those three reproduce every figure. The fourth is optional:

```bash
tar xzf mutpred-ppi-structures.tar.gz -C .   # 5.2 GB  AlphaFold 3 structures
```

You need it only to rebuild the contact-graph store from structures. The store itself is
already in the datasets archive.

Check the layout resolved correctly:

```bash
python -c "from paths import describe; describe()"
```

## What is in the deposit

| Archive | Contents | Size | Used for |
|---|---|---|---|
| datasets | `training_eval/*_rows.csv.gz`, `*_splits.csv.gz`, `sequences.csv.gz`, `aliases.csv` | 6.7 MB | all cross-validation |
| datasets | `cv_reference/` | 135 MB | frozen fold assignments for all 30 seeds |
| datasets | `contact_graphs.h5` | 966 MB | every inference and training run |
| datasets | `af3_structures_canonical/manifest.csv` | 17 MB | maps structures to sequences |
| structures | `af3_structures_canonical/` (in-house) | 5.2 GB | rebuilding the graph store |
| datasets | `variant_dbs/{clinvar,gnomad,neurodev,asd}_rows.csv.gz` | 53 MB | variant-repository inference |
| datasets | `annotations/` | 269 MB | allele frequencies, pLDDT caches, ClinGen modes of inheritance |
| datasets | `megascale_rows.csv.gz`, `mega_splits.pkl` | 4.8 MB | stability pretraining |
| datasets | `mutpred2_inputs/` | 0.7 MB | the MutPred2 comparison |
| weights | `weights/` | 13 MB | inference, blind test, ablations |
| results | `gcv/*_detailed_results.pkl`, `*_aucs.npy` | 132 MB | Fig 3, S1, S2, S4 |
| results | `variant_dbs_all_data/{db}_mutpred_ppi_predictions.tsv` | 69 MB | Fig 5, S7 |
| results | `variant_dbs_stability/{db}_stability_predictions.tsv` | 64 MB | S10 |
| results | `master_variant_db_predictions_unrestricted.csv.gz` | 24 MB | per-variant scores in one table |
| results | `variant_dbs_all_data/{clinvar,gnomad,neurodev,asd}/` | 27 MB | classified strata behind Fig 5 |
| results | robustness, protein-class and bi-class summary tables | 4.2 MB | S5, S8, S9 |

### One contact-graph store

`datasets/contact_graphs.h5` holds all 100,739 graphs. Training/evaluation complexes and
variant-repository complexes share it, because graphs are keyed on the sorted pair of
`sha256(chain_sequence)[:16]`, so a pair appearing in both is stored once. Code resolves it
through `paths.contact_graph_store()`; do not depend on the filename.

### Regenerating rather than downloading

Everything in the deposit can be rebuilt given the inputs and enough compute. Two items are
worth downloading anyway:

- `datasets/cv_reference/` rebuilds with
  `python src/analysis/export_cv_reference.py --dataset all --n-seeds 30`, which needs
  `cd-hit`. The result is deterministic and `tests/test_splits_reproducible.py` asserts it
  reproduces the deposited table, so downloading only saves the clustering run.
- `results/gcv/*_detailed_results.pkl` hold the raw predictions and labels per seed, fold
  and test class. Every ROC and PR curve is computed from them, so having them means the
  figures redraw without re-running cross-validation. For CSVs instead of pickles:
  `python src/analysis/export_reconstruction_tables.py --figure all`.

## Structures

The canonical tree merges AlphaFold 3 predictions made for this study with predicted
complexes downloaded from ProtVar (EMBL-EBI). Only the in-house structures are deposited:

| Provenance | Structures | Size | Deposited |
|---|---|---|---|
| In-house AlphaFold 3 | 24,716 (24.5%) | 5.2 GB | yes |
| ProtVar (EMBL-EBI) | 76,023 (75.5%) | 7.4 GB | no |

The `provenance` column of `manifest.csv` records which is which for every structure. To
obtain the ProtVar half (~57 GB), download it from the EMBL-EBI FTP site and link the
extracted `pdb/` directory into `external/`:

```bash
curl -O https://ftp.ebi.ac.uk/pub/databases/ProtVar/predictions/interfaces/2024.05.28_interface_models_high_confidence.tar
tar xf 2024.05.28_interface_models_high_confidence.tar
ln -s "$PWD/pdb" external/protvar_pdb
```

You need it only to rebuild the graph store from structures. The deposited
`datasets/contact_graphs.h5` already contains all 100,739 graphs, so every figure reproduces
without a single ProtVar file.

Both sources are AlphaFold output and subject to the
[AlphaFold Server output terms](https://alphafoldserver.com/output-terms). Cite Abramson et
al. 2024.

## What is not deposited

### Licence-restricted

COSMIC and HGMD require a licence, so nothing derived from them is included: their row
tables, prediction and stability TSVs, classified strata, and the annotation dictionaries
under `datasets/annotations_licensed/`. Regenerate with
`src/data_processing/variant_databases/map_cosmic.py` and `map_hgmd.py` (COSMIC v101, HGMD
Professional 2025). Analyses skip the affected panels with a warning and draw the rest.

### Unpublished

VarChAMP measurements were unpublished IGVF consortium data at the time of writing, and will
be cross-linked from [data.igvf.org](https://data.igvf.org) on release. Excluded: the
VarChAMP row and split tables, the matching `cv_reference` entries and GCV result pickles,
the blind-test arrays, and the eSIG-Net supplement caches. Fig 4, S3, S6 and Table 1 cannot
be reproduced until then.

The all-data model is trained partly on VarChAMP, so it cannot be retrained without it.
Scoring with it needs no such thing: `weights/MutPred-PPI.pt` is in the deposit, and
`notebooks/reproduce_all_figures.py` uses it to reproduce the published variant-repository
numbers whether or not the measurements are present.

The deposit also includes a Sahni+Fragoza model in `weights/sahni_fragoza/`, for running the
variant-repository pipeline end to end from a model you can retrain yourself. Its scores are
not the published ones: it writes to a separate results tree and marks every figure it
produces.

### Too large to be useful as bytes

Protein language model caches: ProtT5 embeddings and 2-hop subgraphs per variant repository
(687 GB), ESM2/MINT/PPLM embeddings for the training sets (396 GB), and the merged
blind-test caches (139 GB). Regenerate what you need:

```bash
DB="$MUTPRED_DATA_ROOT/clinvar"
python src/variant_db_inference/precompute_prott5.py \
    --fasta "$DB/clinvar_interaction_loss_wt_and_vt.fasta" \
    --out   "$DB/prott5_embeddings.h5"
python src/variant_db_inference/compress_to_subgraphs.py --dataset clinvar
python src/data_processing/precompute_prott5_datasets.py --dataset sahni_fragoza
```

Check what is present and current:

```bash
python src/variant_db_inference/audit_caches.py
```

## Environment variables

All optional; the defaults resolve for a standard checkout.

| Variable | Overrides | Default |
|---|---|---|
| `MUTPRED_DATA_ROOT` | root of the external data tree | the repository's parent directory |
| `MUTPRED_CV_DIR` | CV reference artifacts | `datasets/cv_reference/` if present |
| `MUTPRED_CACHE_DIR` | large regenerable caches | `mutpred_ppi_data/` beside the repository |
| `MUTPRED_CDHIT` | the `cd-hit` binary | first on `PATH` |

## Verifying a download

```bash
sha256sum -c --ignore-missing MANIFEST.sha256
python -m pytest tests/ --run-data
```
