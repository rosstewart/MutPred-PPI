# Data deposit

The repository holds code only. Everything the analyses read — datasets, structures,
contact graphs, model weights and precomputed predictions — is deposited on Zenodo.

| Part | DOI | Size | What it gives you |
|---|---|---|---|
| Analysis layer | *(pending)* | ~1.8 GB | Every unrestricted figure and table, **without a GPU** |
| AlphaFold 3 structures | [10.5281/zenodo.18701748](https://doi.org/10.5281/zenodo.18701748) | ~12 GB | Rebuild the contact graphs from structures |
| Models + training data | [10.5281/zenodo.17645488](https://doi.org/10.5281/zenodo.17645488) | ~30 MB | Earlier deposit; superseded by the analysis layer |

Start with the analysis layer. You only need the structures if you want to rebuild the
contact-graph store yourself.

```bash
# from the repository root
mkdir -p datasets results
tar xf mutpred-ppi-analysis.tar    -C .          # Part 1
tar xf af3_structures.tar          -C datasets/  # Part 2, training/eval complexes
tar xf af3_structures_variant_dbs.tar -C datasets/
sha256sum -c MANIFEST.sha256
```

Then confirm the layout with `python -c "from paths import describe; describe()"`, and see
[`SETUP.md`](SETUP.md) for what each directory is.

---

## Part 1 — Analysis layer (~1.8 GB)

| Contents | Size | Unlocks |
|---|---|---|
| `datasets/training_eval/*_rows.csv.gz`, `*_splits.csv.gz`, `sequences.csv.gz`, `aliases.csv` | 6.7 MB | all cross-validation |
| `datasets/cv_reference/` (310 files) | 135 MB | **required** — see below |
| `datasets/training_eval/contact_graphs.h5` (100,739 graphs) | 966 MB | all inference and training |
| `datasets/af3_structures_canonical/manifest.csv` | 17 MB | maps structures to sequences |
| `datasets/variant_dbs/{clinvar,gnomad,neurodev,asd}_rows.csv.gz`, `aliases.csv` | 53 MB | variant-repository inference |
| `datasets/annotations/` | 269 MB | allele frequencies, pLDDT caches, ClinGen modes of inheritance |
| `datasets/megascale_rows.csv.gz`, `mega_splits.pkl` | 4.8 MB | stability pretraining |
| `datasets/mutpred2_inputs/` | 0.7 MB | the MutPred2 comparison |
| `weights/` (8 checkpoints + 2 scalers) | 13 MB | inference, blind test, ablations, demonstration tier |
| `results/gcv/*_detailed_results.pkl`, `*_aucs.npy` | 132 MB | Fig 3, S1, S3, S7 — every ROC curve |
| Variant-repository prediction TSVs (ClinVar, gnomAD, NDD, ASD) | 69 MB | Fig 5, S8 |
| Stability (ΔΔG) prediction TSVs, same four | 64 MB | the stability figures |
| `results/master_variant_db_predictions_unrestricted.csv.gz` | 24 MB | per-variant scores in one table, with a `training_overlap` column flagging variants the model was trained on |
| `results/variant_dbs_all_data/{clinvar,gnomad,neurodev,asd}/` | 27 MB | classified strata behind Fig 5 |
| Robustness, protein-class, stability and bi-class summary tables | 4.2 MB | the remaining supplements |

### `cv_reference/` is the one thing you cannot regenerate

Everything else in Part 1 is derivable given enough compute. The frozen fold assignments in
`datasets/cv_reference/` are not: the pooled datasets cannot reproduce their splits without
them. Keep this directory.

### The GCV pickles are the canonical source for every curve

`results/gcv/*_detailed_results.pkl` store the raw `preds` and `labels` arrays per seed (30),
fold (10) and test class (C1/C2/C3). Every ROC and PR curve in the paper is computed from
them. If you prefer CSVs to pickles:

```bash
python src/analysis/export_reconstruction_tables.py --figure all
```

That writes flat, documented tables to `datasets/reconstruction_tables/`. They are not
deposited because they contain nothing the pickles do not.

### The contact-graph store has three names

`datasets/training_eval/contact_graphs.h5`, `datasets/variant_dbs/contact_graphs.h5` and
`datasets/af3_structures_canonical/contact_graphs_v4.h5` are **one file**, hard-linked. One
copy is deposited; create the other two as links or copies.

---

## Part 2 — AlphaFold 3 structures (~12 GB)

Two archives of AlphaFold Server predictions, plus the canonical tree they were merged into:

| File | Contents |
|---|---|
| `af3_structures.tar` (600 MB) | 4,302 training/evaluation complexes |
| `af3_structures_variant_dbs.tar` (6.6 GB) | 24,121 ClinVar / gnomAD / NDD / ASD complexes |
| `af3_structures_canonical_in_house.tar` (5.2 GB) | 24,716 deduplicated, content-named mmCIFs |
| `manifest.csv` (17 MB) | **all 100,739** canonical structures, including those not deposited |

### Only structures we generated are deposited

The canonical tree merges our AlphaFold 3 predictions with predicted complexes downloaded
from **ProtVar** (EMBL-EBI). Those are not ours to redistribute, and they are the majority:

| Provenance | Structures | Size | Deposited |
|---|---|---|---|
| In-house AlphaFold 3 | 24,716 (24.5%) | 5.22 GB | yes |
| ProtVar (EMBL-EBI) | 76,023 (75.5%) | 7.40 GB | no — download it yourself |

The `provenance` column of `manifest.csv` records which is which for every structure, so you
can see exactly what is missing and fetch it. To obtain the ProtVar structures, download
`2024.05.28_interface_models_high_confidence.tar` (~57 GB) from ProtVar and point
`external/protvar_pdb` at it; see [`SETUP.md`](SETUP.md).

**This costs you nothing for figures.** The deposited `contact_graphs.h5` already contains all
100,739 graphs. You need the ProtVar structures only if you want to rebuild the graph store
from structures with `src/data_processing/rebuild_graphs_from_structures.py`.

Both archives are AlphaFold Server output and are subject to its
[output terms of use](https://alphafoldserver.com/output-terms); cite Abramson et al. 2024.

---

## What is not deposited, and why

### Licence-restricted

COSMIC and HGMD require a licence, so nothing derived from them is included: their row
tables, prediction and stability TSVs, classified strata, and the COSMIC annotation
dictionaries under `datasets/annotations_licensed/`. With your own licence, regenerate them
with `src/data_processing/variant_databases/map_cosmic.py` and `map_hgmd.py` (COSMIC v101,
HGMD Professional 2025).

Every consumer degrades gracefully when they are absent: figures skip the affected panels
with a warning rather than failing.

### Unpublished

The VarChAMP measurements were unpublished IGVF consortium data at the time of release.
Excluded: the VarChAMP row and split tables, the blind-test arrays under
the blind-test results tree, the matching `cv_reference` entries, and the
eSIG-Net supplement caches. **Fig 4, Fig S2 and Table 1 are therefore not independently
reproducible.** See the reproducibility table in the [README](../README.md).

Because the all-data model is trained partly on VarChAMP, it cannot be retrained without it —
so the deposit also ships a **Sahni+Fragoza demonstration model** (`weights/sahni_fragoza/`).
It lets you run the whole variant-repository pipeline end to end. Its scores are *not* the
published numbers; it writes to a separate results tree and stamps every figure it produces.

### Too large to be useful as bytes (~1.2 TB)

Protein language model caches: ProtT5 embeddings and 2-hop subgraphs per variant database
(687 GB), ESM2/MINT/PPLM embeddings for the training sets (396 GB), and the merged blind-test
caches (139 GB). Regenerate what you need:

```bash
DB="$MUTPRED_DATA_ROOT/clinvar"
python src/variant_db_inference/precompute_prott5.py \
    --fasta "$DB/clinvar_interaction_loss_wt_and_vt.fasta" \
    --out   "$DB/prott5_embeddings.h5"
python src/variant_db_inference/compress_to_subgraphs.py --dataset clinvar
python src/data_processing/precompute_prott5_datasets.py --dataset sahni_fragoza
```

See [REPRODUCING_ANALYSES.md](REPRODUCING_ANALYSES.md#variant-repository-inference-fig-5-s8-s9)
for what `$MUTPRED_DATA_ROOT` resolves to.

Check what is present and current at any point with:

```bash
python src/variant_db_inference/audit_caches.py
```

---

## Verifying a download

```bash
sha256sum -c MANIFEST.sha256
python -m pytest tests/ --run-data      # asserts caches, structures and predictions agree
```
