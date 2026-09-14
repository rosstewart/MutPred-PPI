# Setup: where the code looks for data

Every path the code uses is resolved through one module, [`src/paths.py`](../src/paths.py).
Nothing under `src/` contains an absolute path, so the repo runs from any
directory and on any machine once the data is in place.

Check what resolves where at any time:

```bash
conda run -n ppi python src/paths.py      # prints every resolved path and whether it exists
```

## Install the package first

`pyproject.toml` makes `src/` the package root, so modules import as top-level packages —
`training.train_fold`, `utils.gcv_common`, `evaluation.mint_cv`, `analysis.roc_plots` — plus the
standalone modules `paths`, `ids`, `model`, `contact_graphs`. Nothing outside
`src/inference/` imports without this:

```bash
pip install -e .
```

## Get the data first

Everything except the inference quickstart needs `datasets/`, which is gitignored
and arrives from Zenodo:

```bash
cd <repo root>
mkdir -p datasets

# Models + training data
#   https://doi.org/10.5281/zenodo.17645488
# AlphaFold 3 structures
#   https://doi.org/10.5281/zenodo.18701748

# Download both records, then extract into datasets/:
tar xf mutpred-ppi-datasets.tar -C datasets/
tar xf af3_structures.tar       -C datasets/     # extract on demand; 6.8 GB
```

Check what resolved:

```bash
conda run -n ppi python src/paths.py    # prints every path and whether it exists
conda run -n ppi python -m pytest tests/ -q
```

The test suite is the fastest way to confirm an install: it runs in under a
minute, needs no GPU, and skips anything whose data is absent rather than
failing.

VarChAMP is unpublished IGVF data and is **not** in either deposit; COSMIC and
HGMD are licence-restricted. Analyses needing them report a clear message and
skip. See [DATA_SOURCES.md](DATA_SOURCES.md).

## The three tiers

**1. In the repo — nothing to configure.**
`datasets/` holds everything the *analysis* layer needs. It is gitignored (VarChAMP is
unpublished and the AF3 tars are large), so it arrives via the Zenodo bundle:

| Directory | Size | Contents |
|---|---|---|
| `datasets/training_eval/` | 6.7 MB + store | The canonical train/eval layer: `<dataset>_rows.csv.gz`, `<dataset>_splits.csv.gz`, `sequences.csv.gz`, `aliases.csv`, and `contact_graphs.h5`. |
| `datasets/variant_dbs/` | 53 MB + store | `{clinvar,gnomad,neurodev,asd}_rows.csv.gz` (plus `cosmic`/`hgmd` if you hold those licences) and `contact_graphs.h5`. One self-contained table per variant repository. |
| `datasets/af3_structures_canonical/` | 12.6 GB | 100,739 gzipped mmCIFs, one per pair, `{ACC_LO}__{ACC_HI}.cif.gz`, plus `manifest.csv` (17 MB). Only the 24,716 in-house structures (5.2 GB) are deposited; the rest come from ProtVar — see [ZENODO.md](ZENODO.md). |
| `datasets/cv_reference/` | 135 MB | Canonical row orderings, cd-hit clusters, fold splits, per-seed test classes, label tables. **The one artifact that cannot be regenerated** — the pooled datasets cannot reproduce their splits without it. |
| `datasets/annotations/` | 269 MB | Allele frequencies, ClinVar variant subsets, pLDDT/Pfam caches, ID maps, ClinGen modes of inheritance, SWING label files. |
| `datasets/annotations_licensed/` | 143 MB | COSMIC and HGMD derived summaries. **Not deposited** — licence-restricted. Analyses that need them print a clear message and skip when absent. |
| `datasets/esignet_supplements/` | 916 MB | The two ESM-2 supplement caches the eSIG-Net blind test reads. VarChAMP-derived, so not deposited. |
| `datasets/af3_structures*.tar` | 7.1 GB | AlphaFold 3 complexes, `af3_structures.tar` (600 MB) and `af3_structures_variant_dbs.tar` (6.6 GB). **Extract both** — the variant-repository figures need the second. |

### One contact-graph store, three names

`datasets/training_eval/contact_graphs.h5`, `datasets/variant_dbs/contact_graphs.h5` and
`datasets/af3_structures_canonical/contact_graphs_v4.h5` are the **same 966 MB file**
containing all 100,739 graphs — hard links, not three stores. The deposit ships one copy;
make the others links or copies.

**2. Symlinked — large, machine-local, or third-party.**
Anything too big to ship, unpublishable, or belonging to someone else is reached through
`external/`:

```bash
bash scripts/link_external.sh                       # uses the default data root
MUTPRED_DATA_ROOT=/your/path bash scripts/link_external.sh
```

That creates 19 links (training data, unpublished VarChAMP, comparator checkpoints, AFDB
monomers, regenerable caches). Links it cannot satisfy are reported and skipped, so a
partial environment still runs whatever it has. `external/` is gitignored.

ProtVar's precomputed AlphaFold3 interface models are a separate ~57 GB download and are
not redistributed here:

```bash
curl -O https://ftp.ebi.ac.uk/pub/databases/ProtVar/predictions/interfaces/2024.05.28_interface_models_high_confidence.tar
tar xf 2024.05.28_interface_models_high_confidence.tar -C <somewhere>
ln -s <somewhere>/pdb external/protvar_pdb
```

`canonicalize_structures.py` reads that directory and keeps only the complexes whose two
chain sequences appear in the canonical tables, so the full download can be pointed at
directly without filtering it first.

**3. Regenerated — never shipped.**
These are model output caches. Deleting them costs compute, not information:

| Artifact | Size | Regenerate with |
|---|---|---|
| `mint_cache.pkl`, `pplm_cache.pkl` | 84 GB each | `src/evaluation/precompute_{mint,pplm}_embeddings.py` |
| `esm2_residue_embeddings*.pkl` | 28 / 40 GB | eSIG-Net's own precompute step |
| `{clinvar,gnomad,cosmic}/prott5_subgraphs.h5` | 122–164 GB | `precompute_prott5.py` then `compress_to_subgraphs.py` |
| `megascale_preprocessed/` | 97 GB | `src/training/preprocess_stability_data.py` |
| `contact_graphs.h5` | 966 MB | `src/data_processing/rebuild_graphs_from_structures.py --structures <canonical dir> --out <h5>` |
| `datasets/annotations/plddt_cache.pkl` | small | `src/analysis/build_plddt_cache.py` (AlphaFold DB monomers) |
| `datasets/annotations/confidence_scores.pkl` | small | `src/analysis/build_confidence_cache.py` (AF3 `*_summary_confidences.json`) |

## Environment variables

All optional — the defaults resolve correctly for a standard checkout.

| Variable | Overrides | Default |
|---|---|---|
| `MUTPRED_DATA_ROOT` | Root of the external data tree | the repo's parent directory |
| `MUTPRED_CV_DIR` | CV reference artifacts | `datasets/cv_reference/` if present |
| `MUTPRED_CACHE_DIR` | Large regenerable caches | `mutpred_ppi_data/` beside the repo |
| `MUTPRED_CDHIT` | `cd-hit` binary | first on `PATH` |

`cd-hit` supplies the GroupKFold groups, so cross-validation cannot run without it:

```bash
conda install -c bioconda cd-hit
```

It is only needed to generate splits from scratch. `datasets/cv_reference/` already contains
the published splits for all 30 seeds of every dataset, and those are loaded in preference to
re-clustering — so the normal path never invokes `cd-hit` at all.

## Minimum footprint

To regenerate every figure and table with no training and no GPU, you need only tier 1 —
about **1.9 GB** of small annotation inputs plus the reconstruction tables. Training from
scratch, re-running a comparator, or scoring a new variant repository additionally requires
tier 2 and possibly tier 3.
