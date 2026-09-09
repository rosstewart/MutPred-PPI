# Setup: where the code looks for data

Every path the code uses is resolved through one module, [`src/paths.py`](../src/paths.py).
Nothing under `src/` contains an absolute path any more, so the repo runs from any
directory and on any machine once the data is in place.

Check what resolves where at any time:

```bash
conda run -n ppi python src/paths.py      # prints every resolved path and whether it exists
```

## The three tiers

**1. In the repo — nothing to configure.**
`datasets/` holds everything the *analysis* layer needs. It is gitignored (VarChAMP is
unpublished and the AF3 tars are large), so it arrives via the Zenodo bundle:

| Directory | Size | Contents |
|---|---|---|
| `datasets/cv_reference/` | 355 MB | Canonical row orderings, cd-hit clusters, fold splits, per-seed test classes, label tables. Replaces the external `cv_splits/` the code used to read. |
| `datasets/annotations/` | 258 MB | Allele frequencies, ClinVar variant subsets, pLDDT/Pfam caches, ID maps, ClinGen MOI, SWING label files. |
| `datasets/annotations_licensed/` | 143 MB | COSMIC and HGMD derived summaries. **Not in the Zenodo deposit** — licence-restricted. Analyses that need them report a clear message and skip when absent. |
| `datasets/esignet_supplements/` | 916 MB | The two ESM-2 supplement caches the eSIG-Net blind test reads. |
| `datasets/reconstruction_tables/` | 460 MB | Per-figure prediction/label tables — regenerate every curve with no training. |
| `datasets/af3_structures*.tar` | 6.8 GB | AlphaFold 3 complexes. Extract on demand. |

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

**3. Regenerated — never shipped.**
These are model output caches. Deleting them costs compute, not information:

| Artifact | Size | Regenerate with |
|---|---|---|
| `mint_cache.pkl`, `pplm_cache.pkl` | 84 GB each | `src/evaluation/precompute_{mint,pplm}_embeddings.py` |
| `esm2_residue_embeddings*.pkl` | 28 / 40 GB | eSIG-Net's own precompute step |
| `{clinvar,gnomad,cosmic}/prott5_subgraphs.h5` | 122–164 GB | `precompute_prott5.py` then `compress_to_subgraphs.py` |
| `megascale_preprocessed/` | 97 GB | `src/training/preprocess_stability_data.py` |
| `data_caches/*_cache.pkl` | up to 222 GB | Optional. `--data-cache` defaults to off; passing it trades ~10 h of reload time for disk. |

**One file in `data_caches/` is not a cache.** `data_caches/training_data_internal.csv`
(26 MB) is a required input for five modules, including the Table 1 generator. It is
VarChAMP-derived (~56% of its rows) and so is neither in git nor in the Zenodo deposit —
see [`docs/ZENODO_MANIFEST.md`](ZENODO_MANIFEST.md). Do not delete it when clearing caches;
`repro_test/reclaim_disk.sh` preserves it.

## Environment variables

All optional — the defaults resolve correctly for a standard checkout.

| Variable | Overrides | Default |
|---|---|---|
| `MUTPRED_DATA_ROOT` | Root of the external data tree | the repo's parent directory |
| `MUTPRED_CV_DIR` | CV reference artifacts | `datasets/cv_reference/` if present |
| `MUTPRED_CACHE_DIR` | Large regenerable caches | `$MUTPRED_DATA_ROOT/nm_revisions` |
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
