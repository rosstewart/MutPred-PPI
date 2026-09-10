# Zenodo deposit manifest

What ships, what does not, and why. `datasets/` is gitignored, so the deposit — not the
git clone — is what makes the analysis layer runnable. See [`docs/SETUP.md`](SETUP.md).

Budget was ~40 GB. The deposit comes to **~9 GB**, so nothing was cut for size.

## Already deposited

| Artifact | Size | DOI |
|---|---|---|
| `af3_structures.tar` (training/eval complexes) | 573 MB | [10.5281/zenodo.18701748](https://doi.org/10.5281/zenodo.18701748) |
| `af3_structures_variant_dbs.tar` (ClinVar/gnomAD/NDD/ASD complexes) | 6.2 GB | same |
| `train_eval.tar.gz`, `mega_splits.pkl` | 3 MB | [10.5281/zenodo.17645488](https://doi.org/10.5281/zenodo.17645488) |
| Trained model weights | 24 MB | same |
| `master_variant_db_predictions.csv.gz` | 25 MB | same |

## To add

| Artifact | Size | Unlocks |
|---|---|---|
| `datasets/reconstruction_tables/` | 460 MB | Every ROC/PR curve in the paper, recomputed from stored per-variant predictions — no training, no GPU |
| `results_revisions/macro_aucs/*.pkl,*.npy` | 240 MB | The 30-seed GCV results behind Fig 3, S1, S3, S-abl, S-new |
| `datasets/cv_reference/` | 355 MB | Canonical orderings, clusters, fold splits and per-seed test classes for all 30 seeds. **Required** — the pooled datasets cannot reproduce their splits from raw, and without this a clone cannot reproduce SFVCFP at all |
| `datasets/annotations/` | 258 MB | The small annotation/label inputs every analysis script now resolves against |
| `datasets/esignet_supplements/` | 916 MB | The two ESM-2 caches the eSIG-Net blind test reads (extracted from a 110 GB upstream tree) |
| `datasets/mapped090826/` rows/splits/sequences/`af3_index` + `contact_graphs.h5` | 39 MB store + small tables | The canonical train/eval data layer everything now reads |
| `datasets/variant_dbs/*_rows.csv.gz` | 68 MB | The five self-contained variant-DB tables (ClinVar/gnomAD/autism ship; COSMIC/HGMD rows are licence-restricted, see below) |

Total addition: **~2.3 GB**.

`datasets/variant_dbs/contact_graphs.h5` (248 MB) and the canonical structure trees
(`datasets/af3_structures_canonical/` 510 MB, `datasets/af3_structures_variant_dbs_canonical/`
5.7 GB) are derivable from the deposited `af3_structures*.tar` via
`src/data_processing/canonicalize_structures.py` then `src/data_processing/rebuild_graphs_from_structures.py`, so
depositing them is a convenience rather than a requirement.

## Deliberately excluded

**`datasets/annotations_licensed/` (143 MB) — licence-restricted.**
COSMIC `vt_to_tumor_site.pkl` and `onco_tsg_dict.pkl`, HGMD `hgmd_variant_subset.pkl`. These
are derived summaries, but still derivative works of licensed databases. Obtain COSMIC v101
and HGMD Professional 2025 directly and regenerate with
`src/data_processing/variant_databases/map_{cosmic,hgmd}.py`. Analyses that need them report
a clear message and skip when absent.

**`results/varchamp_seqcnf_newvar_eval/` (34 MB) — unpublished data.**
Checked rather than assumed: the `*_vt_ids.npy` arrays hold variant identities
(`O43790 E402Q`) paired with 0/1 `*_labels.npy` values, and those labels *are* the
unpublished IGVF Y2H measurements. Shipping the arrays would disclose the dataset, and
stripping the labels leaves predictions that cannot be scored. **Fig 4 and S2 are therefore
not independently reproducible until VarChAMP is published** — consistent with the position
already stated in the README.

**`data_caches/training_data_internal.csv` was never actually required — corrected
2026-09-10.** The previous claim above (that it was a required input for five modules) was
wrong: it has no producer anywhere in git history, and its only two code references
(`swing_common.py`'s `load_benchmark()`, and a constant in
`precompute_prott5_datasets.py`) were both dead code with no live callers.
`generate_training_table.py`, `interface_analysis.py`, and `repro_test/build_sfvcfp_table.py`
never referenced it at all — they read the canonical row tables directly, as this file's own
docstring already stated. It has been moved to `archive/pre_090826/data_caches/`.

This does **not** resolve the underlying VarChAMP concern, only relocates it: Table 1's
VarChAMP row (`generate_training_table.py`'s `_VARCHAMP` dataset) reads the canonical
`varchamp_all_mapped090826` table directly, which carries the same unpublished VarChAMP
measurements the archived CSV used to. **Table 1 is still not independently reproducible**,
alongside Fig 4 and S2, for the reason already stated: shipping the VarChAMP rows would
disclose an unpublished dataset. The affected file is now the canonical table, not the
archived CSV.

**Regenerable caches (~1 TB) — scripted, not shipped.**
`mint_cache.pkl` / `pplm_cache.pkl` (84 GB each), `esm2_residue_embeddings*.pkl` (68 GB),
`prott5_subgraphs.h5` (449 GB), `megascale_preprocessed/` (97 GB), `data_caches/*_cache.pkl`
(up to 222 GB). All are model output. Regeneration commands are in
[`docs/SETUP.md`](SETUP.md) and [`docs/REPRODUCING_ANALYSES.md`](REPRODUCING_ANALYSES.md).

**Third-party checkpoints — download from the original authors.**
MINT `mint.ckpt` (3.0 GB), PPLM `pplm_t33_650M.pt` (2.5 GB), the eSIG-Net SDNN source, AFDB
v4 human monomers, SPURS MegaScale inputs, BioGRID. Pinned commits are in
[`docs/METHOD_PROVENANCE.md`](METHOD_PROVENANCE.md).

**SWING's Doc2Vec model (42 MB) — part of the model, not an input.**
Data-dependent and retrained per fold for the blind-test variant; only the leaky
"Test Pretrain" variant loads a prebuilt copy.

## Packaging note

Keep the tars only. `datasets/` currently holds each tar *beside* its own extracted copy —
6.8 GB of pure duplication (`af3_structures_variant_dbs.tar` + the extracted tree, etc.).
`repro_test/reclaim_disk.sh` Tier 4 removes the extracted copies; no script under `src/`
reads them (verified by grep), so they are reconstructible with `tar xf` on demand.

Tier 4 touches only `datasets/af3_structures/` and `datasets/af3_structures_variant_dbs/`. It
does **not** touch the `*_canonical/` trees, and it should not: those are the deduplicated,
content-named structures that `src/data_processing/rebuild_graphs_from_structures.py` reads, and they are
not reconstructible with `tar xf` alone.
