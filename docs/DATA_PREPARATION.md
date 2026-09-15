# Data preparation

The ordered chain that turns published source files into the tables, structures
and graphs every figure script reads. Each stage names its inputs, its outputs
and the script that performs it, so a clone can be brought to a runnable state
without guessing which artifact came from where.

Everything below writes under `datasets/`, which is **not** in git (see
[DATA.md](DATA.md)). Paths resolve through `src/paths.py`; nothing in the
pipeline hardcodes an absolute path.

```
  source files                notebooks/map_ppi_datasets.py
  (published + restricted)  ────────────────────────────────────►  datasets/source_mapping/
                                                                          │
                             src/data_processing/training_sets/            │
                             prepare_gcv_tables.py                         ▼
                           ◄────────────────────────────────────  datasets/training_eval/
                                                                    *_rows.csv.gz
                                                                    *_splits.csv.gz
                                                                    sequences.csv.gz
                                                                          │
  AlphaFold3 predictions                                                  ▼
  ──────────────────────►  canonicalize_structures.py  ──►  datasets/af3_structures_canonical/
         ▲                                                          │        manifest.csv
         │                                                          ▼
         │                 rebuild_graphs_from_structures.py  ──►  contact_graphs.h5
         │                                                          │
         └───── fold externally ◄── prepare_af3_inputs.py ◄─────────┘
```

## Stage 0, source data

Two tiers, both gitignored.

| tier | directory | contents |
|---|---|---|
| published | `datasets/source_data/` | `sahni_wt_and_mt_y2h_scores.csv`, `fragoza_cosmic.csv`, `fragoza_exac.csv`, `fragoza_hgmd.csv`, `skempi_v2.csv`, `pdb_chain_uniprot.csv` |
| restricted | `datasets/source_data_restricted/` | four VarChAMP/IGVF files |

The published files come from the supplementary material of the two source
papers and from two public reference databases. Citations and download
locations for all six are in
[Where every dataset comes from](#where-every-dataset-comes-from) below. The VarChAMP/IGVF files were unpublished consortium data at time of
release and **cannot be redistributed**; they are absent from Zenodo, and every
stage below degrades cleanly without them (the `sahni_only`, `fragoza_only` and
`sahni_fragoza` datasets are buildable from the published files alone).

## Stage 1, mapping

```bash
# as a notebook (recommended -- it emits QC artifacts meant to be read)
jupytext --to ipynb notebooks/map_ppi_datasets.py && jupyter lab
# or straight through
conda run -n ppi python notebooks/map_ppi_datasets.py
```

Resolves every source identifier (accession, RefSeq, Entrez GeneID, gene symbol,
hORFeome ORF) to a reviewed human UniProt accession, validates each mutation
against the resolved sequence, and writes:

| output | contents |
|---|---|
| `datasets/source_mapping/datasets/` | the five mapped dataset CSVs, plus narrower VarChAMP slices |
| `datasets/source_mapping/master_data/` | the complete un-deduplicated record, one row per observation |
| `datasets/source_mapping/intermediate_files/` | 11 QC/audit CSVs, why rows were dropped, remapped, collapsed |
| `datasets/source_mapping/cache/` | UniProt REST responses, so a re-run is offline |

**Carry the cache.** Without it the notebook queries `rest.uniprot.org` live, and
UniProt's answers change over time, which means the mapping is only exactly
reproducible with the cached responses in place. It is 20 MB.

### Ambiguous Entrez GeneIDs

A few NCBI GeneIDs cross-reference more than one *reviewed human* Swiss-Prot
entry. This is not a filter failure, the Entrez route requests
`to_db=UniProtKB-Swiss-Prot` (reviewed-only, server-side) and applies a hard
human filter client-side. UniProt genuinely maps them to several entries.

Two cases, treated differently:

- **Sequence-identical paralog families** (`122183` PRR20A–E, `441521` CT45A5/6/7).
  Every candidate carries the same sequence, so the choice cannot matter. Passes.
- **Genuinely different products under one GeneID** (`51207` DUSP13, `83871`
  RAB34, `9465` AKAP7). Here a wrong pick substitutes an unrelated protein, DUSP13A and DUSP13B share **3.7% identity**. These are pinned explicitly in
  `ENTREZ_AMBIGUOUS_PINS` with the reason for each, and a *new* divergent GeneID
  that is not pinned raises rather than being resolved by a length heuristic.

## Stage 2a, record AlphaFold3 coverage

```bash
conda run -n ppi python src/data_processing/annotate_af3_coverage.py
```

Writes an `af3_failed` column into every mapping CSV: `True` where the row's
`(interactor, partner)` pair is absent from
`datasets/af3_structures_canonical/manifest.csv`. Pairs are compared unordered.

A complex with no structure has no contact graph, so every structure-based
method scores it `NaN`. Stage 2 **drops** these rows before assigning
`row_index`, which keeps that column a contiguous 0..n-1 positional key for the
splits table and every downstream cache. The flagged rows stay in the mapping
CSVs, and only there, as the record of what was excluded and why.

Run this after stage 3 (structures) and re-run it whenever a new batch of folds
lands; currently it flags 0.06-0.34% of rows per dataset.

## Stage 2, GCV row and split tables

```bash
conda run -n ppi python src/data_processing/training_sets/prepare_gcv_tables.py \
    --dataset all --n-seeds 30
```

Consumes the mapped CSVs -- **excluding every `af3_failed = True` row**, see
stage 2a -- and writes `datasets/training_eval/`:
`sequences.csv.gz`, and per dataset `{name}_rows.csv.gz` + `{name}_splits.csv.gz`.

`row_index` is the CSV's natural order and is never renumbered, it is the join
key between the two tables and everything downstream. Clustering is cd-hit at
50% identity **on the full concatenated complex sequence**, which is the
published convention and must not change. Folds are `GroupKFold(10, shuffle=True,
random_state=seed)` over those clusters; `test_class` is C1/C2/C3 by whether each
protein of a test pair was seen in that fold's training set.

Use `--out` to write to a scratch directory and diff against the live tables
without overwriting them.

Expected row counts: `sahni_only` 1,591 · `fragoza_only` 4,726 ·
`sahni_fragoza` 6,212 · `varchamp_all` 17,317 ·
`sahni_fragoza_varchamp_all` 23,254.

## Stage 2b, SKEMPI reference (comparison-method stratification)

```bash
conda run -n ppi python src/data_processing/training_sets/prepare_skempi_reference.py
```

SAAMBE-3D, MutPPI and MutPPI+ ship pretrained on SKEMPI 2.0 rather than being
retrained per dataset, so their C1/C2/C3 split is defined by overlap with
*SKEMPI's* proteins. This derives that set and writes
`datasets/annotations/skempi_train_uniprots.csv` (342 accessions).

Inputs are both in `datasets/source_data/`: `skempi_v2.csv` (SKEMPI 2.0,
semicolon-delimited) and `pdb_chain_uniprot.csv` (SIFTS, per-chain).

**Get the right SIFTS file.** SKEMPI identifies complexes as
`PDB_<chains>_<chains>` (`1CSE_E_I`), so the mapping must be **per chain**, columns `PDB,CHAIN,SP_PRIMARY...`. The similarly named SIFTS `uniprot_pdb`
file maps a UniProt accession to a list of PDB ids with no chain column and
cannot resolve these.

**Every character in a chain group is a chain.** 122 of SKEMPI's 348 complexes
have multi-character groups, `3SE8_HL_G`, `1BD2_ABC_DE`, overwhelmingly
antibodies, where `H` and `L` are the heavy and light chains of one partner.
A chain may also map to more than one accession (chimeric constructs), so
resolution is a set union. 130 (pdb, chain) pairs have no SIFTS mapping, mostly engineered antibody constructs with no UniProt entry, and are excluded.

## Stage 3, structures

AlphaFold3 predictions arrive as per-job directories. Canonicalise them into one
content-addressed tree:

```bash
conda run -n ppi python src/data_processing/canonicalize_structures.py \
    --structures datasets/af3_structures --out datasets/af3_structures_canonical
```

Files are named `{ACC_LO}__{ACC_HI}.cif.gz`. The double underscore is load-bearing:
no UniProt accession contains `_`, so isoform accessions like `O14787-2` survive
a round trip, which a single `-` join cannot (`O14787-2-Q13207` is ambiguous).
**Chain orientation is resolved at load time from the manifest, never from the
filename.**

## Stage 4, contact graphs

```bash
conda run -n ppi python src/data_processing/rebuild_graphs_from_structures.py \
    --structures datasets/af3_structures_canonical \
    --out datasets/training_eval/contact_graphs.h5
```

A 4.5 Å any-heavy-atom contact rule (`contact_graphs.DEFAULT_THRESHOLD`, the one
definition). Keying and orientation are documented once, in
[`docs/INFERENCE.md`](INFERENCE.md#the-contact-graph-store).

**Rebuild this whenever the structure set grows.** A store with fewer graphs than
the manifest has rows is stale, and every newly-folded pair silently scores NaN.

## Stage 5, AlphaFold3 inputs for what is still missing

```bash
conda run -n ppi python src/data_processing/prepare_af3_inputs.py
```

A standing step, not an anomaly report: it always runs, reports coverage, and
exits cleanly when nothing is missing.

- **required** = the union of unordered `(interactor, partner)` pairs across all
  five canonical datasets. Not the pooled table alone: pooling drops rows whose
  label conflicts across sources, so a pair can be absent there while present and
  unconflicted in a smaller dataset.
- **present** = any pair in the manifest, in *either* chain order.

It writes `missing_pairs.csv` (the audit record) and one AF3 JSON per pair.
JSON emission is delegated to `src/inference/00_make_af3_json_input.py`, the one
JSON writer, so the naming, dialect and residue normalisation cannot drift.

Fold the emitted JSONs on a GPU machine, then return to **stage 3**.

## Where every dataset comes from

| Dataset | Source | Notes |
|---------|--------|-------|
| Sahni (Cell 2015) | **Table S3A** of the paper | `sahni_wt_and_mt_y2h_scores.csv`. RefSeq IDs; 562 unique proteins. Sahni N, et al. Widespread macromolecular interaction perturbations in human genetic disorders. *Cell.* 2015 Apr 23;161(3):647-660. [doi:10.1016/j.cell.2015.04.013](https://doi.org/10.1016/j.cell.2015.04.013). PMID: 25910212; PMCID: PMC4441215 |
| Fragoza (Nat Commun 2019) | **Supplementary Data 2, 3 and 4** of the paper | `fragoza_exac.csv` (ExAC variants, Supplementary Data 2), `fragoza_cosmic.csv` (COSMIC somatic mutations, Supplementary Data 3), `fragoza_hgmd.csv` (HGMD disease-associated mutations, Supplementary Data 4). UniProt IDs. Fragoza R, Das J, Wierbowski SD, et al. Extensive disruption of protein interactions by genetic variants across the allele frequency spectrum in human populations. *Nat Commun* 10, 4141 (2019). [doi:10.1038/s41467-019-11959-3](https://doi.org/10.1038/s41467-019-11959-3) |
| SKEMPI 2.0 (`skempi_v2.csv`) | [life.bsc.es/pid/skempi2](https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv) | Semicolon-delimited. Binding-affinity changes on mutation, used to derive the 342 accessions SAAMBE-3D/MutPPI/MutPPI+ were pretrained on. Jankauskaite J, Jimenez-Garcia B, Dapkunas J, Fernandez-Recio J, Moal IH. SKEMPI 2.0: an updated benchmark of changes in protein-protein binding energy, kinetics and thermodynamics upon mutation. *Bioinformatics.* 2019 Feb 1;35(3):462-469. [doi:10.1093/bioinformatics/bty635](https://doi.org/10.1093/bioinformatics/bty635) |
| SIFTS (`pdb_chain_uniprot.csv`) | [EBI FTP](https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz) | The **per-chain** mapping, columns `PDB,CHAIN,SP_PRIMARY,...`; the similarly named `uniprot_pdb` file is per-entry and will not work. Maps SKEMPI's `PDB_<chains>_<chains>` ids to UniProt. Dana JM, et al. SIFTS: updated Structure Integration with Function, Taxonomy and Sequences resource. *Nucleic Acids Res.* 2019;47(D1):D482-D489. [doi:10.1093/nar/gky1114](https://doi.org/10.1093/nar/gky1114) |
| VarChAMP | Unpublished, IGVF Consortium | Not redistributed here; cross-reference [data.igvf.org](https://data.igvf.org) |
| Tsuboyama (Nature 2023) | [doi.org/10.1038/s41586-023-06328-6](https://doi.org/10.1038/s41586-023-06328-6) | MegaScale stability pretraining data; train/val/test splits (`datasets/mega_splits.pkl`) from Li, Z., Luo, Y. Generalizable and scalable protein stability prediction with rewired protein generative models. *Nat Commun* 17, 891 (2026). https://doi.org/10.1038/s41467-025-67609-4 |
| ClinVar | clinvar.ncbi.nlm.nih.gov | January 2, 2025 release. Pathogenic (P/LP) and benign (B/LB) with ≥1 review star; all missense VUS. AR/AD-only disease genes flagged via ClinGen MOI curations (Chen et al. 2026, doi:10.64898/2026.02.17.706269) |
| gnomAD | gnomad.broadinstitute.org | v4.1.0, all chromosomes; AFs assigned via GroupMax flag |
| COSMIC | cancer.sanger.ac.uk | v101. License required; retained only genes assigned to a single oncogene/TSG class |
| HGMD | hgmd.cf.ac.uk | Professional 2025. License required; "DM" variants only ("DM?" excluded) |
| ASD | Fu et al. 2022 | De novo case variants for autism spectrum disorder; scored as the `asd` group within the `neurodev` database |
| NDD | Pejaver et al. 2020 (MutPred2 paper), *Nature Communications* | Case/control variants across four neurodevelopmental disorders (ASD, intellectual disability, schizophrenia, epileptic encephalopathy); labels in `variant_label_dict.pkl` |
| ProtVar precomputed AF3 interfaces | [EBI FTP](https://ftp.ebi.ac.uk/pub/databases/ProtVar/predictions/interfaces/2024.05.28_interface_models_high_confidence.tar) | 126,118 high-confidence predicted complexes, `2024.05.28_interface_models_high_confidence.tar`. Supplies the partner structures for ClinVar/COSMIC/gnomAD/HGMD that were not folded in-house: 76,023 of the 100,739 canonical structures. Not redistributable by us; see [DATA.md](DATA.md). |
| BioGRID | thebiogrid.org | Release 4.4.244; physical binding evidence only, used to build the interactome for variant-repository partner selection |
| MINT (comparator method) | Ullanat et al. 2026 | Third-party pretrained protein-pair language model; model checkpoint downloaded separately (not redistributed here), only its output embedding cache is regeneratable in-repo via `precompute_mint_embeddings.py` |
| PPLM (comparator method) | Liu et al. 2026 | Third-party pretrained protein-pair language model; model checkpoint downloaded separately (not redistributed here), only its output embedding cache is regeneratable in-repo via `precompute_pplm_embeddings.py` |
| SAAMBE-3D (comparator method) | Iqbal et al. 2021, Nucleic Acids Research | Sequence- and structure-based ΔΔG predictor; download the source archive from http://compbio.clemson.edu/SAAMBE-3D/ (a project page, not a git remote) and unpack it into `external_methods/saambe3d/`. Includes its own SKEMPI-trained `*_v01.model` XGBoost boosters and requires `prody`; see REPRODUCING_ANALYSES.md. |
| SWING (comparator method) | Ng et al. 2022, Nucleic Acids Research | Network-based protein-pair interaction predictor; implemented in-repo (`src/evaluation/swing_common.py`) from the published algorithm. No external checkout. |
| eSIG-Net (comparator method) | Du et al. 2023, AAAI | Sequence-based interaction perturbation predictor; checkout from https://github.com/Liu-Jing/eSIG-Net (commit cd36a4a0). Feature extraction code not published; ours is reconstructed and validated (`src/evaluation/predictors/validate_esignet_features.py`). |
| MutPPI / MutPPI+ (comparator method) | Dai et al. 2024 | Structure-based interaction perturbation predictor; checkout from https://github.com/Wang-Lin-boop/MutPPI (commit 7a5c6f76). Requires per-fold checkpoint training; see REPRODUCING_ANALYSES.md. |

## Mapping scripts

Each raw download is mapped to UniProt-keyed variant/partner records by one script under
`src/data_processing/variant_databases/`:

| Dataset | Script |
|---|---|
| ClinVar | `map_clinvar.py` |
| gnomAD | `map_gnomad.py` |
| COSMIC (recurrence, onco/TSG) | `get_cosmic_annotations.py`, `map_cosmic.py` |
| HGMD | `map_hgmd.py` |
| NeuroDev NDD case/control | `map_neurodev.py --mode neurodev` |
| ASD de novo (Fu et al.) | `map_asd_ndd.py` |

The `neurodev` variant database is the **NeuroDev case/control** cohort, not the
Fu et al. ASD set: its `neurodev_label` column (1 = case, 0 = control) comes from
`variant_label_dict.pkl`, which only `map_neurodev.py --mode neurodev` writes.
`map_asd_ndd.py` maps the Fu et al. de novo ASD missense variants and emits no
case/control labels. The two are separate datasets and are not interchangeable.

Their outputs are then collapsed into one self-contained table per database
`datasets/variant_dbs/{db}_rows.csv.gz`, by
`src/variant_db_inference/build_variant_db_tables.py`. Every analysis reads that table rather
than the individual pickles. Columns and row counts:
[`docs/REPRODUCING_ANALYSES.md`](REPRODUCING_ANALYSES.md#variant-database-tables).

Licensing and what is redistributable: [DATA.md](DATA.md#what-is-not-deposited).

## Order

Stages 1 → 2 → 3 → 4, with 5 feeding back into 3. Stage 2 must precede stage 5,
since the required-pair set is derived from the tables stage 2 writes.
`notebooks/reproduce_all_figures.py` runs stages 2–5 in this order and caches on
output existence.
