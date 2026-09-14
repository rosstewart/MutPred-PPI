# Data Sources

| Dataset | Source | Notes |
|---------|--------|-------|
| Sahni (Cell 2015) | Supplementary data from the paper | RefSeq IDs; 562 unique proteins. Sahni N, et al. Widespread macromolecular interaction perturbations in human genetic disorders. *Cell.* 2015 Apr 23;161(3):647-660. [doi:10.1016/j.cell.2015.04.013](https://doi.org/10.1016/j.cell.2015.04.013). PMID: 25910212; PMCID: PMC4441215 |
| Fragoza (Nat Commun 2019) | Supplementary data from the paper | UniProt IDs. Fragoza R, Das J, Wierbowski SD, et al. Extensive disruption of protein interactions by genetic variants across the allele frequency spectrum in human populations. *Nat Commun* 10, 4141 (2019). [doi:10.1038/s41467-019-11959-3](https://doi.org/10.1038/s41467-019-11959-3) |
| VarChAMP | Unpublished, IGVF Consortium | Not redistributed here; cross-reference [data.igvf.org](https://data.igvf.org) |
| Tsuboyama (Nature 2023) | [doi.org/10.1038/s41586-023-06328-6](https://doi.org/10.1038/s41586-023-06328-6) | MegaScale stability pretraining data; train/val/test splits (`datasets/mega_splits.pkl`) from Li, Z., Luo, Y. Generalizable and scalable protein stability prediction with rewired protein generative models. *Nat Commun* 17, 891 (2026). https://doi.org/10.1038/s41467-025-67609-4 |
| ClinVar | clinvar.ncbi.nlm.nih.gov | January 2, 2025 release. Pathogenic (P/LP) and benign (B/LB) with ≥1 review star; all missense VUS. AR/AD-only disease genes flagged via ClinGen MOI curations (Chen et al. 2026, doi:10.64898/2026.02.17.706269) |
| gnomAD | gnomad.broadinstitute.org | v4.1.0, all chromosomes; AFs assigned via GroupMax flag |
| COSMIC | cancer.sanger.ac.uk | v101. License required; retained only genes assigned to a single oncogene/TSG class |
| HGMD | hgmd.cf.ac.uk | Professional 2025. License required; "DM" variants only ("DM?" excluded) |
| ASD | Fu et al. 2022 | De novo case variants for autism spectrum disorder; scored as the `asd` group within the `neurodev` database |
| NDD | Pejaver et al. 2020 (MutPred2 paper), *Nature Communications* | Case/control variants across four neurodevelopmental disorders (ASD, intellectual disability, schizophrenia, epileptic encephalopathy); labels in `variant_label_dict.pkl` |
| ProtVar precomputed AF3 interfaces | [EBI FTP](https://ftp.ebi.ac.uk/pub/databases/ProtVar/predictions/interfaces/2024.05.28_interface_models_high_confidence.tar) | 126,118 high-confidence predicted complexes, `2024.05.28_interface_models_high_confidence.tar`. Supplies the partner structures for ClinVar/COSMIC/gnomAD/HGMD that were not folded in-house: 76,023 of the 100,739 canonical structures. Not redistributable by us — see [ZENODO.md](ZENODO.md). |
| BioGRID | thebiogrid.org | Release 4.4.244; physical binding evidence only, used to build the interactome for variant-repository partner selection |
| MINT (comparator method) | Ullanat et al. 2026 | Third-party pretrained protein-pair language model; model checkpoint downloaded separately (not redistributed here), only its output embedding cache is regeneratable in-repo via `precompute_mint_embeddings.py` |
| PPLM (comparator method) | Liu et al. 2026 | Third-party pretrained protein-pair language model; model checkpoint downloaded separately (not redistributed here), only its output embedding cache is regeneratable in-repo via `precompute_pplm_embeddings.py` |
| SAAMBE-3D (comparator method) | Iqbal et al. 2021, Nucleic Acids Research | Sequence- and structure-based ΔΔG predictor; download the source archive from http://compbio.clemson.edu/SAAMBE-3D/ (a project page, not a git remote) and unpack it into `external_methods/saambe3d/`. Ships its own SKEMPI-trained `*_v01.model` XGBoost boosters and requires `prody`; see REPRODUCING_ANALYSES.md. |
| SWING (comparator method) | Ng et al. 2022, Nucleic Acids Research | Network-based protein-pair interaction predictor; implemented in-repo (`src/evaluation/swing_common.py`) from the published algorithm. No external checkout. |
| eSIG-Net (comparator method) | Du et al. 2023, AAAI | Sequence-based interaction perturbation predictor; checkout from https://github.com/Liu-Jing/eSIG-Net (commit cd36a4a0). Feature extraction code not published; ours is reconstructed and validated (`src/evaluation/predictors/validate_esignet_features.py`). |
| MutPPI / MutPPI+ (comparator method) | Dai et al. 2024 | Structure-based interaction perturbation predictor; checkout from https://github.com/Wang-Lin-boop/MutPPI (commit 7a5c6f76). Requires per-fold checkpoint training; see REPRODUCING_ANALYSES.md. |
| AlphaFold3 structures (this study) | Zenodo: [10.5281/zenodo.18701748](https://doi.org/10.5281/zenodo.18701748) | Subject to AlphaFold Server Output Terms of Use. Training/evaluation complexes (`datasets/af3_structures.tar`) and variant-repository complexes for ClinVar/gnomAD/NDD/ASD (`datasets/af3_structures_variant_dbs.tar`) — both individually gzipped per-structure in an uncompressed outer tar for random-access extraction. COSMIC/HGMD variant-DB structures excluded (licensing). |
| Trained models + Sahni/Fragoza training data | Zenodo: [10.5281/zenodo.17645488](https://doi.org/10.5281/zenodo.17645488) | Post-AF3-structure-filtering; used for Fig 3 GCV |

## Source files for the training/evaluation mapping

`notebooks/map_ppi_datasets.py` reads eight files across two gitignored
tiers. They are not vendored in this repo -- download the four published ones
from the supplementary material of the two papers cited above.

| tier | directory | files |
|---|---|---|
| published | `datasets/source_data/` | `sahni_wt_and_mt_y2h_scores.csv`, `fragoza_cosmic.csv`, `fragoza_exac.csv`, `fragoza_hgmd.csv` |
| restricted | `datasets/source_data_restricted/` | four VarChAMP/IGVF files -- unpublished, **not redistributable**, absent from Zenodo |

Without the restricted tier the `sahni_only`, `fragoza_only` and `sahni_fragoza`
datasets still build from the published files alone; the two `varchamp*` datasets
do not.

The mapping also carries a 20 MB UniProt REST cache
(`datasets/source_mapping/cache/`, 4,478 gzipped responses). Carrying it is
optional but **strongly recommended**: without it the notebook queries
`rest.uniprot.org` live, and UniProt's answers change over time, so the mapping
is only exactly reproducible with the cached responses in place.

See [DATA_PREPARATION.md](DATA_PREPARATION.md) for the full ordered chain.

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

Their outputs are then collapsed into one self-contained table per database,
`datasets/variant_dbs/{db}_rows.csv.gz`, by
`src/variant_db_inference/build_variant_db_tables.py`. Every analysis reads that table rather
than the individual pickles. Columns and row counts:
[`docs/REPRODUCING_ANALYSES.md`](REPRODUCING_ANALYSES.md#variant-database-tables).

## Licensing / exclusions

VarChAMP raw data is unpublished IGVF consortium data and is excluded from all public releases (git, Zenodo). COSMIC and HGMD variant-partner interaction data are excluded due to commercial/academic licensing restrictions — obtain directly from their respective sources using the versions above. gnomAD and ClinVar must also be downloaded directly (public, no redistribution restriction, but not bundled here for size reasons).
