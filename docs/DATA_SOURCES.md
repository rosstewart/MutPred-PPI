# Data Sources

| Dataset | Source | Notes |
|---------|--------|-------|
| Sahni (PNAS 2015) | Supplementary data from paper | RefSeq IDs; 562 unique proteins |
| Fragoza (Nat Commun 2019) | Supplementary data | UniProt IDs |
| VarChAMP | Unpublished, IGVF Consortium | Not redistributed here; cross-reference [data.igvf.org](https://data.igvf.org) |
| Tsuboyama (Science 2023) | [doi.org/10.1038/s41586-023-06328-6](https://doi.org/10.1038/s41586-023-06328-6) | MegaScale stability pretraining data; train/val/test splits (`datasets/mega_splits.pkl`) from Li, Z., Luo, Y. Generalizable and scalable protein stability prediction with rewired protein generative models. *Nat Commun* 17, 891 (2026). https://doi.org/10.1038/s41467-025-67609-4 |
| ClinVar | clinvar.ncbi.nlm.nih.gov | January 2, 2025 release. Pathogenic (P/LP) and benign (B/LB) with ≥1 review star; all missense VUS. AR/AD-only disease genes flagged via ClinGen MOI curations (Chen et al. 2026, doi:10.64898/2026.02.17.706269) |
| gnomAD | gnomad.broadinstitute.org | v4.1.0, all chromosomes; AFs assigned via GroupMax flag |
| COSMIC | cancer.sanger.ac.uk | v101. License required; retained only genes assigned to a single oncogene/TSG class |
| HGMD | hgmd.cf.ac.uk | Professional 2025. License required; "DM" variants only ("DM?" excluded) |
| ASD | Fu et al. 2022 | Case variants specifically associated with autism spectrum disorder; labels in `variant_label_dict.pkl` |
| NDD | Pejaver et al. 2020 (MutPred2 paper), *Nature Communications* | Case/control variants across four neurodevelopmental disorders (ASD, intellectual disability, schizophrenia, epileptic encephalopathy); labels in `variant_label_dict.pkl` |
| BioGRID | thebiogrid.org | Release 4.4.244; physical binding evidence only, used to build the interactome for variant-repository partner selection |
| MINT (comparator method) | Ullanat et al. 2026 | Third-party pretrained protein-pair language model; model checkpoint downloaded separately (not redistributed here), only its output embedding cache is regeneratable in-repo via `precompute_mint_embeddings.py` |
| PPLM (comparator method) | Liu et al. 2026 | Third-party pretrained protein-pair language model; model checkpoint downloaded separately (not redistributed here), only its output embedding cache is regeneratable in-repo via `precompute_pplm_embeddings.py` |
| AlphaFold3 structures (this study) | Zenodo: [10.5281/zenodo.18701748](https://doi.org/10.5281/zenodo.18701748) | Subject to AlphaFold Server Output Terms of Use. Training/evaluation complexes (`datasets/af3_structures.tar`) and variant-repository complexes for ClinVar/gnomAD/NDD/ASD (`datasets/af3_structures_variant_dbs.tar`) — both individually gzipped per-structure in an uncompressed outer tar for random-access extraction. COSMIC/HGMD variant-DB structures excluded (licensing). |
| Trained models + Sahni/Fragoza training data | Zenodo: [10.5281/zenodo.17645488](https://doi.org/10.5281/zenodo.17645488) | Post-AF3-structure-filtering; used for Fig 3 GCV |

## Mapping scripts

Each raw download is mapped to UniProt-keyed variant/partner records by one script under
`src/data_processing/variant_databases/`:

| Dataset | Script |
|---|---|
| ClinVar | `map_clinvar.py` |
| COSMIC (recurrence, onco/TSG) | `get_cosmic_annotations.py`, `map_cosmic.py` |
| HGMD | `map_hgmd.py` |
| ASD / NDD | `map_tulika_autism.py` |

Their outputs are then collapsed into one self-contained table per database,
`datasets/variant_dbs/{db}_rows.csv.gz`, by
`src/variant_db_inference/build_variant_db_tables.py`. Every analysis reads that table rather
than the individual pickles. Columns and row counts:
[`docs/REPRODUCING_ANALYSES.md`](REPRODUCING_ANALYSES.md#variant-database-tables).

## Licensing / exclusions

VarChAMP raw data is unpublished IGVF consortium data and is excluded from all public releases (git, Zenodo). COSMIC and HGMD variant-partner interaction data are excluded due to commercial/academic licensing restrictions — obtain directly from their respective sources using the versions above. gnomAD and ClinVar must also be downloaded directly (public, no redistribution restriction, but not bundled here for size reasons).
