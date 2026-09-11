# MutPred-PPI

Official repository for "Predicting interaction-specific protein–protein interaction perturbations by missense variants with MutPred-PPI", published in RECOMB 2026.

## Paper Links

- **RECOMB 2026 Proceedings**  
  https://recomb.org/proceedings/proceedings/2030-2026/2026/

- **PDF link**  
  https://doi.org/10.64898/2025.12.20.695738

## Overview

MutPred-PPI is a deep learning framework that predicts whether missense mutations disrupt protein-protein interactions. It combines structural information from protein complexes with sequence embeddings from protein language models to achieve high-accuracy predictions.

**Key Features:**
- Graph neural networks with attention mechanisms for structural analysis
- ProtT5 protein language model embeddings for sequence representation
- Binary classification: probabilistic score ranging from 0-1, where 1 indicates high probability of interaction disruption and 0 indicates preserved interaction
- Parallel processing support for large-scale analysis

## Installation

```bash
# Clone repository
git clone https://github.com/rosstewart/mutpred-ppi.git
cd mutpred-ppi

# Create conda environment
# Named `ppi` because every command in docs/ uses `conda run -n ppi`.
conda create -n ppi python=3.10 -y
conda activate ppi

# Install sentencepiece (required for ProtT5)
conda install sentencepiece -c conda-forge -y

# Install dependencies. requirements.txt pins the exact published environment,
# including torch==2.5.1 -- so install it FIRST if you need a specific CUDA
# build, or requirements.txt will pull the default PyPI wheel over the top.
#   CUDA 12.1:  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
#   CPU only:   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Install this repo as a package.
# `pyproject.toml` makes src/ the package root, so modules import as
# `training.train_fold`, `utils.gcv_common`, `contact_graphs`, ... Scripts
# outside src/inference/ will not import without this.
pip install -e .
```

`requirements.txt` pins the exact environment the published results were produced in.
Cross-validation
additionally needs `cd-hit`, which supplies the clustering groups:

```bash
conda install -c bioconda cd-hit -y
```

See [`docs/SETUP.md`](docs/SETUP.md) for where the code expects data to live.

### System Requirements

- Python 3.10 or higher (`pyproject.toml` requires it)
- CUDA-capable GPU (recommended for faster inference)
- 16GB+ RAM recommended
- ~15 GB disk space for the Zenodo data bundle; the inference quickstart needs no download (structures are bundled)


## Quick Start

```bash
# Step 1: Prepare AlphaFold3 inputs (if using AlphaFold3)
python src/inference/00_make_af3_json_input.py proteins.fasta variants.tsv af3_inputs/

# Step 2: Obtain protein complex structures (see docs/INFERENCE.md)

# Step 3: Generate contact graphs
python src/inference/01_make_contact_graphs_and_fasta.py working_dir/ mmcif_dir/ variants.tsv

# Step 4: Run predictions
python src/inference/02_run_mutpred-ppi_inference.py working_dir/
```

For a small, ready-to-run example (no data download required), see
[`src/inference/example/`](src/inference/example/).

## Data Availability

Pre-trained model weights, Sahni+Fragoza training data (post-AF3 structure filtering, as used in Fig 3 GCV), and AF3 complex structures are available on Zenodo:
- **Models + training data**: https://doi.org/10.5281/zenodo.17645488
- **AF3 structures**: https://doi.org/10.5281/zenodo.18701748

VarChAMP data was unpublished IGVF consortium data at time of release and is excluded. COSMIC and HGMD require licensed access and are not distributed. gnomAD and ClinVar data must be downloaded from their respective public portals. Full source/version details: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Further Documentation

This README covers installation and a minimal inference example. For everything else, see [`docs/`](docs/):

- **[docs/SETUP.md](docs/SETUP.md)** — where the code looks for data, the three data tiers, environment variables
- **[docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md)** — the ordered chain from published source files to canonical tables, structures and contact graphs
- **[docs/INFERENCE.md](docs/INFERENCE.md)** — detailed inference usage, file formats, full worked example, troubleshooting
- **[docs/TRAINING.md](docs/TRAINING.md)** — stability pretraining + model training from scratch
- **[notebooks/reproduce_all_figures.py](notebooks/reproduce_all_figures.py)** — one runnable, cached, resumable script/notebook (jupytext percent format) that regenerates every figure and table end to end, from the Zenodo bundle.
  **It ships with `QUICK = True`**, which finishes in hours instead of days but uses 1 cross-validation seed instead of 30 and subsampled variant databases, writing to `results_quick/`. Those are *not* the paper's numbers. Set `QUICK = False` (line 48) to reproduce the published results into `results/`.
- **[docs/REPRODUCING_ANALYSES.md](docs/REPRODUCING_ANALYSES.md)** — every figure/table in the paper (GCV benchmarking, VarChAMP blind test, variant-repository inference and enrichment analyses, supplementary figures), with exact commands
- **[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)** — where every dataset comes from and licensing notes
- **[docs/MANUSCRIPT_FIGURES.md](docs/MANUSCRIPT_FIGURES.md)** — every figure and table in the paper, the command that produces it, and what it depends on

## Project Structure

```
MutPred-PPI/
├── src/
│   ├── paths.py             # single point of path resolution — no absolute paths anywhere else
│   ├── model.py             # the one GAT_mut_processor definition
│   ├── contact_graphs.py    # ContactGraphStore: the HDF5 contact-graph store, keyed by sequence
│   ├── ids.py               # accession / variant-id parsing
│   ├── utils/               # shared layers: gcv_common, structures, mutpred_ppi_data
│   ├── inference/           # public 3-step inference pipeline
│   │   └── example/         # runnable end-to-end inference quickstart
│   ├── training/            # stability pretraining, train_fold, final-model fitting
│   ├── evaluation/          # cross-validation benchmarking + comparator methods
│   ├── data_processing/     # dataset, structure and graph preparation
│   │   └── training_sets/   # prepare_gcv_tables.py -> the GCV data layer
│   ├── variant_db_inference/# large-scale variant-DB scoring + canonical DB tables
│   └── analysis/            # figure/table generation + paper analyses
├── scripts/
│   └── link_external.sh     # builds external/ (see docs/SETUP.md)
├── notebooks/               # runnable jupytext notebooks (mapping, figure reproduction)
├── tests/                   # pytest test suite (run with: conda run -n ppi python -m pytest tests/ -q)
├── docs/                    # setup, inference, training, reproduction, provenance
├── weights/                 # model checkpoints
├── figures/                 # generated LaTeX tables (PNGs are gitignored)
├── archive/                 # superseded scripts and dataset references, kept for provenance
├── LICENSE
└── README.md

Not in git — created locally or delivered via Zenodo (see docs/SETUP.md):
  datasets/        Zenodo bundle + locally built data:
                     source_data/, source_data_restricted/  raw inputs
                     source_mapping/                        mapping outputs
                     training_eval/                         GCV tables, graphs, caches
                     variant_dbs/, af3_structures*/         repositories, structures
  external/        symlinks to large machine-local data
  results/         generated by the evaluation scripts
  results_quick/   generated by the notebook's QUICK mode
  data_caches/     optional dataset caches (regenerable)
```


Model weights and training data are distributed via Zenodo (see [Data Availability](#data-availability) above).

## License

This software is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

**Important Note:** While MutPred-PPI itself is open source, users must comply with the licensing terms of any structural data they use as input:
- **AlphaFold3 structures**: Subject to [AlphaFold3 Output Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md) (non-commercial only)
- **PDB structures**: Check individual structure licenses
- **Other sources**: Comply with respective terms

## Citation

If you use MutPred-PPI in your research, please cite:

```bibtex
@inproceedings{stewart2026mutpred-ppi,
  title={Predicting interaction-specific protein--protein interaction perturbations by missense variants with MutPred-PPI},
  author={Stewart, Ross and Laval, Florent and Coppin, Georges and Spirohn-Fitzgerald, Kerstin and Tixhon, Maxime and Hao, Tong and Calderwood, Michael A and Mort, Matthew and Cooper, David N and Vidal, Marc and Radivojac, Predrag},
  booktitle={Proceedings of the 30th Annual International Conference on Research in Computational Molecular Biology (RECOMB)},
  year={2026},
  note={Also available as bioRxiv preprint (2025.12.20.695738)},
  doi={10.64898/2025.12.20.695738}
}
```

## Contact

- **Issues**: Please open an issue on GitHub for bug reports or feature requests
- **Email**: stewart.ro@northeastern.edu
