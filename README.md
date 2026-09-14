# MutPred-PPI

Predicting interaction-specific protein–protein interaction perturbations by missense
variants.

MutPred-PPI scores how likely a missense variant is to disrupt a *specific* protein–protein
interaction. It combines a graph attention network over the predicted structure of the
complex with ProtT5 sequence embeddings, and returns one probability per
(protein, partner, variant) triple: 1 means the interaction is likely disrupted, 0 that it is
likely preserved.

This repository contains the model, the full analysis pipeline behind the paper, and a
runnable inference example.

- **Paper** — Predicting interaction-specific protein–protein interaction perturbations by
  missense variants with MutPred-PPI, RECOMB 2026.
  [doi:10.64898/2025.12.20.695738](https://doi.org/10.64898/2025.12.20.695738)
- **Proceedings** — [RECOMB 2026](https://recomb.org/proceedings/proceedings/2030-2026/2026/)
- **Data** — deposited on Zenodo; see [`docs/ZENODO.md`](docs/ZENODO.md)

## Installation

```bash
git clone https://github.com/rosstewart/mutpred-ppi.git
cd mutpred-ppi

conda create -n ppi python=3.10 -y
conda activate ppi
conda install sentencepiece -c conda-forge -y

# requirements.txt pins the exact published environment, including torch==2.5.1.
# Install torch FIRST if you need a particular CUDA build, or the pinned PyPI
# wheel will be installed over the top of it:
#   CUDA 12.1:  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
#   CPU only:   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Install the repository as a package. pyproject.toml makes src/ the package
# root, so modules import as `training.train_fold`, `utils.gcv_common`,
# `contact_graphs`, and so on. Scripts will not import without this.
pip install -e .
```

Cross-validation additionally needs `cd-hit`, which supplies the sequence-identity clusters
used for grouping:

```bash
conda install -c bioconda cd-hit -y
```

Check the install:

```bash
python -m pytest tests/ -q
```

**Requirements:** Python 3.10+, 16 GB RAM. A CUDA GPU is recommended for inference and
required in practice for training.

## Quick start

A complete worked example ships with the repository — three protein pairs, three variants,
bundled structures, and the model weights. No download needed:

```bash
bash src/inference/example/run_example.sh --device cpu
```

It runs the real three-step pipeline and compares its output against a committed reference,
so it doubles as an installation check. A few minutes on CPU, well under one on GPU.

To score your own variants:

```bash
# 1. Prepare AlphaFold 3 inputs for each complex (skip if you already have structures)
python src/inference/00_make_af3_json_input.py proteins.fasta variants.tsv af3_inputs/

# 2. Fold them (AlphaFold Server or a local AlphaFold 3 install)

# 3. Build contact graphs from the structures
python src/inference/01_make_contact_graphs_and_fasta.py working_dir/ mmcif_dir/ variants.tsv

# 4. Score
python src/inference/02_run_mutpred-ppi_inference.py working_dir/
```

Input and output formats, and a full worked example, are in
[`docs/INFERENCE.md`](docs/INFERENCE.md).

## Reproducing the paper

Download the analysis layer from Zenodo ([`docs/ZENODO.md`](docs/ZENODO.md)), then:

```bash
python notebooks/reproduce_all_figures.py
```

One resumable script (jupytext percent format, so it opens as a notebook) that regenerates
every figure and table in order, skipping work whose output is already present.

It ships with **`QUICK = True`** (line 50), which finishes in hours rather than days by using
1 cross-validation seed instead of 30 and subsampled variant repositories, writing to
`results_quick/`. **Those are not the paper's numbers.** Set `QUICK = False` to reproduce the
published results into `results/`.

### What you can reproduce

Two datasets are not ours to redistribute. Their effect is uneven, so it is worth being
explicit:

| | With the Zenodo deposit | Also with a COSMIC + HGMD licence | Not reproducible |
|---|---|---|---|
| **Figures** | Fig 3, Fig 5, S1, S3, S7, S8, S9, and the protein-class, stability, robustness and bi-class supplements | the COSMIC and HGMD panels within Fig 5, S8 and the stability figures | **Fig 4, S2, Table 1** |
| **Why** | — | COSMIC v101 and HGMD Professional 2025 require a licence | depend on VarChAMP interaction measurements, unpublished IGVF consortium data at time of release |

The VarChAMP restriction also means the published all-data model cannot be retrained. So the
deposit includes a **Sahni+Fragoza demonstration model**: the notebook falls back to it
automatically, letting you run the entire variant-repository pipeline end to end and see what
it produces. Its scores are not the published ones — it writes to a separate results tree and
stamps every figure it draws.

Full source and licence details for every dataset: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Documentation

| | |
|---|---|
| [`docs/ZENODO.md`](docs/ZENODO.md) | what is deposited, what is not, and how to unpack it |
| [`docs/SETUP.md`](docs/SETUP.md) | where the code expects data to live; environment variables |
| [`docs/INFERENCE.md`](docs/INFERENCE.md) | the three-step pipeline in detail; file formats; troubleshooting |
| [`docs/TRAINING.md`](docs/TRAINING.md) | stability pretraining and model fitting from scratch |
| [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) | provenance, versions and licensing for every dataset |
| [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md) | rebuilding the datasets from the original source files |
| [`docs/REPRODUCING_ANALYSES.md`](docs/REPRODUCING_ANALYSES.md) | every figure and table, with the command that produces it |

## Repository layout

```
src/
  paths.py              single point of path resolution
  model.py              the GAT_mut_processor definition
  contact_graphs.py     the HDF5 contact-graph store, keyed by sequence
  ids.py                accession and variant-id parsing
  utils/                shared data layer (gcv_common, structures, mutations)
  inference/            the public three-step pipeline
    example/            runnable quickstart
  training/             stability pretraining, per-fold and final-model fitting
  evaluation/           cross-validation, blind test, comparator methods
  data_processing/      dataset, structure and graph preparation
  variant_db_inference/ large-scale variant-repository scoring
  analysis/             figures, tables and paper analyses
  verification/         standalone consistency checks over the built artifacts
notebooks/              mapping and figure-reproduction notebooks (jupytext)
tests/                  pytest suite
docs/                   setup, inference, training, reproduction, deposit
figures/                generated figures and tables (output only)
weights/                model checkpoints
```

Created locally or downloaded from Zenodo, and not in git: `datasets/`, `results/`,
`external/`, `external_methods/`. See [`docs/SETUP.md`](docs/SETUP.md).

## Licence

MIT — see [LICENSE](LICENSE).

Structural inputs carry their own terms. AlphaFold 3 predictions are subject to the
[AlphaFold 3 output terms of use](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md)
(non-commercial only); check the licence of any other structure source you use.

## Citation

```bibtex
@inproceedings{stewart2026mutpred-ppi,
  title={Predicting interaction-specific protein--protein interaction perturbations by missense variants with MutPred-PPI},
  author={Stewart, Ross and Laval, Florent and Coppin, Georges and Spirohn-Fitzgerald, Kerstin and Tixhon, Maxime and Hao, Tong and Lambourne, Luke and Calderwood, Michael A and Mort, Matthew and Cooper, David N and Vidal, Marc and Radivojac, Predrag},
  booktitle={Proceedings of the 30th Annual International Conference on Research in Computational Molecular Biology (RECOMB)},
  year={2026},
  doi={10.64898/2025.12.20.695738}
}
```

## Contact

Bug reports and questions: please open a GitHub issue.
Correspondence: stewart.ro@northeastern.edu
