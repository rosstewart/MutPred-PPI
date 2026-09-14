# MutPred-PPI

MutPred-PPI scores how likely a missense variant is to disrupt a particular
protein-protein interaction. It combines a graph attention network over the predicted
structure of the complex with ProtT5 sequence embeddings, and returns one probability per
(protein, variant, partner) triple: 1 means the interaction is likely disrupted, 0 that it
is likely preserved.

- **Web server**: <https://mutpred.mutdb.org/mutpredppi>
- **Paper**: [doi:10.64898/2025.12.20.695738](https://doi.org/10.64898/2025.12.20.695738),
  RECOMB 2026 ([proceedings](https://recomb.org/proceedings/proceedings/2030-2026/2026/))
- **Data**: see [`docs/DATA.md`](docs/DATA.md) <!-- DOI: pending -->

Upload structures and a list of missense variants to the web server to score them without
installing anything. Install locally for large variant sets, or to retrain.

## Installation

```bash
git clone https://github.com/rosstewart/mutpred-ppi.git
cd mutpred-ppi

conda create -n ppi python=3.10 -y
conda activate ppi

# requirements.txt pins the published environment, including torch==2.5.1. Install
# torch first if you need a particular CUDA build:
#   CUDA 12.1:  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
#   CPU only:   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

Cross-validation also needs `cd-hit`, which supplies the sequence-identity clusters used for
grouping. It is a C++ binary with no PyPI package:

```bash
conda install -c bioconda cd-hit -y
```

Python 3.10 or later, 16 GB RAM, and a CUDA GPU for anything beyond the example.

## Quick start

To score a small number of variants, use the
[web server](https://mutpred.mutdb.org/mutpredppi) instead of installing anything.

We include an end-to-end inference example with precomputed structures. No download needed:

```bash
bash src/inference/example/run_example.sh --device cpu
```

To score your own variants:

```bash
# 1. Prepare AlphaFold 3 inputs for each complex (skip if you already have structures)
python src/inference/00_make_af3_json_input.py proteins.fasta variants.tsv af3_inputs/

# 2. Fold them, with AlphaFold Server or a local AlphaFold 3 install

# 3. Build contact graphs
python src/inference/01_make_contact_graphs_and_fasta.py working_dir/ mmcif_dir/ variants.tsv

# 4. Score
python src/inference/02_run_mutpred-ppi_inference.py working_dir/
```

File formats and a worked example: [`docs/INFERENCE.md`](docs/INFERENCE.md).

## Reproducing the paper

Download the data ([`docs/DATA.md`](docs/DATA.md)), then:

```bash
python notebooks/reproduce_all_figures.py
```

A Jupytext-format notebook that reproduces all analyses from the paper. `QUICK = True`
(line 52) by default to reduce computation, using 1 cross-validation seed instead of 30 and
subsampled variant repositories, and writing to `results_quick/`. Set `QUICK = False` for the
published numbers in `results/`.

### What you can reproduce

VarChAMP data was unpublished at the time of writing and will be cross-linked from the IGVF
portal ([data.igvf.org](https://data.igvf.org)) on release. Until then Fig 4, S3 and Table 1
cannot be reproduced. COSMIC and HGMD analyses need a licence for the underlying data; the
panels that use them are skipped without it, and the rest of each figure is drawn.

Everything else reproduces from the deposit: Fig 3, Fig 5, and S1, S2, S4 through S10.

Dataset versions and licensing: [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md).

## Documentation

| | |
|---|---|
| [`docs/DATA.md`](docs/DATA.md) | what is deposited, how to unpack it, where it goes |
| [`docs/INFERENCE.md`](docs/INFERENCE.md) | the inference pipeline, file formats, troubleshooting |
| [`docs/TRAINING.md`](docs/TRAINING.md) | stability pretraining and model fitting |
| [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md) | dataset provenance and how to rebuild from source |
| [`docs/REPRODUCING_ANALYSES.md`](docs/REPRODUCING_ANALYSES.md) | every figure and table, with the command that produces it |

## Repository layout

```
src/
  paths.py              path resolution
  model.py              the GAT_mut_processor definition
  contact_graphs.py     the HDF5 contact-graph store, keyed by sequence
  ids.py                accession and variant-id parsing
  utils/                shared data layer
  inference/            the inference pipeline
    example/            runnable example
  training/             stability pretraining and model fitting
  evaluation/           cross-validation, blind test, comparator methods
  data_processing/      dataset, structure and graph preparation
  variant_db_inference/ variant-repository scoring
  analysis/             figures, tables and analyses
  verification/         consistency checks over the built artifacts
notebooks/              mapping and figure-reproduction notebooks
tests/                  pytest suite
docs/
figures/                generated figures and tables
weights/                model checkpoints
```

`datasets/`, `results/`, `external/` and `external_methods/` are created locally or
downloaded; see [`docs/DATA.md`](docs/DATA.md).

## Licence

MIT, see [LICENSE](LICENSE). AlphaFold 3 predictions are subject to the
[AlphaFold 3 output terms of use](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md)
(non-commercial only); check the licence of any other structure source you use.

## Citation

```bibtex
@inproceedings{stewart2026mutpred-ppi,
  title={Predicting interaction-specific protein--protein interaction perturbations by missense variants with MutPred-PPI},
  author={Stewart, Ross and Laval, Florent and Coppin, Georges and Spirohn-Fitzgerald, Kerstin and Tixhon, Maxime and Hao, Tong and Calderwood, Michael A and Mort, Matthew and Cooper, David N and Vidal, Marc and Radivojac, Predrag}
  booktitle={Proceedings of the 30th Annual International Conference on Research in Computational Molecular Biology (RECOMB)},
  year={2026},
  doi={10.64898/2025.12.20.695738}
}
```

## Contact

GitHub issues, or stewart.ro@northeastern.edu.
