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

# Install PyTorch
# For GPU with CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

`requirements.txt` pins the exact environment the published results were produced in
(`src/inference/requirements.txt` is a narrower inference-only subset). Cross-validation
additionally needs `cd-hit`, which supplies the clustering groups:

```bash
conda install -c bioconda cd-hit -y
```

See [`docs/SETUP.md`](docs/SETUP.md) for where the code expects data to live.

### System Requirements

- Python 3.9 or higher
- CUDA-capable GPU (recommended for faster inference)
- 16GB+ RAM recommended
- ~4GB disk space for models and data


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
[`examples/inference_quickstart/`](examples/inference_quickstart/).

## Data Availability

Pre-trained model weights, Sahni+Fragoza training data (post-AF3 structure filtering, as used in Fig 3 GCV), and AF3 complex structures are available on Zenodo:
- **Models + training data**: https://doi.org/10.5281/zenodo.17645488
- **AF3 structures**: https://doi.org/10.5281/zenodo.18701748

VarChAMP data was unpublished IGVF consortium data at time of release and is excluded. COSMIC and HGMD require licensed access and are not distributed. gnomAD and ClinVar data must be downloaded from their respective public portals. Full source/version details: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Further Documentation

This README covers installation and a minimal inference example. For everything else, see [`docs/`](docs/):

- **[docs/SETUP.md](docs/SETUP.md)** — where the code looks for data, the three data tiers, environment variables
- **[docs/INFERENCE.md](docs/INFERENCE.md)** — detailed inference usage, file formats, full worked example, troubleshooting
- **[docs/TRAINING.md](docs/TRAINING.md)** — stability pretraining + model training from scratch
- **[docs/REPRODUCING_ANALYSES.md](docs/REPRODUCING_ANALYSES.md)** — every figure/table in the paper (GCV benchmarking, VarChAMP blind test, variant-repository inference and enrichment analyses, supplementary figures), with exact commands
- **[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)** — where every dataset comes from and licensing notes
- **[docs/METHOD_PROVENANCE.md](docs/METHOD_PROVENANCE.md)** — every comparison method's upstream commit and each way our usage deviates from it

## Project Structure

```
MutPred-PPI/
├── src/
│   ├── paths.py             # single point of path resolution — no absolute paths anywhere else
│   ├── model.py             # the one GAT_mut_processor definition
│   ├── inference/           # public 3-step inference pipeline
│   ├── training/            # stability pretraining + fine-tuning
│   ├── evaluation/          # cross-validation benchmarking + comparator methods
│   ├── data_processing/     # dataset and variant-repository preparation
│   ├── variant_db_inference/# large-scale variant-DB scoring
│   └── analysis/            # figure/table generation + paper analyses
├── scripts/
│   └── link_external.sh     # builds external/ (see docs/SETUP.md)
├── examples/                # two runnable end-to-end quickstarts
├── docs/                    # setup, inference, training, reproduction, provenance
├── weights/                 # model checkpoints
├── figures/                 # generated LaTeX tables (PNGs are gitignored)
├── LICENSE
└── README.md

Not in git — created locally or delivered via Zenodo (see docs/SETUP.md):
  datasets/        Zenodo bundle: CV reference, annotations, structures
  external/        symlinks to large machine-local data
  results/         generated by the evaluation scripts
  results_revisions/  generated: per-figure prediction/label artifacts
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
