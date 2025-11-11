# GATMutPPI

Official repository for "Predicting interaction-specific protein–protein interaction perturbations by missense variants with GATMutPPI"

## Overview

GATMutPPI is a deep learning framework that predicts whether missense mutations disrupt protein-protein interactions. It combines structural information from protein complexes with sequence embeddings from protein language models to achieve high-accuracy predictions.

**Key Features:**
- Graph neural networks with attention mechanisms for structural analysis
- ProtT5 protein language model embeddings for sequence representation
- Binary classification: probabilistic score ranging from 0-1, where 1 indicates high probability of interaction disruption and 0 indicates preserved interaction
- Parallel processing support for large-scale analysis

## Installation

```bash
# Clone repository
git clone https://github.com/rosstewart/gatmutppi.git
cd gatmutppi

# Create conda environment
conda create -n gatmutppi python=3.8 -y
conda activate gatmutppi

# Install PyTorch with CUDA support (for GPU)
conda install pytorch==2.6.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Or install PyTorch for CPU only
# conda install pytorch==2.6.0 cpuonly -c pytorch -y

# Install remaining dependencies
pip install -r src/requirements.txt
```

### System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster inference)
- 16GB+ RAM recommended
- ~10GB disk space for models and data

### Dependencies

Core dependencies from `src/requirements.txt`:
- `biopython==1.85`
- `torch==2.6.0`
- `torch_geometric==2.6.1`
- `transformers==4.46.0`
- `numpy`, `scipy`, `joblib`


## Quick Start

```bash
# Step 1: Prepare AlphaFold3 inputs (if using AlphaFold3)
python src/00_make_af3_input_files.py proteins.fasta variants.tsv af3_inputs/

# Step 2: Obtain protein complex structures (see Step 1.5 below)

# Step 3: Generate contact graphs
python src/01_make_contact_graphs_and_fasta.py working_dir/ mmcif_dir/ variants.tsv

# Step 4: Run predictions
python src/02_run_gatmutppi_inference.py working_dir/
```

## Data Availability

Datasets used in experiments are freely available on Zenodo: [https://doi.org/10.5281/zenodo.17409377](https://doi.org/10.5281/zenodo.17409377)

## Detailed Usage

### Step 1: Prepare AlphaFold3 Input Files (Optional)

If using AlphaFold3 for structure generation, prepare JSON files for submission:

```bash
python src/00_make_af3_json_input.py \
    <fasta_file> \
    <triplet_tsv> \
    <output_directory> \
    [--seeds N]
```

**Arguments:**
- `fasta_file`: FASTA file containing all protein sequences
- `triplet_tsv`: TSV file with columns (protein_a, variant, protein_b)
- `output_directory`: Where to save JSON files
- `--seeds`: Number of AlphaFold3 model seeds (1-5, default: 1)

**Input Format:**

FASTA file:
```
>PROT1
MKTLLILAVVAAALA...
>PROT2
MSEQNNTEMTFQIQR...
```

TSV file:
```
PROT1	V123A	PROT2
PROT1	G456D	PROT3
PROT2	W89R	PROT3
```

### Step 1.5: Obtain Protein Complex Structures

GATMutPPI requires protein complex structures in mmCIF format. You can use structures from any source:

#### Option A: AlphaFold3 Structures
Submit protein complex queries to the [AlphaFold3 Server](https://alphafoldserver.com/) using the JSON files from Step 1, or generate structures locally if you have access to AlphaFold3. 

**Note:** AlphaFold3 structures are subject to AlphaFold3's Terms of Use (non-commercial use only). See their [terms](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md) for details.

#### Option B: Experimental Structures
Download experimental structures from the [Protein Data Bank](https://www.rcsb.org/). Convert to mmCIF format if needed.

#### Option C: Other Structure Prediction Tools
Use any other structure prediction tool that outputs mmCIF or PDB format (e.g., AlphaFold-Multimer, RoseTTAFold).

Save all mmCIF structure files to `<mmcif_dir>` for use in Step 2. **Important:** The sequences in the structure file must exactly match the sequences in the FASTA files.

### Step 2: Generate Contact Graphs

Process structures to create residue contact graphs:

```bash
python src/01_make_contact_graphs_and_fasta.py \
    <working_dir> \
    <mmcif_dir> \
    <variants_file> \
    [n_jobs]
```

**Arguments:**
- `working_dir`: Output directory for graphs and sequences
- `mmcif_dir`: Directory containing structure files (.cif or .mmcif)
- `variants_file`: Same TSV file from Step 1
- `n_jobs`: Number of parallel jobs (default: 1). Parallel processing is recommended for large-scale analysis. Large protein complexes may take several minutes to process.

**Outputs:**
- `working_dir/af3_graphs/`: Contact graph matrices (.mat and helper files)
- `working_dir/wt_and_vt.fasta`: Combined wild-type and variant sequences for ProtT5 embedding generation

### Step 3: Run GATMutPPI Inference

Predict interaction disruption for all variants:

```bash
python src/02_run_gatmutppi_inference.py \
    <working_dir> \
    [--device DEVICE]
```

**Arguments:**
- `working_dir`: Directory from Step 2 containing graphs and FASTA
- `--device`: Compute device (default: cuda:0, use 'cpu' if no GPU available)

**Output:**
- `working_dir/results/GATMutPPI_preds.tsv`: Prediction scores for each input variant
  - Format: `variant_partner_id\tprediction_score`
  - Example: `PROT1_V123A_PROT2\t0.87`

## File Formats

### Variant Notation

Variants use standard notation: `[WT_residue][position][MT_residue]`
- Example: `V123A` (Valine at position 123 to Alanine)
- Position numbering starts at 1
- Use single-letter amino acid codes

### Structure Files

The pipeline accepts mmCIF files with flexible naming:
- `PROT1_PROT2.cif`
- `prefix_PROT1_PROT2_suffix.mmcif`
- Case-insensitive matching supported

### Output Format

Predictions are tab-separated values (TSV):
- Column 1: Variant-partner identifier
- Column 2: Disruption probability (0-1)

## Example Workflow

A complete minimal example using 10 test variants is provided in `src/example/`.

**Note:** Example AlphaFold3 structures are subject to AlphaFold 3 Output Terms of Use and provided for non-commercial research only. See: https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

```bash
# Activate conda environment (if using conda)
conda activate gatmutppi

# Navigate to the src/ directory
cd src/

# 1. Generate AlphaFold3 inputs (if using AlphaFold3)
python 00_make_af3_json_input.py \
    example/test_proteins.fasta \
    example/test_variants.tsv \
    example/

# NOTE: For this example, structure generation is already done
# Example structure: example/af3_models/fold_o00548_p46531_model_0.cif

# 2. Process structures to generate contact graphs
python 01_make_contact_graphs_and_fasta.py \
    example/ \
    example/af3_models/ \
    example/test_variants.tsv \
    4  # Use 4 parallel jobs

# 3. Run predictions
python 02_run_gatmutppi_inference.py \
    example/ \
    --device cuda:0

# View results
cat example/results/GATMutPPI_preds.tsv
```

**Expected output format:**
```
O00548_A653T_P46531	0.085
O00548_R661S_P46531	0.068
O00548_N34I_P46531	0.354
...
```

## Project Structure

```
gatmutppi/
├── src/
│   ├── 00_make_af3_json_input.py       # AlphaFold3 input preparation
│   ├── 01_make_contact_graphs_and_fasta.py  # Contact graph generation
│   ├── 02_run_gatmutppi_inference.py    # Model inference
│   ├── utils/
│   │   ├── inference_utils.py           # Core inference functions
│   │   ├── model_loader.py              # Model loading utilities
│   │   └── prott5_loader.py             # ProtT5 embedding generation
│   ├── models/                          # Pre-trained model weights
│   ├── example/                         # Example data and workflow
│   └── requirements.txt                 # Python dependencies
├── LICENSE                               # MIT License
└── README.md
```

## Performance

- **Inference speed**: ~100 variant-partner combinations/minute on GPU (V100) with precomputed structures
- **Memory usage**: ~4GB GPU memory for typical complexes
- **Accuracy**: AUC 0.85 (seen proteins), 0.72 (unseen proteins) - see publication for detailed benchmarks

## Troubleshooting

### Common Issues

**CUDA out of memory error:**
```bash
# Use CPU instead
python src/02_run_gatmutppi_inference.py working_dir/ --device cpu

# Or use a different GPU
python src/02_run_gatmutppi_inference.py working_dir/ --device cuda:1
```

**Missing structures:**
- Ensure structure files contain both protein IDs in filename
- Check that files have .cif or .mmcif extension
- Verify protein IDs match between FASTA and TSV files

**Sequence mismatch errors:**
- Ensure sequences in structure files exactly match FASTA sequences
- Check for missing or extra residues in structure files
- Verify correct protein pairing in filenames

**Invalid amino acids in sequences:**
- Only standard 20 amino acids supported (ACDEFGHIKLMNPQRSTVWY)
- Remove non-standard residues or replace with closest standard amino acid

**Module import errors:**
```bash
# Ensure you're in the correct directory
cd gatmutppi/

# Reinstall dependencies
pip install -r src/requirements.txt --upgrade
```

**ProtT5 loading issues:**
- First run will download ProtT5 model (~2GB)
- Ensure stable internet connection
- Model is cached locally after first download

**Conda environment issues:**
```bash
# If having package conflicts, create fresh environment
conda deactivate
conda env remove -n gatmutppi
conda create -n gatmutppi python=3.8 -y
conda activate gatmutppi
# Then reinstall following Installation steps
```

## License

This software is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

**Important Note:** While GATMutPPI itself is open source, users must comply with the licensing terms of any structural data they use as input:
- **AlphaFold3 structures**: Subject to [AlphaFold3 Output Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md) (non-commercial only)
- **PDB structures**: Check individual structure licenses
- **Other sources**: Comply with respective terms

## Citation

If you use GATMutPPI in your research, please cite:

```bibtex
@article{stewart2025gatmutppi,
  title={Predicting interaction-specific protein–protein interaction perturbations by missense variants with GATMutPPI},
  author={Stewart, Ross and Laval, Florent and Calderwood, Michael A and Vidal, Marc and Starita, Lea M and Fowler, Douglas M and Radivojac, Predrag},
  journal={[Journal TBD]},
  year={2025},
  doi={[DOI TBD]}
}
```

## Contact

- **Issues**: Please open an issue on GitHub for bug reports or feature requests
- **Email**: stewart.ro@northeastern.edu
