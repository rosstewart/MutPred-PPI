# GATMutPPI

Official repository for "Predicting interaction-specific protein–protein interaction perturbations by missense variants with GATMutPPI"

## Overview

GATMutPPI is a deep learning framework that predicts whether missense mutations disrupt protein-protein interactions. It combines structural information from AlphaFold3-predicted complexes with sequence embeddings from protein language models to achieve high-accuracy predictions.

**Key Features:**
- Graph neural networks with attention mechanisms for structural analysis
- ProtT5 protein language model embeddings for sequence representation
- Binary classification: disrupted (1) vs. maintained (0) interactions
- Parallel processing support for large-scale analysis

## Installation

```bash
# Clone repository
git clone https://github.com/rosstewart/gatmutppi.git
cd gatmutppi

# Install dependencies
pip install -r src/requirements.txt
```

### System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster inference)
- 16GB+ RAM recommended
- ~10GB disk space for models and data

### Dependencies

Core dependencies from `src/requirements.txt`:
- `biopython==1.85` - Biological sequence parsing
- `torch==2.6.0` - Deep learning framework
- `torch_geometric==2.6.1` - Graph neural networks
- `transformers==4.46.0` - Protein language models
- `numpy`, `scipy`, `joblib` - Scientific computing

## Quick Start

```bash
# Step 1: Prepare AlphaFold3 inputs
python src/00_make_af3_input_files.py proteins.fasta variants.tsv af3_inputs/

# Step 2: Generate in-house AlphaFold3 structures or submit to AlphaFold3 Server and download structures (note: submitting to AlphaFold3 Server may require differently formatted input files)

# Step 3: Generate contact graphs
python src/01_make_contact_graphs_and_fasta.py working_dir/ mmcif_dir/ variants.tsv

# Step 4: Run predictions
python src/02_run_gatmutppi_inference.py working_dir/
```

## Detailed Usage

### Step 1: Prepare AlphaFold3 Input Files

Generate JSON files for AlphaFold3 Server submission:

```bash
python src/00_make_af3_input_files.py \
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

### Step 2: Generate Contact Graphs

Process AlphaFold3 structures to create residue contact graphs:

```bash
python src/01_make_contact_graphs_and_fasta.py \
    <working_dir> \
    <mmcif_dir> \
    <variants_file> \
    [n_jobs]
```

**Arguments:**
- `working_dir`: Output directory for graphs and sequences
- `mmcif_dir`: Directory containing AlphaFold3 .cif files
- `variants_file`: Same TSV file from Step 1
- `n_jobs`: Number of parallel jobs (default: 1)

**Outputs:**
- `working_dir/af3_graphs/`: Contact graph matrices (.mat and helper files)
- `working_dir/wt_and_vt.fasta`: Combined wild-type and variant sequences

### Step 3: Run GATMutPPI Inference

Predict interaction disruption for all variants:

```bash
python src/02_run_gatmutppi_inference.py \
    <working_dir> \
    [--device DEVICE]
```

**Arguments:**
- `working_dir`: Directory from Step 2 containing graphs and FASTA
- `--device`: GPU device (default: cuda:0, use 'cpu' if no GPU)

**Output:**
- `working_dir/results/GATMutPPI_preds.tsv`: Prediction scores for each input variant. Format: `(variant_partner_id, prediction_score)`

## File Formats

### Variant Notation

Variants use standard notation: `[WT_residue][position][MT_residue]`
- Example: `V123A` (Valine at position 123 to Alanine)
- Position numbering starts at 1

### AlphaFold3 Structure Files

The pipeline accepts mmCIF files with flexible naming:
- `PROT1_PROT2.cif`
- `prefix_PROT1_PROT2_suffix.mmcif`
- Case-insensitive matching supported

### Output Format

Predictions are saved as:
- Per-variant results in TSV format

## Example Workflow

Complete example using provided test data:

```bash
# Navigate to example directory
cd src/example/

# 1. Generate AlphaFold3 inputs
python ../00_make_af3_input_files.py \
    test_proteins.fasta \
    test_variants.tsv \
    af3_inputs/

# 2. Generate AlphaFold3 structures in-house or submit JSON files to AlphaFold3 Server
# Save structures to structures/

# 3. Process structures
python ../01_make_contact_graphs_and_fasta.py \
    output/ \
    structures/ \
    test_variants.tsv \
    4  # Use 4 parallel jobs

# 4. Run predictions
python ../02_run_gatmutppi_inference.py \
    output/ \
    --device cuda:0

# View results
cat output/results/predictions.csv
```

## Project Structure

```
gatmutppi/
├── src/
│   ├── 00_make_af3_input_files.py       # AlphaFold3 input preparation
│   ├── 01_make_contact_graphs_and_fasta.py  # Contact graph generation
│   ├── 02_run_gatmutppi_inference.py    # Model inference
│   ├── utils/
│   │   ├── inference_utils.py           # Core inference functions
│   │   ├── model_loader.py              # Model loading utilities
│   │   └── prott5_loader.py             # ProtT5 embedding generation
│   ├── models/                           # Pre-trained model weights
│   ├── example/                          # Example data and workflow
│   └── requirements.txt                  # Python dependencies
└── README.md
```

## Performance

- **Inference speed**: ~100 variant-partner combinations/minute on GPU (after contact graph generation)
- **Memory usage**: ~4GB GPU memory for typical complexes
- **Accuracy**: See publication for detailed benchmarks

## Troubleshooting

### Common Issues

**CUDA out of memory error:**
```bash
# Use CPU instead
python src/02_run_gatmutppi_inference.py working_dir/ --device cpu

# Or use a different GPU
python src/02_run_gatmutppi_inference.py working_dir/ --device cuda:1
```

**Missing AlphaFold3 structures:**
- Ensure structure files contain both protein IDs in filename
- Check that files have .cif or .mmcif extension
- Verify proteins IDs match between FASTA and TSV files

**Invalid amino acids in sequences:**
- Only standard 20 amino acids supported (ACDEFGHIKLMNPQRSTVWY)
- Remove non-standard residues or replace with closest standard

**Module import errors:**
```bash
# Ensure you're in the correct directory
cd gatmutppi/

# Reinstall dependencies
pip install -r src/requirements.txt --upgrade
```

## Citation

If you use GATMutPPI in your research, please cite:

```bibtex
@article{xxx,
  title={Predicting interaction-specific protein–protein interaction perturbations by missense variants with GATMutPPI},
  author={Stewart, Ross and others},
  journal={xxx},
  year={xxx},
  doi={xxxx}
}
```

## License

MIT License - see LICENSE file for details

## Contact

- **Issues**: Please open an issue on GitHub for bug reports or feature requests
- **Email**: stewart.ro@northeastern.edu
