# GATMutPPI

Official repository for "Predicting interaction-specific protein–protein interaction perturbations by missense variants with GATMutPPI"

## Overview

GATMutPPI is a deep learning framework that predicts whether missense mutations disrupt protein-protein interactions. It combines structural information from protein complexes with sequence embeddings from protein language models to achieve high-accuracy predictions.

**Key Features:**
- Graph neural networks with attention mechanisms for structural analysis
- ProtT5 protein language model embeddings for sequence representation
- Binary classification: probabilistic score ranging from [0-1], disrupted (1) vs. maintained (0) interactions
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

Datasets used in experiments are freely available on Zenodo [here](https://doi.org/10.5281/zenodo.17409377).

## Detailed Usage

### Step 1: Prepare AlphaFold3 Input Files (Optional)

If using AlphaFold3 for structure generation, prepare JSON files for submission:

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

### Step 1.5: Obtain Protein Complex Structures

GATMutPPI requires protein complex structures in mmCIF format. You can use structures from any source:

#### Option A: AlphaFold3 Structures
Submit protein complex queries to the [AlphaFold3 Server](https://alphafoldserver.com/) using the JSON files from Step 1, or generate structures locally if you have access to AlphaFold3. 

**Note:** AlphaFold3 structures are subject to AlphaFold3's Terms of Use (non-commercial use only). See their [terms](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md) for details.

#### Option B: Experimental Structures
Download experimental structures from the [Protein Data Bank](https://www.rcsb.org/). Convert to mmCIF format if needed.

#### Option C: Other Structure Prediction Tools
Use any other structure prediction tool that outputs mmCIF or PDB format.

Save all mmCIF structure files to `<mmcif_dir>` for use in Step 2. NOTE: The sequences in the structure file must exactly match the sequences in the FASTA files.

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

### Structure Files

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

# 1. Generate AlphaFold3 inputs (if using AlphaFold3)
python ../00_make_af3_input_files.py \
    test_proteins.fasta \
    test_variants.tsv \
    af3_inputs/

# 2. Obtain structures (from AlphaFold3 Server, PDB, or other source)
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
cat output/results/GATMutPPI_preds.tsv
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
│   ├── models/                          # Pre-trained model weights
│   ├── example/                         # Example data and workflow
│   └── requirements.txt                 # Python dependencies
├── LICENSE                               # MIT License
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

**Missing structures:**
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

## License

This software is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

**Important Note:** While GATMutPPI itself is open source, users must comply with the licensing terms of any structural data they use as input:
- If using AlphaFold3 structures: Subject to [AlphaFold3 Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md) (non-commercial only)
- If using PDB structures: Check individual structure licenses
- Other structure sources: Comply with respective terms

## Citation

If you use GATMutPPI in your research, please cite:

```bibtex
@article{xxx,
  title={Predicting interaction-specific protein–protein interaction perturbations by missense variants with GATMutPPI},
  author={Stewart, Ross and Laval, Florent and Calderwood, Michael A and Vidal, Marc and Starita, Lea M and Fowler, Douglas M and Radivojac, Predrag},
  journal={[xxx]},
  year={xxx},
  doi={[DOI]}
}
```

## Contact

- **Issues**: Please open an issue on GitHub for bug reports or feature requests
- **Email**: stewart.ro@northeastern.edu
