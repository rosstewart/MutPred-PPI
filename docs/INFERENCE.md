# Running Inference at Scale

This covers the public 3-step inference pipeline (`src/inference/`) for scoring your own variant/partner sets. For a minimal working example, see the [Quick Start](../README.md#quick-start) in the main README.

## Step 1: Prepare AlphaFold3 Input Files (Optional)

If using AlphaFold3 for structure generation, prepare JSON files for submission:

```bash
python src/inference/00_make_af3_json_input.py \
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

## Step 1.5: Obtain Protein Complex Structures

MutPred-PPI requires protein complex structures in mmCIF format. You can use structures from any source:

### Option A: AlphaFold3 Structures
Submit protein complex queries to the [AlphaFold3 Server](https://alphafoldserver.com/) using the JSON files from Step 1, or generate structures locally if you have access to AlphaFold3.

**Note:** AlphaFold3 structures are subject to AlphaFold3's Terms of Use (non-commercial use only). See their [terms](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md) for details.

### Option B: Experimental Structures
Download experimental structures from the [Protein Data Bank](https://www.rcsb.org/). Convert to mmCIF format if needed.

### Option C: Other Structure Prediction Tools
Use any other structure prediction tool that outputs mmCIF or PDB format (e.g., AlphaFold-Multimer, RoseTTAFold).

Save all mmCIF structure files to `<mmcif_dir>` for use in Step 2. **Important:** The sequences in the structure file must exactly match the sequences in the FASTA files.

## Step 2: Generate Contact Graphs

Process structures to create residue contact graphs:

```bash
python src/inference/01_make_contact_graphs_and_fasta.py \
    <working_dir> \
    <mmcif_dir> \
    <variants_file> \
    [n_jobs]
```

**Arguments:**
- `working_dir`: Output directory for graphs and sequences
- `mmcif_dir`: Directory containing structure files (.cif or .mmcif)
- `variants_file`: Same TSV file from Step 1
- `n_jobs`: Number of parallel jobs (default: `-1`, i.e. all cores). Large protein complexes may take several minutes to process.

**Outputs:**
- `working_dir/af3_graphs/contact_graphs.h5`: a `ContactGraphStore`
  ([`src/contact_graphs.py`](../src/contact_graphs.py)) — one HDF5 store for every graph,
  content-addressed rather than filename-addressed. There are no per-complex `.mat` files any more.
- `working_dir/af3_graphs/complexes.csv`: columns `complex_id, interactor, partner,
  interactor_sequence, partner_sequence`. Step 3 joins on this and looks graphs up **by
  sequence**, so it never has to split an accession out of a filename.
- `working_dir/af3_graphs/`: per-complex `.labels`, `.labels_separated`, `.num_residues_a`
  and `.interaction_loss_pos`/`_neg` helper files for the variant/FASTA steps
- `working_dir/wt_and_vt.fasta`: Combined wild-type and variant sequences for ProtT5 embedding generation

### The contact-graph store

Graphs are keyed on the sorted pair of `sha256(chain_sequence)[:16]`, so one pair of sequences
has exactly one key regardless of which accessions, filenames or chain order produced it
(`contact_graphs.pair_key`; single-chain entries get a `mono_` prefix). Accessors are
**keyword-only and take sequences, not accessions**:

```python
from contact_graphs import ContactGraphStore

with ContactGraphStore("working_dir/af3_graphs/contact_graphs.h5") as store:
    G  = store.load_dense(interactor=seq_a, partner=seq_b)       # dense adjacency
    ei = store.load_edge_index(interactor=seq_a, partner=seq_b)  # what the GAT consumes
```

Both return the graph **already oriented to the requested interactor** — the interactor occupies
nodes `[0, len(interactor))` — so orientation is never re-derived downstream from a filename
or a stored split point. Self-loops are added on read, unconditionally; they are not stored
and cannot be disabled.

An edge joins two residues when any pair of their atoms is within 4.5 Å
(`contact_graphs.contact_graph_from_structure`), which is the single contact-graph definition in
the repo.

## Step 3: Run MutPred-PPI Inference

Predict interaction disruption for all variants:

```bash
python src/inference/02_run_mutpred-ppi_inference.py \
    <working_dir> \
    [--device DEVICE]
```

**Arguments:**
- `working_dir`: Directory from Step 2 containing graphs and FASTA
- `--device`: Compute device (default: cuda:0, use 'cpu' if no GPU available)

**Output:**
- `working_dir/results/MutPred-PPI_preds.tsv`: Prediction scores for each input variant
  - Tab-separated, with headers: `interactor`, `partner`, `mutation`, `score`
    (`src/inference/pipeline/inference_utils.py::write_output`)
  - `mutation` is 1-based, matching every other table in the repo.

**Note on schema.** Both pipelines now emit the same four explicit columns; the
variant-database pipeline (`src/variant_db_inference/run_variant_db_inference.py`)
writes an identical header. The composite `complex_id` (`{interactor}_{partner}`)
this step used to write is gone: splitting such an id back on `_` mis-assigns
isoform and RefSeq accessions (`NP_002046_GFAP` has no unambiguous split).

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

`find_mmcif_file` recovers chain order from the filename here (it detects the swapped case and
reassigns chains A/B accordingly), so accessions must be splittable out of the name.

The structure trees shipped with the paper do **not** use this convention. They are canonicalized
to `{ACC_LO}__{ACC_HI}.cif.gz` — accessions uppercase, sorted, joined by a double underscore, one
gzipped mmCIF per pair, alongside a `manifest.csv`. Because the accessions are sorted, **the
filename encodes no orientation**; orientation is a property of a row and is resolved from
sequences at load time. See [`docs/REPRODUCING_ANALYSES.md`](REPRODUCING_ANALYSES.md#af3-structures).

## Full Example Workflow

A runnable end-to-end example ships in
[`src/inference/example/`](../src/inference/example/) — three protein pairs
with their AlphaFold 3 structures bundled, so it needs no download and no cluster:

```bash
conda activate ppi
bash src/inference/example/run_example.sh                # GPU
bash src/inference/example/run_example.sh --device cpu   # CPU, a few minutes
```

It runs steps 2 and 3 of the real pipeline and prints the predictions, then diffs them against
the committed `expected_output/MutPred-PPI_preds.tsv` (it does not overwrite that reference —
it only writes it if it is missing).

**This currently fails on a fresh clone.** `af3_graphs/`, `results/` and `wt_and_vt.fasta` are
gitignored inside the example, so a clone starts from a clean working directory and hits the
empty-`wt_and_vt.fasta` problem described above; step 3 then has no embeddings to read. Only a
checkout that still carries a pre-existing `af3_graphs/` from before the `.mat` removal will
complete.

**Note:** the bundled AlphaFold 3 structures are subject to the AlphaFold 3 Output Terms of
Use and are provided for non-commercial research only. See
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

### Step 1 separately (only if you are generating your own structures)

The quickstart starts from precomputed structures, so it skips step 1. To build AlphaFold 3
job inputs for your own pairs:

```bash
python src/inference/00_make_af3_json_input.py \
    src/inference/example/test_proteins.fasta \
    src/inference/example/test_variants.tsv \
    my_af3_inputs/
```

Submit those JSONs to AlphaFold 3 (or AlphaFold Server), put the returned mmCIF files in a
directory, and then run steps 2 and 3 as the quickstart does.

**Expected output** (from the quickstart's `expected_output/MutPred-PPI_preds.tsv`):

```
interactor	partner	mutation	score
P40259	O43765	G137S	0.972222626209259
O75603	Q96LI6	G63S	0.6895588040351868
Q4ACX1	O43765	L171R	0.9620879888534546
```

(`run_example.sh` compares sorted, so row order does not matter.)

## Performance

- **Inference speed**: ~100 variant-partner combinations/minute on GPU (V100) with precomputed structures
- **Memory usage**: ~4GB GPU memory for typical complexes (ProtT5 usage)

## Troubleshooting

### Common Issues

**CUDA out of memory error:**
```bash
# Use CPU instead
python src/inference/02_run_mutpred-ppi_inference.py working_dir/ --device cpu

# Or use a different GPU
python src/inference/02_run_mutpred-ppi_inference.py working_dir/ --device cuda:1
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
- Do NOT edit or remove non-standard residues; the pipeline handles them. The
  ambiguity/rare codes `B`, `U`, `Z`, `O` are folded to `X` before embedding
  (`utils.embeddings.clean_sequence`), and `MSE` (selenomethionine) is read as
  `M` when parsing structures. Nothing is ever truncated or dropped.

**Module import errors:**
```bash
# Ensure you're in the correct directory
cd mutpred-ppi/

# Reinstall dependencies (one requirements file for the whole repo)
pip install -r requirements.txt --upgrade

# The package itself must be installed for `contact_graphs`/`utils` to import
pip install -e .
```

**Conda environment issues:**
```bash
# If having package conflicts, create fresh environment
conda deactivate
conda env remove -n ppi
conda create -n ppi python=3.10 -y
conda activate ppi
# Then reinstall following Installation steps in the main README
```

## AlphaFold3 input dialects

`src/inference/00_make_af3_json_input.py` emits **either** AF3 input dialect. They are not
interchangeable — a file in one will not run under the other.

| | `--format local` (default) | `--format server` |
|---|---|---|
| target | open-source AlphaFold3 executable | AlphaFold Server web UI |
| chain key | `"protein"` | `"proteinChain"` |
| chain identity | `"id": "A"` (a bare string or a list are both accepted) | `"count": 1` |

Use `local` for the AlphaFold 3 executable you run yourself, and `server` for the
AlphaFold Server web interface. A file in one dialect will not run under the other.

Two input modes:

```bash
# FASTA + triplet TSV
python src/inference/00_make_af3_json_input.py <fasta> <triplets.tsv> <out_dir>

# a rows CSV carrying sequences inline (interactor, partner, mutation columns)
python src/inference/00_make_af3_json_input.py --csv rows.csv <out_dir> [--format server]
```

**Non-standard residues are substituted, not rejected.** AF3 accepts only the 20 standard letters
inside a `sequence` string, so `U` (selenocysteine) becomes `C` and `O` (pyrrolysine) becomes `K` —
each is structurally near-identical to its replacement at the resolution AF3 models, and every
substitution is logged. Ambiguity codes (`BJXZ`) have no sensible substitute and still fail loudly.
The canonical training data affects exactly one protein, `P59797` (one `U` in 346 aa).

**Isoform accessions are preserved.** Hyphens are sanitised to underscores in the *filename* only.
Do not canonicalise a trailing `-1`: the mapping keeps a suffix only where the isoform sequence
genuinely differs from canonical, and both `Q9BRI3-1` and bare `Q9BRI3` are present — collapsing
them pairs an accession with the wrong sequence.
