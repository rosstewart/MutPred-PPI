# Inference Quickstart

A minimal, self-contained, real end-to-end run of the public MutPred-PPI inference
pipeline (`src/inference/`) — no external downloads, no GPU cluster, no full
datasets required. Runs in well under a minute on GPU (a few minutes on CPU).

This example builds its own tiny mmCIF structure set from
protein pairs pulled from the real Sahni+Fragoza training data, so everything needed
to run it ships inside this directory — nothing outside the repo, and no VarChAMP (unpublished) data.

## What it does

Runs the real 3-file pipeline described in `docs/INFERENCE.md`, steps 2-3
(step 1, `00_make_af3_json_input.py`, is skipped because structures are already
provided — see "Where the inputs came from" below):

1. `src/inference/01_make_contact_graphs_and_fasta.py` — builds residue contact
   graphs from the 3 mmCIF structures in `af3_models/`, using the variant list in
   `test_variants.tsv`.
2. `src/inference/02_run_mutpred-ppi_inference.py` — runs the primary MutPred-PPI
   model (`weights/MutPred-PPI.pt`) on the resulting graphs + ProtT5 embeddings.

## Inputs (all in this directory)

- `test_proteins.fasta` — full-length sequences for the 5 proteins involved (for
  reference; not required at inference time since structure sequences are read
  directly from the mmCIF files).
- `test_variants.tsv` — 3 variant/partner triplets (`protein_a  variant  protein_b`),
  one variant per complex:
  ```
  Q4ACX1   L171R   O43765
  O75603   G63S    Q96LI6
  P40259   G137S   O43765
  ```
- `af3_models/*.cif` — 3 AlphaFold3 complex structures, one per pair above, converted
  to mmCIF from the corresponding files already shipped in
  `datasets/af3_structures/*.pdb.gz` (that directory is Zenodo-downloaded / gitignored
  normally, so the 3 needed structures were extracted and converted here once so this
  example is self-contained and doesn't require a Zenodo download).

### Where the inputs came from

The 3 protein pairs/variants were taken from the Sahni+Fragoza training data
(`Q4ACX1 L171R O43765`, `O75603 G63S Q96LI6`, `P40259 G137S O43765`), chosen because
matching AlphaFold3 structures already exist in `datasets/af3_structures/`. Two of the
three are still present in the current canonical table
(`datasets/training_eval/sahni_fragoza_mapped090826_rows.csv.gz`, rows 501 and 1944);
`Q4ACX1` was dropped by the 2026-09-08 accession remapping. The example is a
self-contained format/plumbing demo -- its inputs ship inside this directory and it
does not read any dataset -- so this does not affect whether it runs. The
wild-type residue at each mutation position was verified against the structure
sequence before inclusion (all 3 match exactly).

## How to run it

```bash
conda activate ppi   # or: conda run -n ppi ...
bash src/inference/example/run_example.sh --device cuda:0
# or, with no GPU:
bash src/inference/example/run_example.sh --device cpu
```

This regenerates (into this directory, untracked so the repo stays clean):
- `af3_graphs/` — contact graphs + derived FASTA/label files (Step 2 output)
- `wt_and_vt.fasta` — combined WT/variant sequences for ProtT5 (Step 2 output)
- `results/MutPred-PPI_preds.tsv` — final predictions (Step 3 output), also copied to
  `expected_output/MutPred-PPI_preds.tsv`

## Expected output

A 3-row (+ header) TSV with columns `interactor`, `partner`, `mutation`, `score`.
`score` is a probability in `[0, 1]` — higher means a higher predicted probability
that the variant disrupts the interaction. `mutation` is 1-based.

```
interactor	partner	mutation	score
P40259	O43765	G137S	0.972222626209259
O75603	Q96LI6	G63S	0.6895588040351868
Q4ACX1	O43765	L171R	0.9620879888534546
```

(Row order may vary run-to-run — the pipeline processes complexes via `glob`, whose
order is filesystem-dependent — but the 3 rows and their scores should match, modulo
last-digit floating-point noise from GPU non-determinism.) A copy of this exact output
is saved at `expected_output/MutPred-PPI_preds.tsv` for reference.
