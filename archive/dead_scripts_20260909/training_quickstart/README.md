# Training Quickstart

Builds the MutPred-PPI training inputs -- contact graphs, pos/neg labels and
ProtT5 embeddings -- for a tiny (40-row) real subset of the Sahni+Fragoza
training data, using the real public pipeline scripts. No external downloads
required. About 1.5 minutes end to end on GPU (ProtT5 embedding generation
dominates); a few minutes on CPU.

## The training step is retired

This example used to end with `train_on_subset.py`, which drove
`mutpred_ppi_cv.py`'s per-source loader helpers (`_build_emb_dict`,
`_load_graphs`, `_gather_labels_pos_neg`). Those were removed when the loaders
were replaced by the canonical tables and `utils/mutpred_ppi_data.build_tensors`,
so the script no longer ran; it is kept in `archive/dead_scripts_20260909/`.

It was not rewritten against the canonical tables because that would make a
deliberately self-contained example depend on `datasets/mapped090826/` -- the
rows tables, the sequence-keyed contact-graph store and the ProtT5 caches --
none of which ship with a fresh clone. To actually train, use
`src/training/train_final_model.py`, which reads those tables directly.

## What it does

1. `sahni_fragoza_train_subset.csv` — 40 real rows (15 disrupted / 25
   maintained) from `datasets/train_eval/sahni_fragoza_train.csv`, selected
   because AlphaFold3 structures for their complexes already exist in
   `datasets/af3_structures/` and the wild-type residue at each mutation
   position was verified against the structure sequence.
2. `af3_models/*.cif` — the 40 corresponding structures, converted to mmCIF
   from `datasets/af3_structures/*.pdb.gz` (self-contained here so this
   example doesn't require a Zenodo download of `datasets/`).
3. `run_example.sh` runs, in order:
   - **Step 1**: `src/inference/01_make_contact_graphs_and_fasta.py`
     (real, unmodified public pipeline script) — builds contact graphs +
     `wt_and_vt.fasta` from `af3_models/` + `train_variants.tsv`.
   - **Step 2**: `fix_labels.py` — splits the position-only
     `.interaction_loss_pos` files that script wrote into true
     `.interaction_loss_pos` (disrupted) / `.interaction_loss_neg`
     (maintained) files, using `Y2H_score` from
     `sahni_fragoza_train_subset.csv` (01's script has no notion of a true
     label -- it only records which position is mutated).
   - **Step 3**: `generate_embeddings.py` — real ProtT5 loader utilities
     (`src/inference/utils/prott5_loader.py`), batched once over the whole
     small FASTA.

## How to run it

```bash
conda activate ppi   # or: conda run -n ppi ...
bash examples/training_quickstart/run_example.sh --device cuda:0
# or, with no GPU:
bash examples/training_quickstart/run_example.sh --device cpu
```

This regenerates (into this directory, untracked so the repo stays clean):
`af3_graphs/`, `wt_and_vt.fasta` and `wt_and_vt_t5_embs.h5`.

## Expected output

`af3_graphs/` populated from the 40 mmCIF structures, `wt_and_vt.fasta` with
one wild-type and one variant record per row, and `wt_and_vt_t5_embs.h5`
holding an (L, 1024) per-residue ProtT5 embedding per FASTA record.
