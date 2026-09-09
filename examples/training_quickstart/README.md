# Training Quickstart

A minimal, self-contained, real end-to-end run of the MutPred-PPI training
pipeline: trains the real `GAT_mut_processor` architecture from scratch for
1 epoch on a tiny (40-row) real subset of the Sahni+Fragoza training data,
and verifies a `.pt` checkpoint is produced. No external downloads required.
Takes about 1.5 minutes end to end on GPU (ProtT5 embedding generation
dominates); a few minutes on CPU.

## Known issues surfaced by this example (please read before running)

`src/training/train_final_model.py` — the script this example was asked to
exercise — **cannot currently be run at all**, for any dataset, ablation, or
CLI combination:

1. **Broken import.** `train_final_model.py` (lines 41-53) does
   `from mutpred_ppi_cv import (..., _SCALER_PATH, ..., _PRETRAINED_PATH, ...)`,
   but `src/evaluation/mutpred_ppi_cv.py` does not define those names — it
   defines `_V1_0_SCALER_PATH` / `_V1_0_PRETRAINED_PATH` and
   `_MEGASCALE_SCALER_PATH` / `_MEGASCALE_PRETRAINED_PATH` instead. Just
   `import train_final_model` raises
   `ImportError: cannot import name '_SCALER_PATH' from 'mutpred_ppi_cv'`.
   This looks like a leftover from a rename and is independent of anything in
   this example.
2. **No CLI/data override, and hardcoded internal paths.** Even if the
   import worked, `train_final_model.py`'s only data entry point,
   `load_dataset(cfg)`, dispatches to loaders in `mutpred_ppi_cv.py`
   (e.g. `load_sahni_fragoza()`) that hardcode absolute paths to this
   machine's internal, non-Zenodo-distributed pre-cached graphs/embeddings
   (e.g. `/data/ross/ppi_lossgain/interaction_loss/swing_train`), and
   `align_to_vt_ids()` requires a canonical `all_vt_ids` pickle from yet
   another hardcoded internal path
   (`/home/rcstewart/gnn/ppi_interaction_loss/cv_splits`). Neither has a
   config flag or env var to point elsewhere, and both paths are outside
   this repo and not part of the public release/Zenodo archive — so there is
   **no way**, even after fixing bug (1), to point `train_final_model.py` at
   a custom CSV, or to run it standalone from a fresh clone of this repo.
3. **CD-HIT dependency in the wrong conda env.** The real pipeline's
   sequence-identity clustering (`cluster_sequences()` in `mutpred_ppi_cv.py`,
   used for `GroupKFold` grouping) shells out to
   `/home/rcstewart/miniconda3/envs/pytorch_env/bin/cd-hit` — a different,
   personal conda environment's binary, not present in this repo's `ppi` env
   used for everything else. This example substitutes one cluster per
   distinct protein pair (see `train_on_subset.py`) instead, since all 40
   rows here are already distinct complexes.

Given all this, **this example does not call `train_final_model.py` or run
its CLI**. Per the task's own fallback guidance, it instead reuses everything
that *is* generic and reusable — the real `GAT_mut_processor` model class and
the real, non-hardcoded-path data-loading helpers
(`_build_emb_dict` / `_load_graphs` / `_gather_labels_pos_neg`), all imported
unmodified from `mutpred_ppi_cv.py` — and drives them with a short,
self-written training loop (`train_on_subset.py`) that mirrors
`train_final_model.py`'s `_train_loop` / `mutpred_ppi_cv.py`'s `train_fold`
almost line-for-line (same model, same optimizer/loss/early-stopping recipe;
`--ablation scratch`: random init, all params trainable, scaler fit on this
run's own data — exactly the fallback the task description itself suggested).
**No file outside `examples/` was modified.**

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
     label — see "Known issues" above).
   - **Step 3**: `generate_embeddings.py` — real ProtT5 loader utilities
     (`src/inference/utils/prott5_loader.py`), batched once over the whole
     small FASTA.
   - **Step 4**: `train_on_subset.py` — builds the training data via
     `mutpred_ppi_cv.py`'s real loader helpers, trains
     `GAT_mut_processor` from scratch for 1 epoch, saves the best-val-loss
     checkpoint.

## How to run it

```bash
conda activate ppi   # or: conda run -n ppi ...
bash examples/training_quickstart/run_example.sh --device cuda:0 --epochs 1
# or, with no GPU:
bash examples/training_quickstart/run_example.sh --device cpu --epochs 1
```

This regenerates (into this directory, untracked so the repo stays clean):
`af3_graphs/`, `wt_and_vt.fasta`, `wt_and_vt_t5_embs.h5`, and
`output/MutPred-PPI_quickstart_scratch.pt`.

## Expected output

A `.pt` checkpoint at `output/MutPred-PPI_quickstart_scratch.pt` — a
`state_dict` with 16 tensors (~407,585 parameters total, matching
`GAT_mut_processor`'s architecture), loadable with:

```python
import torch
sd = torch.load("output/MutPred-PPI_quickstart_scratch.pt", map_location="cpu", weights_only=True)
```

Console output ends with something like:

```
40 labeled rows loaded (15 disrupted / 25 maintained)
train/val split: 35 train, 5 val (9-group split)
Epoch 1: train loss=7.8042
  new best (val loss=1.4621)

Saved checkpoint: .../examples/training_quickstart/output/MutPred-PPI_quickstart_scratch.pt
Checkpoint has 16 state_dict tensors, 407585 total parameters
```

Exact loss values will vary run to run (random init, no fixed data order
guarantee beyond the seed) — what matters is that a well-formed checkpoint
file is produced with no errors. This is a smoke test of the training
mechanics, not a meaningful model (1 epoch on 40 rows will not produce a
predictive model — see the real training data volumes and results in the
paper / `docs/TRAINING.md`).
