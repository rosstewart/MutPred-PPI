# Training MutPred-PPI From Scratch

Covers stability pretraining and PPI fine-tuning. For inference with the pre-trained model, see
[`docs/INFERENCE.md`](INFERENCE.md). For reproducing paper figures/tables, see
[`docs/REPRODUCING_ANALYSES.md`](REPRODUCING_ANALYSES.md). For a fast working example, see
[`src/inference/example/`](../src/inference/example/).

## Downloads

From Zenodo (see [`docs/DATA_SOURCES.md`](DATA_SOURCES.md) for links):
- Model weights: `weights/`
- Training data: `datasets/training_eval/sahni_fragoza_mapped090826_rows.csv.gz`,
  `datasets/training_eval/sahni_only_mapped090826_rows.csv.gz` (and the other three
  canonical datasets; see `utils.gcv_common.DATASET_CONFIGS`). These are produced by
  `src/data_processing/training_sets/prepare_gcv_tables.py` -- see
  [`docs/DATA_PREPARATION.md`](DATA_PREPARATION.md).
- AF3 structures: `datasets/af3_structures.tar`, which extracts to `datasets/af3_structures/`.
  The canonicalized, one-structure-per-pair tree the graph builder reads is
  `datasets/af3_structures_canonical/` (3,854 gzipped mmCIFs + `manifest.csv`).

## Model weights

All models live in `weights/`. Only four files are git-tracked:

| File | Training data | Used for |
|---|---|---|
| `MutPred-PPI.pt` | Sahni + Fragoza + VarChAMP | Public inference, variant-repository inference (Fig 5, S4, S-stability) |
| `MutPred-PPI_sahni_fragoza.pt` | Sahni + Fragoza only | Grouped cross-validation (Fig 3), VarChAMP blind test (Fig 4) |
| `MutPred-PPI_stability_pretrain.pt` | MegaScale (Tsuboyama et al. 2023) | Pretraining checkpoint all final models fine-tune from |
| `mutation_diff_scaler.pkl` | MegaScale | Required alongside every model above |

`MutPred-PPI_sahni_fragoza.pt` (not `MutPred-PPI.pt`) is required for Fig 3/4 because both are
blind tests of generalization to unseen data — using the model trained on all data (including
VarChAMP) would defeat that purpose.

`weights/folds/` (per-fold checkpoints) and `weights/MutPred-PPI_sahni.pt` (Sahni-only, used for
the Fig S2 comparison) are present but not git-tracked.

## Stability Pretraining

Required once before any fine-tuning. Source data: [Tsuboyama et al. 2023](https://doi.org/10.1038/s41586-023-06328-6).
Train/val/test splits (`datasets/mega_splits.pkl`) are from
[SPURS](https://doi.org/10.1038/s41467-025-67609-4). `datasets/` is gitignored, so this
arrives with the Zenodo bundle rather than the git clone -- see [`docs/SETUP.md`](SETUP.md).

```bash
conda run -n ppi python src/training/preprocess_stability_data.py \
    --csv     /path/to/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
    --pdb-dir /path/to/AlphaFold_model_PDBs/ \
    --splits  datasets/mega_splits.pkl \
    --outdir  megascale_preprocessed/ \
    --subgraph-hops 2 \
    --device  cuda:0 --n-jobs 16
# Outputs: preprocessed.pkl, mutation_diff_scaler.pkl

conda run -n ppi python src/training/pretrain_stability.py \
    --data     megascale_preprocessed/preprocessed.pkl \
    --scaler   megascale_preprocessed/mutation_diff_scaler.pkl \
    --outmodel weights/MutPred-PPI_stability_pretrain.pt \
    --device   cuda:0
```

## Training Data Preparation

Contact graphs live in one HDF5 `ContactGraphStore`
([`src/contact_graphs.py`](../src/contact_graphs.py)) per data tier, not in per-complex `.mat`
files:

| Store | Used by |
|---|---|
| `datasets/training_eval/contact_graphs.h5` | training and evaluation |
| `datasets/variant_dbs/contact_graphs.h5` | variant-database inference |

Entries are keyed on the sorted pair of `sha256(chain_sequence)[:16]`, so identity comes from
sequence content rather than from a filename, and `load_dense(interactor=..., partner=...)` /
`load_edge_index(interactor=..., partner=...)` (keyword-only, sequences not accessions) return
the graph already oriented to the requested interactor, with self-loops added. See
[`docs/INFERENCE.md`](INFERENCE.md#the-contact-graph-store).

Rebuild the training/eval store from the canonical structures:

```bash
conda run -n ppi python src/data_processing/rebuild_graphs_from_structures.py \
    --structures datasets/af3_structures_canonical \
    --out datasets/training_eval/contact_graphs.h5
```

`--compare-to <existing.h5>` reports per-key differences instead of silently replacing them.
Deriving graphs from the canonicalized one-structure-per-pair tree is what removed the arbitrary
choice the old `.mat` migration had to make.

`src/inference/01_make_contact_graphs_and_fasta.py` builds a store for *your own* structures in
the standalone inference pipeline; it is not the path used for the training data. It also has a
known defect that empties `wt_and_vt.fasta` —
see [`docs/INFERENCE.md`](INFERENCE.md#known-issue-empty-wt_and_vtfasta).

`src/training/train_fold.py` holds the single training loop (`train_fold`) shared by
`src/evaluation/mutpred_ppi_gcv.py` and `src/training/train_final_model.py`, so the CV numbers
and the released weights come from one implementation. **It has no CLI** — it is imported, never
invoked directly. Tensor construction lives in `src/utils/mutpred_ppi_data.py::build_tensors`,
which reads rows from the canonical tables and graphs from the store.

CV fold assignments (30-seed grouped cross-validation, used for Fig 3) are generated inline
during the GCV run — no separate step required.

## Model Training

```bash
# Sahni+Fragoza (Fig 3, Fig 4 blind test)
conda run -n ppi python src/training/train_final_model.py \
    --dataset sahni_fragoza --ablation megascale_all --device cuda:0

# Sahni+Fragoza+VarChAMP (public/variant-DB inference)
conda run -n ppi python src/training/train_final_model.py \
    --dataset sahni_fragoza_varchamp_full_pooled --ablation megascale_all --device cuda:0
```

VarChAMP training data is unpublished IGVF consortium data — cross-reference
[data.igvf.org](https://data.igvf.org). It is required only for the second command above; the
public Sahni+Fragoza model needs no VarChAMP data.
