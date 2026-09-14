# Training MutPred-PPI From Scratch

Covers stability pretraining and PPI fine-tuning. For inference with the pre-trained model, see
[`docs/INFERENCE.md`](INFERENCE.md). For reproducing paper figures/tables, see
[`docs/REPRODUCING_ANALYSES.md`](REPRODUCING_ANALYSES.md). For a fast working example, see
[`src/inference/example/`](../src/inference/example/).

## Downloads

From Zenodo (see [`docs/DATA_PREPARATION.md`](DATA_PREPARATION.md) for links):
- Model weights: `weights/`
- Training data: `datasets/training_eval/<dataset>_rows.csv.gz` for each of the five
  canonical datasets (`sahni_fragoza`, `sahni_only`, and the other three
  canonical datasets; see `utils.gcv_common.DATASET_CONFIGS`). These are produced by
  `src/data_processing/training_sets/prepare_gcv_tables.py` -- see
  [`docs/DATA_PREPARATION.md`](DATA_PREPARATION.md).
- AF3 structures: `datasets/af3_structures.tar`, which extracts to `datasets/af3_structures/`.
  The canonicalized, one-structure-per-pair tree the graph builder reads is
  `datasets/af3_structures_canonical/` (100,739 gzipped mmCIFs + `manifest.csv`).

## Model weights

All models live in `weights/`. Two files ship with the repository:

| File | Training data | Used for |
|---|---|---|
| `MutPred-PPI_stability_pretrain.pt` | MegaScale (Tsuboyama et al. 2023) | Pretraining checkpoint all final models fine-tune from |
| `mutation_diff_scaler.pkl` | MegaScale | Required alongside every model above |

**The prediction models are produced by the commands below, not shipped.**
`MutPred-PPI.pt` (all data) and the Sahni+Fragoza model are both trained from the
stability-pretrained checkpoint; see [`weights/README.md`](../weights/README.md)
for how to obtain predictions without training anything.

A Sahni+Fragoza-only model, not the all-data one, is what Fig 3 and Fig 4 require:
both are blind tests of generalisation, and scoring them with a model that had
seen VarChAMP would defeat the purpose. The VarChAMP blind test trains that model
on demand rather than reading a checkpoint, so it is always in step with the
current tables.

`weights/folds/` (per-fold checkpoints) is written by the GCV scripts and is not
tracked.

## Stability Pretraining

Required once before any fine-tuning. Source data: [Tsuboyama et al. 2023](https://doi.org/10.1038/s41586-023-06328-6).
Train/val/test splits (`datasets/mega_splits.pkl`) are from
[SPURS](https://doi.org/10.1038/s41467-025-67609-4). `datasets/` is gitignored, so this
arrives with the Zenodo bundle rather than the git clone -- see [`DATA.md`](DATA.md).

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

Contact graphs live in a single HDF5 `ContactGraphStore`
([`src/contact_graphs.py`](../src/contact_graphs.py)), keyed by sequence rather than by
filename, and reached under two names:

| Store | Used by |
|---|---|
| `datasets/training_eval/contact_graphs.h5` | training and evaluation |
| `datasets/variant_dbs/contact_graphs.h5` | variant-database inference |

Entries are keyed by sequence content rather than by filename, and the accessors take
sequences and return the graph already oriented to the requested interactor. The store's
keying, accessors and orientation rules are documented once, in [`docs/INFERENCE.md`](INFERENCE.md#the-contact-graph-store).

Rebuild the training/eval store from the canonical structures:

```bash
conda run -n ppi python src/data_processing/rebuild_graphs_from_structures.py \
    --structures datasets/af3_structures_canonical \
    --out datasets/training_eval/contact_graphs.h5
```

`--compare-to <existing.h5>` reports per-key differences instead of silently replacing them.

`src/inference/01_make_contact_graphs_and_fasta.py` builds a store for *your own* structures in
the standalone inference pipeline; it is not the path used for the training data.

`src/training/train_fold.py` holds the single training loop (`train_fold`) shared by
`src/evaluation/mutpred_ppi_gcv.py` and `src/training/train_final_model.py`, so the CV numbers
and the released weights come from one implementation. **It has no CLI**, it is imported, never
invoked directly. Tensor construction lives in `src/utils/mutpred_ppi_data.py::build_tensors`
which reads rows from the canonical tables and graphs from the store.

CV fold assignments (30-seed grouped cross-validation, used for Fig 3) are generated inline
during the GCV run, no separate step required.

## Model Training

```bash
# Sahni+Fragoza (Fig 3, Fig 4 blind test) -- per-fold checkpoints
conda run -n ppi python src/training/train_final_model.py \
    --dataset sahni_fragoza --ablation megascale_all \
    --save-models-dir weights/folds/ --device cuda:0

# Sahni+Fragoza+VarChAMP, all data, no CV -- this is MutPred-PPI.pt
conda run -n ppi python src/training/train_final_model.py \
    --dataset sahni_fragoza_varchamp_all --ablation megascale_all \
    --save-models-dir weights/ --device cuda:0 --no-cv
```

`--save-models-dir` is required. `--dataset` accepts the short names above or the
full stamped filenames on disk.

VarChAMP training data is not redistributable; see [DATA.md](DATA.md#unpublished). Cross-reference
[data.igvf.org](https://data.igvf.org). It is required only for the second command above; the
public Sahni+Fragoza model needs no VarChAMP data.
