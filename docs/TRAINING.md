# Training MutPred-PPI From Scratch

Covers stability pretraining and PPI fine-tuning. For inference with the pre-trained model, see
[`docs/INFERENCE.md`](INFERENCE.md). For reproducing paper figures/tables, see
[`docs/REPRODUCING_ANALYSES.md`](REPRODUCING_ANALYSES.md). For a fast working example, see
[`examples/`](../examples/).

## Downloads

From Zenodo (see [`docs/DATA_SOURCES.md`](DATA_SOURCES.md) for links):
- Model weights: `weights/`
- Training data: `datasets/train_eval/sahni_fragoza_train.csv`, `datasets/train_eval/sahni_train.csv`
- AF3 structures: `datasets/af3_structures.tar`

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
[SPURS](https://doi.org/10.1038/s41467-025-67609-4) and included in this repo.

```bash
conda run -n ppi python src/training/preprocess_stability_data.py \
    --csv     /path/to/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
    --pdb-dir /path/to/AlphaFold_model_PDBs/ \
    --splits  datasets/mega_splits.pkl \
    --outdir  megascale_preprocessed/ \
    --device  cuda:0 --n-jobs 16
# Outputs: preprocessed.pkl, mutation_diff_scaler.pkl

conda run -n ppi python src/training/pretrain_stability.py \
    --data     megascale_preprocessed/preprocessed.pkl \
    --scaler   megascale_preprocessed/mutation_diff_scaler.pkl \
    --outmodel weights/MutPred-PPI_stability_pretrain.pt \
    --device   cuda:0
```

## Training Data Preparation

Contact graphs (`.mat` files) are derived from AF3 structures:

```bash
conda run -n ppi python src/inference/01_make_contact_graphs_and_fasta.py \
    working_dir/ datasets/af3_structures/ variants.tsv
```

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
