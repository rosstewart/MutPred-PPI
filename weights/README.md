# MutPred-PPI model weights

## What is here now

| File | Trained on | Use |
|---|---|---|
| `MutPred-PPI.pt` | sahni_fragoza_varchamp_all (full training set, all data) | The published prediction model — use this to score your own variants |
| `MutPred-PPI_stability_pretrain.pt` | MegaScale (Tsuboyama et al. 2023) | The stability-pretrained checkpoint every final model fine-tunes from |
| `mutation_diff_scaler.pkl` | MegaScale | Required alongside every model; scales the mutation-difference features |

`v1_0/` holds the earlier pre-MegaScale stability pretrain
(`MutPred-PPI_v1_0_stability_pretrain.pt` + `mutation_diff_scaler_v1_0.pkl`), used
only by the `full` / `full_all` ablations. It is local-only and not distributed.

`MutPred-PPI.pt` is also produced by training the final model from scratch (see
[`docs/TRAINING.md`](../docs/TRAINING.md)); the figure-reproduction notebook does this.

Per-fold cross-validation checkpoints (`folds/`) are written by the GCV scripts
and are local-only.

## Zenodo

Training data and AlphaFold 3 structures are deposited separately — see
[Data Availability](../README.md#data-availability) in the main README for the
two DOIs. (This file previously cited the *structures* DOI as the source of
model weights; it is not.)

The scaler must sit in the same directory as the model, or be passed explicitly
with `--scaler`.
