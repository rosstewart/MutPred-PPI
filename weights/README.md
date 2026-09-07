# MutPred-PPI Model Weights

Pre-trained weights for MutPred-PPI (Zenodo: https://doi.org/10.5281/zenodo.18701748).

| File | Training data | Use |
|------|--------------|-----|
| `MutPred-PPI.pt` | Sahni + Fragoza + VarChAMP | Primary model — recommended for all users |
| `MutPred-PPI_sahni_fragoza.pt` | Sahni + Fragoza only | Grouped cross-validation (Fig 3), VarChAMP blind test (Fig 4) |
| `MutPred-PPI_stability_pretrain.pt` | MegaScale (Tsuboyama et al.) | Stability-pretrained checkpoint all final models fine-tune from |
| `mutation_diff_scaler.pkl` | MegaScale (Tsuboyama et al.) | Required alongside every model above |

Not distributed via Zenodo, present locally only: `folds/` (per-fold checkpoints),
`MutPred-PPI_sahni.pt` (Sahni-only, Fig S2 comparison), `v1_0/MutPred-PPI_v1_0_stability_pretrain.pt`
+ `v1_0/mutation_diff_scaler_v1_0.pkl` (pre-MegaScale stability pretrain, used only for the
`full`/`full_all` ablation, see `docs/REPRODUCING_ANALYSES.md`).

The scaler must be in the same directory as the model, or passed explicitly via `--scaler`. See
the main [README](../README.md) and [`docs/TRAINING.md`](../docs/TRAINING.md) for usage.
