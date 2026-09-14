# Model checkpoints

Two files are tracked in git, because the inference quickstart cannot run without them:

| File | Size | What it is |
|---|---|---|
| `MutPred-PPI.pt` | 1.6 MB | **the model.** Trained on all labelled interaction data, fine-tuned from the MegaScale stability pretrain. This is what `src/inference/` and `src/variant_db_inference/` load. |
| `mutation_diff_scaler.pkl` | 25 KB | the `StandardScaler` for the mutation-difference features. Must sit beside the model — scores are meaningless without it. |

Everything else is distributed through Zenodo (see [`../docs/ZENODO.md`](../docs/ZENODO.md)),
because checkpoints are generated artifacts and git is the wrong place for them:

| File | What it is |
|---|---|
| `MutPred-PPI_stability_pretrain.pt` | MegaScale ΔΔG pretrain. Every final model fine-tunes from it, and `run_stability_inference.py` uses it directly to predict ΔΔG. |
| `sahni_fragoza/MutPred-PPI.pt` + `mutation_diff_scaler.pkl` | the **demonstration tier**. Selected by `run_variant_db_inference.py --model-tier sahni_fragoza` when the unpublished VarChAMP data needed to train the all-data model is absent. Its scores are not the published ones; see the reproducibility table in the [README](../README.md). |
| `v1_0/` | the previous published model and its scaler. Used only by the "Prior Best" ablation in `run_roc_ablation.py`. |

## `blind_test/` is an output directory

`src/evaluation/run_varchamp_blind_test.py` trains a model on demand and writes it to
`weights/blind_test/`. Those checkpoints are produced by a run, not inputs to one; they are
not tracked and do not need to be kept.

## Which model produced which number

- Published variant-repository predictions (Fig 5, S8, the stability figures) — `MutPred-PPI.pt`.
- Published ΔΔG predictions — `MutPred-PPI_stability_pretrain.pt`.
- Cross-validation figures (Fig 3, S1, S7) — models trained per fold during the run and not retained.
- Blind test (Fig 4, S2) — trained on demand into `blind_test/`.
