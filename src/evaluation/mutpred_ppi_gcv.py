#!/usr/bin/env python
"""MutPred-PPI group cross-validation on the canonical tables.

The last method to move off the legacy data layer. It reads the same rows and
splits as every other method (gcv_common) and runs through the same shared
runner; its graph tensors come from `mutpred_ppi_data`, which resolves structures by
sequence rather than by filename.

Only the training loop is MutPred-PPI-specific -- it is `train_fold` from
mutpred_ppi_cv, unchanged, including the freeze strategy. `--ablation` selects
that strategy; the default `megascale_all` freezes nothing (megascale
initialisation, full fine-tune) and is the headline configuration.

Rows whose structure or embeddings are unavailable get a NaN prediction and are
excluded by the shared per-class AUC, with a counted reason printed at load.

Usage:
    conda run -n ppi python src/evaluation/mutpred_ppi_gcv.py \\
        --dataset sahni_fragoza_varchamp_all_mapped090826 --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent

import joblib  # noqa: E402

from utils.gcv_common import dataset_arg, dataset_config, DATASET_CHOICES, DATASET_CONFIGS, load_data, run_gcv  # noqa: E402
from utils.runtime import resolve_device  # noqa: E402
from utils.mutpred_ppi_data import build_tensors  # noqa: E402
from training.train_fold import (  # noqa: E402
    _MEGASCALE_SCALER_PATH, _PRIOR_BEST_SCALER_PATH, _V1_0_SCALER_PATH,
    train_fold)

# Which pretrained scaler an ablation uses. Mirrors mutpred_ppi_cv.run(): the
# mutation-diff scaler is fitted once during pretraining and reused, so the
# scaled diffs are identical every fold and every seed. Only `scratch` refits
# per fold, which train_fold does itself.
_MEGASCALE_ABLATIONS = {
    "megascale", "freeze_mut_processor", "freeze_gat", "megascale_all",
    "megascale_head",
    "megascale_all_no-gat", "megascale_all_no-mut", "megascale_all_wt-emb",
    "pretrain_zero_shot",
}
from paths import GCV_RESULTS_DIR  # noqa: E402


def run(args: argparse.Namespace) -> None:
    # cudnn is left nondeterministic on purpose: the reproducibility that matters
    # is the split, which comes from the seed-keyed splits table and is identical
    # across every method. Kernel-level nondeterminism buys speed and only
    # perturbs the last digits of a fold's predictions.
    cfg = dataset_config(args.dataset)
    device = resolve_device(args.device)

    rows = load_data(cfg)
    # require_complete: build_tensors raises unless every row has a structure and
    # all three embeddings, so this method is scored on exactly the rows the
    # sequence-only methods are. No NaN, no shrunken denominator.
    t = build_tensors(rows, args.dataset,
                      use_wt_emb=args.ablation in ("wt-emb", "megascale_all_wt-emb"),
                      two_hop=args.two_hop)

    # train_fold walks the whole tensor list, not just the fold's indices, so it
    # must be handed dense lists. Every row is usable, so position is identity.
    dense = {k: list(t[k])
             for k in ("node_emb", "edge_index", "pos_labels", "neg_labels",
                       "clusters", "mut_diff", "seq_lengths")}

    # Pre-scale the mutation diffs once, exactly as mutpred_ppi_cv.run() does.
    # For every ablation but `scratch`, train_fold expects them already scaled.
    if args.ablation == "prior_best":
        # The prior model's own scaler, not the MegaScale one -- its
        # mutation_diff_processor was fitted against these statistics, so pairing
        # it with a different scaler would misrepresent the arm it stands for.
        prefit_scaler = joblib.load(_PRIOR_BEST_SCALER_PATH)
    elif args.ablation in _MEGASCALE_ABLATIONS:
        prefit_scaler = joblib.load(_MEGASCALE_SCALER_PATH)
    elif args.ablation != "scratch":
        prefit_scaler = joblib.load(_V1_0_SCALER_PATH)
    else:
        prefit_scaler = None
    precomputed_diffs = (None if prefit_scaler is None
                         else prefit_scaler.transform(np.array(dense["mut_diff"])))

    def _fit_predict_fold(train_df, test_df, *, fold, train_idx, test_idx,
                          gcv_seed, **_):
        # numpy arrays: train_fold fancy-indexes these with a GroupKFold result
        tr = np.asarray(train_idx, dtype=int)
        te = np.asarray(test_idx, dtype=int)
        fold_seed = args.seed * 10000 + gcv_seed * 100 + fold
        preds, _labels = train_fold(
            tr, te, fold,
            dense["node_emb"], dense["edge_index"], dense["pos_labels"],
            dense["neg_labels"], dense["clusters"], dense["mut_diff"],
            dense["seq_lengths"], device,
            ablation=args.ablation, seed=fold_seed,
            precomputed_diffs=precomputed_diffs,
            fuse_batch=args.fuse_batch,
        )
        # train_fold returns predictions in the order of `te`, which is test_idx.
        return np.asarray(preds, dtype=float)

    run_gcv(cfg, args,
            result_stem=f"MutPredPPI_{cfg.name}_{args.ablation}",
            fit_predict_fold=_fit_predict_fold)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=dataset_arg, choices=list(DATASET_CONFIGS))
    # Restricted to what the data layer supports, so an unimplemented ablation
    # fails at parse time rather than silently reproducing `megascale_all`.
    p.add_argument("--ablation", default="megascale_all",
                   choices=["full", "full_all",
                            "megascale", "freeze_mut_processor", "freeze_gat", "megascale_all",
                            "megascale_head", "megascale_all_no-gat",
                            "megascale_all_no-mut", "megascale_all_wt-emb",
                            "scratch", "no-gat", "no-mut", "wt-emb",
                            "prior_best", "pretrain_zero_shot"])
    p.add_argument("--device", default="cuda:0")
    # 0, not 42: the published commands pass no --seed, and fold_seed derives
    # from it, so a different default silently reseeds every fold.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-gcv", type=int, default=30)
    p.add_argument("--outdir", default=str(GCV_RESULTS_DIR))
    p.add_argument(
        "--two-hop", action=argparse.BooleanOptionalAction, default=True,
        help="Restrict each sample to the two-hop neighbourhood of its mutation "
             "site (default: True). Exact for this model -- it reads one node "
             "after two GAT layers and pools nothing -- and ~30x smaller than "
             "the full complex. --no-two-hop restores whole-complex behaviour.")
    p.add_argument(
        "--fuse-batch", action=argparse.BooleanOptionalAction, default=True,
        help="Run one forward per optimiser step instead of one per sample "
             "(default: True). The loop already steps once per batch; this only "
             "fuses the forwards, by concatenating the batch's graphs into one "
             "disconnected graph. Same loss and gradient, ~5x faster. Dropout "
             "draws one mask per batch rather than per sample, so a fused run "
             "matches a serial one in distribution, not bitwise -- which is "
             "already true of any two GPU runs. --no-fuse-batch reverts.")
    p.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume from an existing checkpoint, continuing after the last "
             "completed GCV seed (default: True).",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
