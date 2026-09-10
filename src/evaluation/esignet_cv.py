#!/usr/bin/env python
"""eSIG-Net group cross-validation training script.

Mirrors the four SWING GCV scripts (SWING_MutInt_Notebook_*_gcv_iter.py) but
replaces Doc2Vec + XGBoost with eSIG-Net (dual-channel SDNN + ESM-2 discriminator).

Key difference from SWING: no STRINGENT_PRETRAIN mode — eSIG-Net always trains
from scratch on each fold's training data only.

Usage:
    conda run -n ppi python esignet_gcv_iter.py --dataset sahni_fragoza --device cuda:0
    conda run -n ppi python esignet_gcv_iter.py --dataset sahni_fragoza_mapped090826 \\
        --device cuda:0 --n-gcv 30 --outdir /path/to/results/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── eSIG-Net import (vendored in-repo, see src/evaluation/predictors/) ─────────

import evaluation.predictors.esignet as _esignet_mod                                    # noqa: E402
from evaluation.predictors.esignet import ESigNetPredictor, _compute_573, _FEAT_CACHE  # noqa: E402
from evaluation.predictors.nn_base import load_cache                                    # noqa: E402

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import DATASETS_DIR, GCV_RESULTS_DIR  # noqa: E402


# Surface load_cache's INFO logs ("Loading cache: …", "Loaded N entries").
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── shared GCV infrastructure ─────────────────────────────────────────────────
# Lives in src/utils/gcv_common.py: mint_cv, pplm_cv, swing_gcv and the two
# precompute_* scripts each import these names from there, not from here.
from utils.gcv_common import DATASET_CONFIGS, DatasetConfig, run_gcv




# ── feature pre-warming ───────────────────────────────────────────────────────



def audit_esm_cache(
    ordered_df: pd.DataFrame,
    cfg: DatasetConfig,
    min_hit_rate: float,
    require: bool,
) -> None:
    """Load the active ESM cache and report WT/MT key hit rates.

    Aborts when require=True and the mutant-key hit rate falls below
    min_hit_rate. Without a usable cache eSIG-Net's discriminator branch
    silently degenerates to zero-vectors — fail fast instead.
    """
    path = _esignet_mod._ESM_CACHE_PATH
    print(f"\n── ESM-2 cache audit ──", flush=True)
    print(f"path: {path}", flush=True)

    cache = load_cache(path)
    if cache is None:
        msg = (
            f"ESM cache not found at {path}. eSIG-Net's discriminator branch "
            f"will receive all-zero embeddings — model is effectively neutered."
        )
        if require:
            raise SystemExit("ABORT: " + msg + "\nPass --require-esm=false to override.")
        print("WARNING: " + msg, flush=True)
        return

    print(f"entries: {len(cache)}", flush=True)

    # Filter out NaN rows (unmatched vt_ids produce NaN mutations)
    valid_df    = ordered_df
    interactors = valid_df["interactor"].astype(str)
    mutations   = valid_df["mutation"].astype(str)
    unique_wt   = set(interactors)
    unique_mt   = set(zip(interactors, mutations))

    # Keys are 1-based, matching the canonical tables. This used to count hits
    # under both conventions and take whichever won, which is guessing.
    wt_hits = sum(1 for k in unique_wt if k in cache)
    mt_hits = sum(1 for ia, mu in unique_mt if f"{ia}_{mu}" in cache)
    n_wt, n_mt = len(unique_wt), len(unique_mt)

    print(f"interactor (WT) keys: {wt_hits}/{n_wt} hit "
          f"({wt_hits / max(n_wt, 1):.1%})", flush=True)
    print(f"mutant keys: {mt_hits}/{n_mt} hit "
          f"({mt_hits / max(n_mt, 1):.1%})", flush=True)

    mt_rate = mt_hits / max(n_mt, 1)
    if wt_hits < n_wt or mt_rate < min_hit_rate:
        missing_wt = sorted(k for k in unique_wt if k not in cache)
        missing_mt = sorted(f"{ia}_{mu}" for ia, mu in unique_mt
                            if f"{ia}_{mu}" not in cache)
        msg = (
            f"ESM-2 cache is incomplete: {n_wt - wt_hits} WT and "
            f"{n_mt - mt_hits} mutant keys missing "
            f"(mutant hit rate {mt_rate:.1%}, required {min_hit_rate:.1%}). "
            f"Every method is scored on the same rows, so a partial cache is an "
            f"error. Run "
            f"src/data_processing/precompute_esm2_datasets.py --dataset {cfg.name} "
            f"to fill it.\n  missing WT : {missing_wt[:10]}"
            f"\n  missing MUT: {missing_mt[:10]}"
        )
        if require:
            raise SystemExit("ABORT: " + msg)
        print("WARNING: " + msg, flush=True)


def prewarm_features(ordered_df: pd.DataFrame) -> None:
    """Pre-populate the module-level _FEAT_CACHE for all unique sequences.

    eSIG-Net computes 573-dim AAC+CTD features on-the-fly and caches them by
    sequence string. Pre-warming here avoids redundant computation across folds.
    """
    unique_seqs = set()
    for col in ("interactor_sequence", "partner_sequence"):
        unique_seqs.update(ordered_df[col].dropna().unique())
    print(f"Pre-computing 573-dim features for {len(unique_seqs)} unique sequences...",
          flush=True)
    for seq in unique_seqs:
        _compute_573(seq)
    print(f"  done ({len(_FEAT_CACHE)} sequences cached)", flush=True)




# ── main GCV loop ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    if args.esm_cache:
        _esignet_mod._ESM_CACHE_PATH = args.esm_cache
        print(f"ESM cache path overridden: {args.esm_cache}", flush=True)
    else:
        canonical = DATASETS_DIR / "mapped090826" / f"{args.dataset}_esm2.pkl"
        if canonical.exists():
            _esignet_mod._ESM_CACHE_PATH = str(canonical)
            print(f"ESM cache: {canonical}", flush=True)

    cfg = DATASET_CONFIGS[args.dataset]

    def _preflight(ordered_df, cfg_, a):
        audit_esm_cache(ordered_df, cfg_, a.min_esm_hit_rate, a.require_esm)
        prewarm_features(ordered_df)

    run_gcv(
        cfg, args,
        result_stem=f"ESigNet_{cfg.name}",
        make_predictor=lambda a: ESigNetPredictor(
            seed=a.seed, n_epochs=a.n_epochs, device=a.device),
        preflight=_preflight,
    )

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="eSIG-Net GCV training (mirrors SWING pipeline)"
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIGS),
        help="Dataset configuration to use",
    )
    p.add_argument(
        "--device",
        default="",
        help="PyTorch device string (e.g. 'cuda:0'). Defaults to auto-detect.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for eSIG-Net training (default: 42)",
    )
    p.add_argument(
        "--n-gcv",
        type=int,
        default=30,
        help="Number of GCV iterations (default: 30)",
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=8,
        help="eSIG-Net training epochs per fold (default: 8)",
    )
    p.add_argument(
        "--outdir",
        default=str(GCV_RESULTS_DIR),
        help="Output directory for results (default: CV splits dir)",
    )
    p.add_argument(
        "--esm-cache",
        default="",
        help="Override ESM-2 residue embeddings cache path (default: built-in path in esignet.py)",
    )
    p.add_argument(
        "--min-esm-hit-rate",
        type=float,
        default=1.0,
        help="Minimum mutant-key hit rate in the ESM cache before "
             "--require-esm aborts the run (default: 1.0, i.e. every key must "
             "be present -- all methods are scored on the same rows).",
    )
    p.add_argument(
        "--require-esm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort if the ESM cache is missing or has hit rate below "
             "--min-esm-hit-rate (default: True). Pass --no-require-esm to "
             "run with the discriminator branch zeroed out.",
    )
    p.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume from an existing checkpoint, continuing after the last "
             "completed GCV seed (default: True).",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
