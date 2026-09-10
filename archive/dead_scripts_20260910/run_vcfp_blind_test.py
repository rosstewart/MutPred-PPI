#!/usr/bin/env python
"""Run the SFVCFP blind test for a single comparator method (single-pass).

With the canonical sfvcfp_rows.csv.gz table all rows are in UniProt space and
carry their blind_test_class (C1/C2/C3) directly, so no supplement, merge, or
restratify step is needed.

Usage (Phase 6 — requires Phase 7 schema migration and re-run):
    conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method mutpredppi [--device cuda:1]
    conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method esignet [--device cuda:1] [--seed 42]
    conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method mint --predictor seq_diff [--seed 42]
    conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method pplm --predictor site_diff [--seed 42]
    conda run -n ppi python src/evaluation/run_vcfp_blind_test.py --method swing [--test-pretrain] [--seed 42]

For MutPred2 (external tool), see src/analysis/import_mutpred2_vcfp_scores.py instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_EVAL_DIR = Path(__file__).resolve().parent          # src/evaluation
_PUB = _EVAL_DIR.parent.parent                       # repo root

from paths import DATASETS_DIR, RESULTS_DIR  # noqa: E402

SFVCFP_CSV  = DATASETS_DIR / "sfvcfp_rows.csv.gz"
OUT_DIR     = RESULTS_DIR / "varchamp_seqcnf_newvar_eval"
METHODS     = ("mutpredppi", "esignet", "mint", "pplm", "swing")


def load_rows() -> pd.DataFrame:
    """Load the canonical SFVCFP row table."""
    df = pd.read_csv(SFVCFP_CSV, index_col="row_index")
    assert {"interactor", "partner", "variant", "label", "blind_test_class"}.issubset(df.columns)
    return df


def save_results(
    scores: np.ndarray,
    df: pd.DataFrame,
    description: str,
    out_dir: Path = OUT_DIR,
) -> None:
    """Save per-class (C1/C2/C3) result arrays, matching the expected filename convention."""
    labels  = df["label"].values.astype(int)
    classes = df["blind_test_class"].values
    # 3-field, space-separated -- matches every existing *_c{1,2,3}_vt_ids.npy array
    # and what export_reconstruction_tables._parse_blind_vt_id expects.  Emitting
    # "{i}_{p} {v}" here would make the arrays non-joinable across methods.
    vt_ids  = (df["interactor"] + " " + df["partner"] + " " + df["variant"]).values

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid = ~np.isnan(scores)
    n_nan = int((~valid).sum())
    if n_nan:
        print(f"  WARNING: {n_nan}/{len(scores)} NaN scores (cache misses / missing graphs)")

    for c in [1, 2, 3]:
        mask = (classes == c) & valid
        n_pos = int((labels[mask] == 1).sum())
        n_neg = int((labels[mask] == 0).sum())
        auc_str = (
            f"{roc_auc_score(labels[mask], scores[mask]):.4f}"
            if n_pos >= 2 and n_neg >= 2
            else "n/a"
        )
        print(f"  C{c}: n={int(mask.sum())} (pos={n_pos}, neg={n_neg}) AUC={auc_str}")
        np.save(out_dir / f"{description}_c{c}_preds.npy",  scores[mask].astype(np.float32))
        np.save(out_dir / f"{description}_c{c}_labels.npy", labels[mask])
        np.save(out_dir / f"{description}_c{c}_vt_ids.npy", vt_ids[mask])

    print(f"Saved → {out_dir}/{description}_c{{1,2,3}}_*.npy")


def _score_mutpredppi(df: pd.DataFrame, device: str) -> tuple[np.ndarray, str]:
    """Train/score MutPred-PPI on the full SFVCFP row set."""
    raise NotImplementedError(
        "MutPred-PPI single-pass blind test not yet implemented. "
        "See Phase 6 in the plan (requires graph + ProtT5 loading for all rows)."
    )


def _score_esignet(df: pd.DataFrame, device: str, seed: int) -> tuple[np.ndarray, str]:
    """Train/score eSIG-Net on the full SFVCFP row set."""
    raise NotImplementedError(
        "eSIG-Net single-pass blind test not yet implemented. "
        "See Phase 6 in the plan (requires combined ESM-2 cache covering all rows)."
    )


def _score_cachemlp(df: pd.DataFrame, method: str, predictor: str, seed: int) -> tuple[np.ndarray, str]:
    """Train/score MINT or PPLM on the full SFVCFP row set."""
    raise NotImplementedError(
        f"{method} single-pass blind test not yet implemented. "
        "See Phase 6 in the plan."
    )


def _score_swing(df: pd.DataFrame, test_pretrain: bool, seed: int) -> tuple[np.ndarray, str]:
    """Train/score SWING on the full SFVCFP row set."""
    raise NotImplementedError(
        "SWING single-pass blind test not yet implemented. "
        "See Phase 6 in the plan."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--predictor", default="seq_diff", choices=("seq_diff", "site_diff"),
                    help="MINT/PPLM only.")
    ap.add_argument("--test-pretrain", action="store_true", help="SWING only.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:1", help="MutPred-PPI/eSIG-Net only.")
    args = ap.parse_args()

    print(f"Loading canonical SFVCFP rows from {SFVCFP_CSV}")
    df = load_rows()
    print(f"  {len(df)} labeled rows, {df['blind_test_class'].value_counts().to_dict()}")

    if args.method == "mutpredppi":
        scores, description = _score_mutpredppi(df, args.device)
    elif args.method == "esignet":
        scores, description = _score_esignet(df, args.device, args.seed)
    elif args.method in ("mint", "pplm"):
        scores, description = _score_cachemlp(df, args.method, args.predictor, args.seed)
    elif args.method == "swing":
        scores, description = _score_swing(df, args.test_pretrain, args.seed)
    else:
        raise ValueError(f"Unknown method: {args.method!r}")

    save_results(scores, df, description)
    print("Done.")


if __name__ == "__main__":
    main()
