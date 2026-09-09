#!/usr/bin/env python
"""SWING grouped cross-validation, for any dataset in DATASET_CONFIGS.

Migrated in from an external, unversioned script
(`SWING_scripts/SWING_MutInt_Notebook_sahni_fragoza_varchamp_full_pooled_gcv_iter.py`)
which was the only way to run SWING on SFVCFP.  Two things are different here:

* **No duplicated SWING code.**  The external script carried its own copies of
  the amino-acid score table, the Doc2Vec/XGBoost hyperparameters, and the
  window-encoding / k-mer / corpus / WT-frame builders.  Those were checked
  against `swing_common` on real benchmark rows and found identical -- the score
  dictionary, encodings, k-mers, tagged corpus and WT frames all match exactly --
  so this module imports them instead.
* **No hardcoded dataset.**  `--dataset` selects the config, so SFVCFP and
  Sahni+Fragoza run through the same path.

Data loading reuses the shared GCV layer in `gcv_common` (`DATASET_CONFIGS`,
`load_data`), which is what the external script did.

Two modes, both reported in the paper and *not* interchangeable:

    default          Doc2Vec is retrained per fold on training rows only, and
                     test vectors are inferred.  This is the blind-test variant.
                     Saved with the suffix `_no_test_pretrain`.

    --test-pretrain  Doc2Vec is trained once on the whole dataset, test folds
                     included.  That leaks test information into the
                     representation; it is reported only as the "Test Pretrain"
                     variant.  Saved with the suffix `_test_pretrain`.

Usage:
    conda run -n ppi python src/evaluation/swing_gcv.py \\
        --dataset sahni_fragoza_varchamp_full_pooled [--test-pretrain]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from xgboost import XGBClassifier

_HERE = Path(__file__).resolve().parent

from paths import GCV_RESULTS_DIR  # noqa: E402
from evaluation.gcv_common import DATASET_CONFIGS, load_data, run_gcv  # noqa: E402
from evaluation.swing_common import (  # noqa: E402  single source for all SWING internals
    _D2V_DIM, _XGB_N_EST, _XGB_DEPTH, _XGB_LR,
    _get_window_encodings, _get_kmers, _get_corpus,
    _build_swing_df, build_wt_df, build_d2v,
)

N_SEEDS = 30


def _global_features(ordered_df: pd.DataFrame, valid_mask: np.ndarray,
                     d2v_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Doc2Vec vectors trained once on the whole dataset (leaky 'test pretrain')."""
    n_total = len(ordered_df)
    valid_pos = np.where(valid_mask)[0]
    n_valid = int(valid_mask.sum())

    valid_df = ordered_df[valid_mask].copy().reset_index(drop=True)
    valid_df["_vidx"] = range(n_valid)
    wt_df = build_wt_df(valid_df)
    wt_df["_vidx"] = range(n_valid)

    combined = (pd.concat([valid_df, wt_df])
                .sample(frac=1, random_state=1).reset_index(drop=True))

    if d2v_path.exists():
        model = Doc2Vec.load(str(d2v_path))
        print(f"  loaded Doc2Vec {d2v_path}", flush=True)
    else:
        print("  training global Doc2Vec ...", flush=True)
        model = build_d2v(combined)
        d2v_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(d2v_path))
        print(f"  saved Doc2Vec -> {d2v_path}", flush=True)

    combined["Vectors"] = model.dv.vectors.tolist()

    mut = np.empty((n_valid, _D2V_DIM))
    wt = np.empty((n_valid, _D2V_DIM))
    for _, row in combined.iterrows():
        v = np.array(row["Vectors"])
        (mut if row["Type"] == "Mutant" else wt)[int(row["_vidx"])] = v

    feats = np.full((n_total, _D2V_DIM), np.nan)
    feats_wt = np.full((n_total, _D2V_DIM), np.nan)
    feats[valid_pos] = mut
    feats_wt[valid_pos] = wt
    return feats, feats_wt


def _fold_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Doc2Vec retrained on this fold's training rows; test vectors inferred."""
    wt_train = build_wt_df(train_df)
    wt_train["_oidx"] = train_df.index.tolist()
    train_df = train_df.copy()
    train_df["_oidx"] = train_df.index.tolist()

    combined = (pd.concat([train_df, wt_train])
                .sample(frac=1, random_state=1).reset_index(drop=True))
    model = build_d2v(combined)
    combined["Vectors"] = model.dv.vectors.tolist()

    mut_rows = combined[combined["Type"] == "Mutant"].sort_values("_oidx")
    wt_rows = combined[combined["Type"] == "WildType"].sort_values("_oidx")
    X_train_mut = np.array(mut_rows["Vectors"].tolist())
    X_train_wt = np.array(wt_rows["Vectors"].tolist())
    assert len(X_train_mut) == len(train_df)

    corpus = list(_get_corpus(_get_kmers(_get_window_encodings(test_df))))
    X_test = np.array([model.infer_vector(doc.words) for doc in corpus])
    return X_train_mut, X_train_wt, X_test


def run(dataset: str, test_pretrain: bool, outdir: Path,
        resume: bool = True) -> None:
    cfg = DATASET_CONFIGS[dataset]
    print(f"dataset={dataset}  mode="
          f"{'test-pretrain (leaky)' if test_pretrain else 'blind test'}", flush=True)

    ordered_df = load_data(cfg)
    n_total = len(ordered_df)
    valid_mask = np.ones(n_total, dtype=bool)   # canonical tables have no gaps
    print(f"  {n_total} rows", flush=True)
    labels = ordered_df["perturbed"].to_numpy().astype(float)

    # Convert canonical column names to SWING format once (Position, Before_AA,
    # After_AA, Mutated_Seq, Type) so swing_common functions work without per-fold
    # column lookups.  _build_swing_df preserves row order and never drops rows on
    # clean canonical data.
    swing_df = _build_swing_df(ordered_df)
    # _build_swing_df skips synonymous and unvalidatable rows. On canonical
    # tables it drops none, and the fold indices address swing_df positionally,
    # so a drop would silently misalign predictions against labels.
    if len(swing_df) != n_total:
        raise ValueError(
            f"{dataset}: _build_swing_df returned {len(swing_df)} rows for "
            f"{n_total} canonical rows; fold indices would misalign")

    feats = feats_wt = None
    if test_pretrain:
        d2v_path = outdir / f"SWING_{dataset}_doc2vec.model"
        feats, feats_wt = _global_features(swing_df, valid_mask, d2v_path)

    def _fit_predict_fold(train_slice, test_slice, *, fold, train_idx, test_idx, **_):
        """SWING's training loop: Doc2Vec features + WT-augmented XGBoost.

        This is the only part of the GCV that is SWING-specific. Everything
        around it -- loading, alignment, fold iteration, AUC, artifacts -- is
        gcv_common.run_gcv, shared with every other method.
        """
        y_tr_mut = labels[np.asarray(train_idx)]
        tr_df = swing_df.iloc[train_idx].reset_index(drop=True)
        te_df = swing_df.iloc[test_idx].reset_index(drop=True)

        if test_pretrain:
            tr_ord, te_ord = np.asarray(train_idx), np.asarray(test_idx)
            X_tr_mut, X_tr_wt, X_te = feats[tr_ord], feats_wt[tr_ord], feats[te_ord]
        else:
            X_tr_mut, X_tr_wt, X_te = _fold_features(tr_df, te_df)

        X_train = np.concatenate([X_tr_mut, X_tr_wt])
        # WT rows are negatives by construction: an unmutated sequence is,
        # definitionally, a non-perturbing "variant".
        y_train = np.concatenate([y_tr_mut, np.zeros(len(X_tr_wt))])

        clf = XGBClassifier(n_estimators=_XGB_N_EST, max_depth=_XGB_DEPTH,
                            learning_rate=_XGB_LR)
        clf.fit(X_train, y_train)

        print(f"  fold {fold}: {len(X_train)} train pts (Mut+WT)", flush=True)
        return clf.predict_proba(X_te)[:, 1]

    code = "_test_pretrain" if test_pretrain else "_no_test_pretrain"

    _args = argparse.Namespace(n_gcv=N_SEEDS, outdir=str(outdir), resume=resume)
    run_gcv(cfg, _args,
            result_stem=f"SWING_{dataset}{code}",
            fit_predict_fold=_fit_predict_fold)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS))
    ap.add_argument("--test-pretrain", action="store_true",
                    help="Train Doc2Vec once on the FULL dataset including test folds. "
                         "Leaks test information; reported only as the 'Test Pretrain' variant.")
    ap.add_argument("--outdir", default=str(GCV_RESULTS_DIR))
    ap.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume from an existing checkpoint, continuing after the last "
             "completed GCV seed (default: True).",
    )
    args = ap.parse_args()
    run(args.dataset, args.test_pretrain, Path(args.outdir), args.resume)


if __name__ == "__main__":
    main()
