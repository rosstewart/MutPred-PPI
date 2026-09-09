#!/usr/bin/env python
"""Shared group-cross-validation infrastructure.

One representation, one loader, one runner. Every trained method uses the same
dataset configs, the same row loading, the same fold splits and the same
per-class AUC; a method supplies only how a fold is trained and scored.

Data comes from the canonical tables in `datasets/mapped090826/`:

    <dataset>_rows.csv.gz    row_index, interactor, partner, mutation, position,
                             wt_aa, mut_aa, perturbed, dataset, dataset_tier,
                             fragoza_source, source_row_id, cluster
    <dataset>_splits.csv.gz  seed, row_index, test_fold, test_class
    sequences.csv.gz         accession, sequence

built by `repro_test/build_canonical_tables.py` from the 090826 mapping. Every
mutation is 1-based and validated against its sequence, accessions are UniProt
(isoform suffix only where the sequence differs from canonical), there are no
duplicate (interactor, partner, mutation) triples and no null labels.

What used to be here and is deliberately gone: FASTA parsing, .pos/.neg
readers, per-source interaction TSVs and their inverted label polarity,
gene-symbol -> UniProt remapping, vt_id strings and the separator guessing they
needed, `align_to_vt_ids` padding, and the NaN handling that padding required.
None of it has an analogue in the canonical tables.
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from paths import DATASETS_DIR  # noqa: E402

TABLES = DATASETS_DIR / "mapped090826"

# Columns the predictors consume. The canonical rows table is already in this
# shape apart from the two sequence columns, which are joined by accession.
PREDICTOR_COLS = ["interactor", "partner", "mutation", "perturbed",
                  "interactor_sequence", "partner_sequence"]


@dataclass
class DatasetConfig:
    name: str
    rows_file: str
    splits_file: str


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    name: DatasetConfig(name, f"{name}_rows.csv.gz", f"{name}_splits.csv.gz")
    for name in (
        "sahni_fragoza_varchamp_all_mapped090826",
        "sahni_fragoza_mapped090826",
        "varchamp_all_mapped090826",
        "sahni_only_mapped090826",
        "fragoza_only_mapped090826",
    )
}

_SEQ_CACHE: dict | None = None


def load_sequences() -> dict:
    """accession -> sequence, read once per process."""
    global _SEQ_CACHE
    if _SEQ_CACHE is None:
        df = pd.read_csv(TABLES / "sequences.csv.gz")
        _SEQ_CACHE = dict(zip(df["accession"], df["sequence"]))
    return _SEQ_CACHE


def load_data(cfg: DatasetConfig) -> pd.DataFrame:
    """Canonical rows joined to their sequences, indexed by row_index.

    The result is exactly what the predictors consume, plus the provenance
    columns. No filtering happens here: the table is already validated, so a
    missing sequence is an error rather than a row to silently drop.
    """
    rows = pd.read_csv(TABLES / cfg.rows_file, index_col="row_index")
    seqs = load_sequences()
    missing = sorted({a for a in rows["interactor"] if a not in seqs} |
                     {b for b in rows["partner"] if b not in seqs})
    if missing:
        raise KeyError(f"{cfg.name}: {len(missing)} accessions absent from "
                       f"sequences.csv.gz, e.g. {missing[:5]}")
    rows["interactor_sequence"] = rows["interactor"].map(seqs)
    rows["partner_sequence"] = rows["partner"].map(seqs)
    rows["perturbed"] = rows["perturbed"].astype(int)
    return rows


def add_mutated_sequence(rows: pd.DataFrame) -> pd.DataFrame:
    """Append `mutated_sequence`: the interactor sequence with the variant applied.

    Not stored in the tables -- it is fully determined by `interactor_sequence`
    and `mutation`, and materialising it for every row would duplicate tens of MB.
    Only the embedding precomputes need it.
    """
    out = rows.copy()
    seqs, muts, pos = out["interactor_sequence"], out["mutation"], out["position"]
    out["mutated_sequence"] = [
        s[:p - 1] + m[-1] + s[p:] for s, m, p in zip(seqs, muts, pos)
    ]
    return out


def load_splits(cfg: DatasetConfig, seed: int):
    """(fold_splits, test_classes) for one GCV seed.

    fold_splits is [(fold, train_idx, test_idx), ...]; test_classes is the flat
    C1/C2/C3 array concatenated in fold order, matching how predictions and
    labels are accumulated in run_gcv.
    """
    sp = pd.read_csv(TABLES / cfg.splits_file)
    sp = sp[sp["seed"] == seed]
    if sp.empty:
        raise ValueError(f"{cfg.name}: no splits for seed {seed}")
    n = int(sp["row_index"].max()) + 1
    fold_splits, classes = [], []
    for fold, g in sp.groupby("test_fold", sort=True):
        test_idx = g["row_index"].to_numpy()
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
        fold_splits.append((int(fold), train_idx, test_idx))
        classes.append(g["test_class"].to_numpy())
    return fold_splits, np.concatenate(classes)


def _compute_class_aucs(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    pair_test_classes: np.ndarray,
    fold_n_test: list,
) -> tuple[list, np.ndarray, dict]:
    """Compute micro and weighted-macro AUROCs by pair_test_class (1/2/3).

    Returns:
        micro_auc         list of 3 AUROCs (over all samples per class)
        macro_auc         np.ndarray shape (3,) weighted per-fold mean
        fold_results      dict of per-fold results
    """
    # Mask NaN in EITHER preds or labels. This is the union of what the three
    # former implementations did: gcv_common masked preds only, mutpred_ppi_cv
    # masked nothing, swing_gcv masked both. It was number-preserving on the
    # historical results (zero NaNs across 14.8M values), but it is no longer
    # merely defensive: mutpred_ppi_gcv emits NaN by design for rows with no
    # structure or no embedding, so this mask is what excludes them. Those rows
    # are dropped from the denominator, which is why MutPred-PPI's per-class
    # counts are legitimately lower than the sequence-only methods'.
    valid_mask = ~np.isnan(all_preds) & ~np.isnan(all_labels)
    micro_auc = []
    for ptc in (1, 2, 3):
        mask = (pair_test_classes == ptc) & valid_mask
        print(f"  c{ptc}: {mask.sum()} preds (valid)", flush=True)
        if mask.sum() > 0 and len(np.unique(all_labels[mask])) > 1:
            auc = roc_auc_score(all_labels[mask], all_preds[mask])
        else:
            auc = float("nan")
        micro_auc.append(auc)

    class_auc_avgs = np.zeros(3)
    class_counts   = np.zeros(3, dtype=int)
    fold_results: dict = {}

    curr_idx = 0
    for fold, n_test in enumerate(fold_n_test):
        preds  = all_preds[curr_idx:curr_idx + n_test]
        labels = all_labels[curr_idx:curr_idx + n_test]
        ptcs   = pair_test_classes[curr_idx:curr_idx + n_test]
        curr_idx += n_test

        fold_res = {}
        print(f"\nfold {fold}", flush=True)
        fold_valid = ~np.isnan(preds) & ~np.isnan(labels)
        for ptc in (1, 2, 3):
            mask   = (ptcs == ptc) & fold_valid
            cp     = preds[mask]
            cl     = labels[mask]
            n_pos  = int((cl == 1).sum())
            n_neg  = int((cl == 0).sum())
            fold_res[f"class_{ptc}"] = {"preds": cp, "labels": cl, "auc": None}

            if n_pos > 0 and n_neg > 0:
                auc = roc_auc_score(cl, cp)
                fold_res[f"class_{ptc}"]["auc"] = auc
                class_auc_avgs[ptc - 1] += len(cp) * auc
                class_counts[ptc - 1]   += len(cp)
                print(f"  c{ptc} (n={len(cp)}): AUC-ROC={auc:.4f}", flush=True)
            else:
                print(f"  c{ptc} (n={len(cp)}): SKIPPED (pos={n_pos}, neg={n_neg})",
                      flush=True)

        fold_results[fold] = fold_res

    macro_auc = np.where(class_counts > 0, class_auc_avgs / class_counts, np.nan)
    return micro_auc, macro_auc, fold_results


# ── shared GCV runner ─────────────────────────────────────────────────────────


def default_fit_predict_fold(train_df, test_df, *, make_predictor, args, fold):
    """The fold body shared by eSIG-Net, MINT and PPLM."""
    print(f"\nFold {fold}: {len(train_df)} train / {len(test_df)} test — "
          f"fitting {getattr(args, 'predictor', 'model')}...", flush=True)
    predictor = make_predictor(args)
    predictor.fit(train_df[PREDICTOR_COLS])
    return predictor.predict(test_df[PREDICTOR_COLS])


def run_gcv(cfg, args, *, result_stem, make_predictor=None, preflight=None,
            fit_predict_fold=None):
    """One GCV implementation for every trained method.

    A method supplies only how a fold is trained and scored: `make_predictor`
    (a BasePredictor factory) or, where the training loop genuinely differs,
    `fit_predict_fold` -- SWING augments with WT rows, MutPred-PPI feeds graph
    tensors. Everything else is shared, including the log format.
    """
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {cfg.name}", flush=True)
    rows = load_data(cfg)
    print(f"  {len(rows)} rows", flush=True)

    if preflight is not None:
        preflight(rows, cfg, args)

    labels = rows["perturbed"].to_numpy()

    if fit_predict_fold is None:
        if make_predictor is None:
            raise ValueError("run_gcv needs either make_predictor or fit_predict_fold")

        def fit_predict_fold(train_df, test_df, *, fold, **_):
            return default_fit_predict_fold(train_df, test_df,
                                            make_predictor=make_predictor,
                                            args=args, fold=fold)

    macro_aucs, micro_aucs = [], []
    detailed_results = {"iterations": {}}
    start_gcv = 0

    # Resume from a partial run: a 30-seed GCV is hours to days, so an
    # interrupted one should not start over. Seeds are independent and each is
    # checkpointed on completion, so resuming reproduces the same artifact.
    ckpt = outdir / f"{result_stem}_detailed_results.pkl"
    if ckpt.exists() and getattr(args, "resume", True):
        with open(ckpt, "rb") as f:
            detailed_results = pickle.load(f)
        completed = sorted(detailed_results["iterations"])
        if completed:
            start_gcv = max(completed) + 1
            micro_aucs = list(np.load(outdir / f"{result_stem}_micro_aucs.npy"))
            macro_aucs = list(np.load(outdir / f"{result_stem}_macro_aucs.npy"))
            print(f"Resuming {result_stem}: {len(completed)} seeds already done, "
                  f"continuing from seed {start_gcv}. Delete "
                  f"{ckpt.name} (or pass --no-resume) to start over -- these "
                  f"results were produced by whatever code was current then.",
                  flush=True)
    if start_gcv >= args.n_gcv:
        print(f"{result_stem}: all {args.n_gcv} seeds already complete.", flush=True)
        return

    for gcv_seed in range(start_gcv, args.n_gcv):
        print(f"\n{'='*60}\nGCV seed {gcv_seed}/{args.n_gcv - 1}", flush=True)
        fold_splits, test_classes = load_splits(cfg, gcv_seed)
        fold_n_test = [len(t) for _, _, t in fold_splits]

        all_preds, all_labels = [], []
        for fold, train_idx, test_idx in fold_splits:
            preds = fit_predict_fold(
                rows.iloc[train_idx].reset_index(drop=True),
                rows.iloc[test_idx].reset_index(drop=True),
                fold=fold, gcv_seed=gcv_seed,
                train_idx=train_idx, test_idx=test_idx)
            preds = np.asarray(preds, dtype=float)
            if len(preds) != len(test_idx):
                raise ValueError(f"fold {fold}: {len(preds)} predictions for "
                                 f"{len(test_idx)} test rows")
            all_preds.extend(preds.tolist())
            all_labels.extend(labels[test_idx].tolist())
            print(f"  Fold {fold} done", flush=True)

        print(f"\nGCV seed {gcv_seed} — per-class AUROCs:", flush=True)
        micro_auc, macro_auc, fold_results = _compute_class_aucs(
            np.array(all_preds), np.array(all_labels, dtype=float),
            test_classes, fold_n_test)
        print(f"micro AUC (c1/c2/c3): {micro_auc}", flush=True)
        print(f"macro AUC (c1/c2/c3): {macro_auc}", flush=True)

        micro_aucs.append(micro_auc)
        macro_aucs.append(macro_auc)
        detailed_results["iterations"][gcv_seed] = {
            "folds": fold_results, "micro_auc": micro_auc, "macro_auc": macro_auc}

        # Checkpoint every seed, not just at the end, so an interrupted run
        # resumes from the last completed seed.
        _save_gcv(outdir, result_stem, micro_aucs, macro_aucs, detailed_results)

    print(f"\nResults saved to {outdir}/  ({result_stem}_*)", flush=True)


def _save_gcv(outdir, result_stem, micro_aucs, macro_aucs, detailed_results):
    np.save(outdir / f"{result_stem}_micro_aucs.npy", np.array(micro_aucs))
    np.save(outdir / f"{result_stem}_macro_aucs.npy", np.array(macro_aucs))
    with open(outdir / f"{result_stem}_detailed_results.pkl", "wb") as f:
        pickle.dump(detailed_results, f)
