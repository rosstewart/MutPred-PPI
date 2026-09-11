#!/usr/bin/env python
"""Run the VarChAMP blind test for a single comparator method.

Train on `sahni_fragoza_mapped090826`, predict on all of
`varchamp_all_mapped090826` -- the two canonical datasets this blind test has
always meant, before a namespace bug and a since-archived helper module
(`vcfp_common.py`) grew a THIRD, separately-built table
(`datasets/sfvcfp_rows.csv.gz`, via `repro_test/build_sfvcfp_table.py`) that
mixed pre-090826 sources (`data_caches/training_data_internal.csv`) with the
canonical ones. That table and its producer are retired: `build_sfvcfp_table`
no longer even imports (its `vcfp_common` dependency is archived), and
everything under the `vcfp`/`sfvcfp`/`varchamp_full_pooled` name family is a
retired dataset per `utils.legacy_guard`. There are only three live GCV
datasets that matter here: `sahni_only_mapped090826`, `sahni_fragoza_mapped090826`
(the training set), and `varchamp_all_mapped090826` (the blind-test target).

C1/C2/C3 classing is NOT one rule for every method:
  - Trained methods (MutPred-PPI, eSIG-Net, MINT, PPLM, SWING): protein
    overlap with the Sahni+Fragoza TRAINING set
    (`gcv_common.compute_blind_test_classes`).
  - SKEMPI-pretrained methods (SAAMBE-3D, MutPPI, MutPPI+): protein overlap
    with SKEMPI, their own (unrelated) training set
    (`gcv_common.skempi_test_class`) -- identical to how they are classed in
    GCV (`roc_plots.py`).
  - MutPred2: constant across C1/C2/C3 (partner-agnostic); handled entirely
    by `import_mutpred2_varchamp_scores.py`, not here.
DDMutPPI is not a method here at all -- excluded outright, see
the method's own module docstring.

Usage:
    conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method mutpredppi [--device cuda:1]
    conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method esignet [--device cuda:1] [--seed 42]
    conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method mint --predictor seq_diff [--seed 42]
    conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method pplm --predictor site_diff [--seed 42]
    conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method swing [--test-pretrain] [--seed 42]
    conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method saambe3d
    conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method mutppi
    conda run -n ppi python src/evaluation/run_varchamp_blind_test.py --method mutppiplus

For MutPred2 (external tool, no model to train), see
src/analysis/import_mutpred2_varchamp_scores.py instead.
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_EVAL_DIR = Path(__file__).resolve().parent          # src/evaluation
_PUB = _EVAL_DIR.parent.parent                       # repo root

from paths import DATASETS_DIR, REPO_ROOT, TRAINING_EVAL_DIR, VARCHAMP_BLIND_TEST_DIR, WEIGHTS_DIR  # noqa: E402
from utils import mutations  # noqa: E402
from utils.gcv_common import (dataset_config,   # noqa: E402
    DATASET_CONFIGS, PREDICTOR_COLS, compute_blind_test_classes, load_data,
    skempi_test_class)
from utils.legacy_guard import reject_legacy  # noqa: E402
from utils.structures import Structures  # noqa: E402

# Default training set is Sahni+Fragoza (the headline Fig 4 comparison).
# --train-dataset sahni_only_mapped090826 reproduces the S2 training-set
# comparison (does including population variants in fine-tuning help?).
DEFAULT_TRAIN_DATASET = "sahni_fragoza_mapped090826"
TEST_CFG  = DATASET_CONFIGS["varchamp_all_mapped090826"]
OUT_DIR   = VARCHAMP_BLIND_TEST_DIR
METHODS   = ("mutpredppi", "esignet", "mint", "pplm", "swing",
             "saambe3d", "mutppi", "mutppiplus")

# Methods pretrained on SKEMPI, not Sahni+Fragoza -- classed by SKEMPI overlap.
_SKEMPI_METHODS = {"saambe3d", "mutppi", "mutppiplus"}


def load_train_test(train_dataset: str = DEFAULT_TRAIN_DATASET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two canonical tables this blind test has always meant."""
    train_df = load_data(dataset_config(train_dataset))
    test_df  = load_data(TEST_CFG)
    return train_df, test_df


def save_results(
    scores: np.ndarray,
    test_df: pd.DataFrame,
    classes: np.ndarray,
    description: str,
    out_dir: Path = OUT_DIR,
) -> None:
    """Save per-class (C1/C2/C3) result arrays.

    `classes` is supplied by the caller (trained-method vs. SKEMPI overlap --
    see module docstring), never re-derived here, so a method cannot
    accidentally get the wrong classing rule silently.
    """
    labels = test_df["perturbed"].to_numpy().astype(int)
    vt_ids = (test_df["interactor"] + " " + test_df["partner"] + " "
              + test_df["mutation"]).to_numpy()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = np.asarray(scores, dtype=float)
    valid = ~np.isnan(scores)
    n_nan = int((~valid).sum())
    if n_nan:
        print(f"  WARNING: {n_nan}/{len(scores)} NaN scores (cache misses / missing graphs)")

    for c in (1, 2, 3):
        mask = (classes == c) & valid
        n_pos = int((labels[mask] == 1).sum())
        n_neg = int((labels[mask] == 0).sum())
        auc_str = (f"{roc_auc_score(labels[mask], scores[mask]):.4f}"
                   if n_pos >= 2 and n_neg >= 2 else "n/a")
        print(f"  C{c}: n={int(mask.sum())} (pos={n_pos}, neg={n_neg}) AUC={auc_str}")
        np.save(out_dir / f"{description}_c{c}_preds.npy",  scores[mask].astype(np.float32))
        np.save(out_dir / f"{description}_c{c}_labels.npy", labels[mask])
        np.save(out_dir / f"{description}_c{c}_vt_ids.npy", vt_ids[mask])

    print(f"Saved -> {out_dir}/{description}_c{{1,2,3}}_*.npy")


# ── merged embedding caches (train + test datasets don't share a cache file) ──

def _merge_pickle_caches(paths: list[Path], merged_path: Path) -> Path:
    """Union two per-dataset embedding-cache pickles into one file.

    MINT/PPLM/eSIG-Net caches are keyed by protein-pair/variant CONTENT, not
    row position, so a blind test spanning two datasets (each with its own
    `{dataset}_{mint,pplm,esm2}.pkl`) needs the union of both -- a predictor
    pointed at only the training dataset's cache would miss every test row.
    Cached by content: skipped if `merged_path` is newer than every input.
    """
    for p in paths:
        reject_legacy(p, check_mtime=False)
    if merged_path.exists() and all(
        merged_path.stat().st_mtime >= p.stat().st_mtime for p in paths if p.exists()
    ):
        return merged_path
    merged: dict = {}
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found -- precompute it first "
                f"(src/evaluation/precompute_{{mint,pplm}}_embeddings.py or "
                f"src/data_processing/precompute_esm2_datasets.py)")
        with open(p, "rb") as fh:
            merged.update(pickle.load(fh))
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_path, "wb") as fh:
        pickle.dump(merged, fh)
    return merged_path


def _merged_cache_dir() -> Path:
    d = OUT_DIR / "_merged_caches"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── per-method scoring ─────────────────────────────────────────────────────────

def _score_mutpredppi(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      device: str, seed: int, train_dataset: str) -> np.ndarray:
    """Train on all of Sahni+Fragoza (no fold split), predict on all of VarChAMP."""
    import torch

    from inference.pipeline.model_loader import load_model, model_predict_subgraph
    from training.train_fold import _MEGASCALE_SCALER_PATH
    from utils.mutpred_ppi_data import build_tensors

    ckpt_dir = WEIGHTS_DIR / "blind_test"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stem = f"MutPred-PPI_{train_dataset}_megascale_all"
    ckpt_path = ckpt_dir / f"{stem}_all.pt"

    if not ckpt_path.exists():
        print(f"Training MutPred-PPI on {train_dataset} (--no-cv) -> {ckpt_path}", flush=True)
        subprocess.run(
            [sys.executable, str(_PUB / "src" / "training" / "train_final_model.py"),
             "--dataset", train_dataset, "--ablation", "megascale_all",
             "--seed", str(seed), "--save-models-dir", str(ckpt_dir),
             "--device", device, "--no-cv"],
            check=True, cwd=str(_PUB))
    reject_legacy(ckpt_path)

    print(f"Building test tensors for {TEST_CFG.name} ({len(test_df)} rows)...", flush=True)
    t = build_tensors(test_df, TEST_CFG.name, require_complete=False)

    scaler = None
    if _MEGASCALE_SCALER_PATH.exists():
        import joblib
        scaler = joblib.load(_MEGASCALE_SCALER_PATH)

    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = load_model(ckpt_path, dev)

    n = len(test_df)
    scores = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if not t["usable"][i]:
            continue
        mut_diff = t["mut_diff"][i]
        if scaler is not None:
            mut_diff = scaler.transform(mut_diff.reshape(1, -1)).ravel()
        pred = model_predict_subgraph(t["node_emb"][i], t["edge_index"][i], [model],
                             t["mutation_idx"][i], mut_diff, dev)
        if pred is not None:
            scores[i] = float(np.asarray(pred).mean())
    print(f"  scored {int(t['usable'].sum())}/{n} rows "
          f"({Counter(t['reasons']) if t['reasons'] else 'no exclusions'})", flush=True)
    return scores


def _score_esignet(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   device: str, seed: int, train_dataset: str) -> np.ndarray:
    import evaluation.predictors.esignet as _esignet_mod
    from evaluation.predictors.esignet import ESigNetPredictor

    merged = _merge_pickle_caches(
        [TRAINING_EVAL_DIR / f"{train_dataset}_esm2.pkl",
         TRAINING_EVAL_DIR / f"{TEST_CFG.name}_esm2.pkl"],
        _merged_cache_dir() / f"esm2_train_{train_dataset}_test_va.pkl")
    _esignet_mod._ESM_CACHE_PATH = str(merged)

    predictor = ESigNetPredictor(seed=seed, device=device)
    predictor.fit(train_df[PREDICTOR_COLS])
    return np.asarray(predictor.predict(test_df[PREDICTOR_COLS]), dtype=float)


def _score_cachemlp(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    method: str, predictor_name: str, seed: int,
                    train_dataset: str) -> np.ndarray:
    """MINT or PPLM, seq_diff or site_diff."""
    if method == "mint":
        from evaluation.predictors.mint_mlp import MINTSeqDiff, MINTSiteDiff
        import evaluation.predictors.mint_mlp as _mod
        cache_stem = "mint"
    else:
        from evaluation.predictors.pplm_mlp import PPLMSeqDiff, PPLMSiteDiff
        import evaluation.predictors.pplm_mlp as _mod
        cache_stem = "pplm"

    PredictorClass = {
        ("mint", "seq_diff"): MINTSeqDiff if method == "mint" else None,
        ("mint", "site_diff"): MINTSiteDiff if method == "mint" else None,
        ("pplm", "seq_diff"): PPLMSeqDiff if method == "pplm" else None,
        ("pplm", "site_diff"): PPLMSiteDiff if method == "pplm" else None,
    }[(method, predictor_name)]

    merged = _merge_pickle_caches(
        [TRAINING_EVAL_DIR / f"{train_dataset}_{cache_stem}.pkl",
         TRAINING_EVAL_DIR / f"{TEST_CFG.name}_{cache_stem}.pkl"],
        _merged_cache_dir() / f"{cache_stem}_train_{train_dataset}_test_va.pkl")
    _mod.CACHE_PATH = str(merged)
    PredictorClass._cache_path = str(merged)

    predictor = PredictorClass(seed=seed)
    predictor.fit(train_df[PREDICTOR_COLS])
    return np.asarray(predictor.predict(test_df[PREDICTOR_COLS]), dtype=float)


def _score_swing(train_df: pd.DataFrame, test_df: pd.DataFrame,
                 test_pretrain: bool) -> np.ndarray:
    from evaluation.swing_gcv import _build_swing_df, _fold_features
    from evaluation.swing_common import build_d2v, build_wt_df
    from gensim.models.doc2vec import Doc2Vec
    from xgboost import XGBClassifier

    tr_swing = _build_swing_df(train_df)
    te_swing = _build_swing_df(test_df)
    if len(tr_swing) != len(train_df) or len(te_swing) != len(test_df):
        raise ValueError(
            "SWING dropped rows converting to its internal format; "
            "fold indices would misalign against labels")

    if test_pretrain:
        wt_train = build_wt_df(tr_swing)
        wt_train["_oidx"] = tr_swing.index.tolist()
        tr_swing = tr_swing.copy()
        tr_swing["_oidx"] = tr_swing.index.tolist()
        combined = pd.concat([tr_swing, te_swing, wt_train]).reset_index(drop=True)
        model = build_d2v(combined)
        # test_pretrain leaks test rows into the Doc2Vec fit -- reported as
        # "Test Pretrain" precisely to flag that, same as GCV.
        raise NotImplementedError(
            "SWING --test-pretrain blind test: Doc2Vec-on-everything path not "
            "wired up for the single train/test split yet; use the default "
            "blind-test mode.")

    X_train_mut, X_train_wt, X_test = _fold_features(tr_swing, te_swing)
    y_train_mut = train_df["perturbed"].to_numpy().astype(float)
    X_train = np.concatenate([X_train_mut, X_train_wt])
    y_train = np.concatenate([y_train_mut, np.zeros(len(X_train_wt))])

    from evaluation.swing_gcv import _XGB_DEPTH, _XGB_LR, _XGB_N_EST
    clf = XGBClassifier(n_estimators=_XGB_N_EST, max_depth=_XGB_DEPTH, learning_rate=_XGB_LR)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def _score_saambe3d(test_df: pd.DataFrame) -> np.ndarray:
    structures = Structures(pdb_cache=OUT_DIR / "_pdb_cache")
    scores = np.full(len(test_df), np.nan, dtype=np.float32)
    from evaluation.saambe3d_cv import _call_saambe
    for i, row in enumerate(test_df.itertuples()):
        wt, pos_int, mt = mutations.parse(row.mutation)
        pdb_path, chain = structures.find_pdb(
            interactor=row.interactor_sequence, partner=row.partner_sequence)
        if pdb_path is None:
            continue
        tmp_out = OUT_DIR / f"_tmp_saambe_{i}.txt"
        try:
            score, _binary = _call_saambe(pdb_path, chain, str(pos_int), wt, mt, "1", tmp_out)
            scores[i] = score
        except Exception as exc:
            print(f"  ERROR {row.interactor} {row.mutation}: {exc}", flush=True)
            if tmp_out.exists():
                tmp_out.unlink()
    return scores


def _score_mutppi(test_df: pd.DataFrame, model_variant: int) -> np.ndarray:
    """MutPPI (model_variant=0) or MutPPI+ (model_variant=1)."""
    from evaluation.mutppi_cv import score_rows
    return score_rows(test_df, model_variant)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--predictor", default="seq_diff", choices=("seq_diff", "site_diff"),
                    help="MINT/PPLM only.")
    ap.add_argument("--test-pretrain", action="store_true", help="SWING only.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:1", help="MutPred-PPI/eSIG-Net only.")
    ap.add_argument("--train-dataset", default=DEFAULT_TRAIN_DATASET,
                    choices=[d for d in DATASET_CONFIGS if d != TEST_CFG.name],
                    help="Trained methods only. Default sahni_fragoza_mapped090826 "
                         "(Fig 4). sahni_only_mapped090826 reproduces the S2 "
                         "training-set comparison.")
    args = ap.parse_args()

    print(f"Loading canonical train={args.train_dataset} / test={TEST_CFG.name}", flush=True)
    train_df, test_df = load_train_test(args.train_dataset)
    print(f"  train: {len(train_df)} rows   test: {len(test_df)} rows", flush=True)

    if args.method in _SKEMPI_METHODS:
        classes = np.array(
            [skempi_test_class(a, b) for a, b in
             zip(test_df["interactor"], test_df["partner"])])
    else:
        classes = compute_blind_test_classes(
            list(zip(train_df["interactor"], train_df["partner"])),
            list(zip(test_df["interactor"], test_df["partner"])))

    sahni_only = args.train_dataset == "sahni_only_mapped090826"
    train_tag = "(sahni train) " if sahni_only else "(Sahni+Fragoza train) "
    if args.method == "mutpredppi":
        scores = _score_mutpredppi(train_df, test_df, args.device, args.seed, args.train_dataset)
        tag = "sahni, megascale_all, all-data" if sahni_only else "megascale_all, all-data"
        description = f"MutPred-PPI ({tag}) (varchamp_blind_test)"
    elif args.method == "esignet":
        scores = _score_esignet(train_df, test_df, args.device, args.seed, args.train_dataset)
        description = f"eSIG-Net {train_tag}(varchamp_blind_test)"
    elif args.method in ("mint", "pplm"):
        scores = _score_cachemlp(train_df, test_df, args.method, args.predictor, args.seed,
                                 args.train_dataset)
        label = f"{args.method.upper()}_{args.predictor}"
        description = f"{label} {train_tag}(varchamp_blind_test)"
    elif args.method == "swing":
        scores = _score_swing(train_df, test_df, args.test_pretrain)
        mode = "test pretrain, " if args.test_pretrain else ""
        description = f"SWING ({mode}{train_tag.strip()}) (varchamp_blind_test)"
    elif args.method == "saambe3d":
        scores = _score_saambe3d(test_df)
        description = "SAAMBE-3D (Sahni+Fragoza train) (varchamp_blind_test)"
    elif args.method == "mutppi":
        scores = _score_mutppi(test_df, model_variant=0)
        description = "MutPPI (Sahni+Fragoza train) (varchamp_blind_test)"
    elif args.method == "mutppiplus":
        scores = _score_mutppi(test_df, model_variant=1)
        description = "MutPPIPlus (Sahni+Fragoza train) (varchamp_blind_test)"
    else:
        raise ValueError(f"Unknown method: {args.method!r}")

    save_results(scores, test_df, classes, description)
    print("Done.")


if __name__ == "__main__":
    main()
