#!/usr/bin/env python
"""Shared group-cross-validation infrastructure.

One representation, one loader, one runner. Every trained method uses the same
dataset configs, the same row loading, the same fold splits and the same
per-class AUC; a method supplies only how a fold is trained and scored.

Data comes from the canonical tables in `datasets/training_eval/`:

    <dataset>_rows.csv.gz    row_index, interactor, partner, mutation, position,
                             wt_aa, mut_aa, perturbed, dataset, dataset_tier,
                             fragoza_source, source_row_id, cluster
    <dataset>_splits.csv.gz  seed, row_index, test_fold, test_class
    sequences.csv.gz         accession, sequence

built by `src/data_processing/training_sets/prepare_gcv_tables.py` from the
090826 mapping (see docs/DATA_PREPARATION.md for the full chain). Every
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

from utils.legacy_guard import DATASET_SUFFIX
import pandas as pd
from sklearn.metrics import roc_auc_score

from paths import (ANNOTATIONS_DIR, DATASETS_DIR, GCV_RESULTS_DIR,  # noqa: E402
                   TRAINING_EVAL_DIR)
from utils.identifiers import bare_accession  # noqa: E402
from utils import mutations  # noqa: E402

TABLES = TRAINING_EVAL_DIR

# Columns the predictors consume. The canonical rows table is already in this
# shape apart from the two sequence columns, which are joined by accession.
PREDICTOR_COLS = ["interactor", "partner", "mutation", "perturbed",
                  "interactor_sequence", "partner_sequence"]


@dataclass
class DatasetConfig:
    name: str
    rows_file: str
    splits_file: str


# The five canonical datasets, as stable BASE names. The mapping-generation
# stamp is applied once, here, from `legacy_guard.DATASET_SUFFIX` -- so a future
# remapping is a one-line change followed by regenerating the tables, and every
# filename derived from a dataset name moves with it.
DATASET_BASES: tuple[str, ...] = (
    "sahni_fragoza_varchamp_all",
    "sahni_fragoza",
    "varchamp_all",
    "sahni_only",
    "fragoza_only",
)


def dataset_name(base: str) -> str:
    """Full canonical dataset name for a base name, e.g. sahni_only -> ..._mapped090826."""
    return base if base.endswith(DATASET_SUFFIX) else f"{base}{DATASET_SUFFIX}"


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    dataset_name(b): DatasetConfig(dataset_name(b),
                                   f"{dataset_name(b)}_rows.csv.gz",
                                   f"{dataset_name(b)}_splits.csv.gz")
    for b in DATASET_BASES
}

# Short, readable aliases for the five canonical datasets.
#
# The `_mapped090826` suffix is a mapping-run date stamp: it is meaningful in a
# FILENAME, where it distinguishes these tables from the pre-rebaseline ones, and
# meaningless to anyone typing a command. Every `--dataset` argument accepts
# either form, so the docs can say `--dataset sahni_fragoza` and a reader never
# has to learn what 090826 means.
DATASET_ALIASES = {b: dataset_name(b) for b in DATASET_BASES}

# What `--dataset` should offer: short names first, full names still valid.
DATASET_CHOICES = list(DATASET_ALIASES) + list(DATASET_CONFIGS)


def resolve_dataset(name: str) -> str:
    """Canonical dataset name from either the short alias or the full name."""
    if name in DATASET_CONFIGS:
        return name
    if name in DATASET_ALIASES:
        return DATASET_ALIASES[name]
    raise KeyError(
        f"unknown dataset {name!r}. Choose one of: "
        f"{', '.join(sorted(DATASET_ALIASES))} "
        f"(or the full *_mapped090826 form).")


def dataset_config(name: str) -> DatasetConfig:
    """`DatasetConfig` for a short alias or a full dataset name."""
    return DATASET_CONFIGS[resolve_dataset(name)]


def dataset_arg(value: str) -> str:
    """argparse `type=` converter: normalise `--dataset` to the canonical name.

    Use this with `choices=list(DATASET_CONFIGS)`. argparse applies `type`
    BEFORE checking `choices`, so a short alias is expanded first and then
    validated, and `args.dataset` is always the full `*_mapped090826` name.

    That matters because `args.dataset` is not only looked up as a config -- it
    is interpolated directly into cache filenames
    (`{dataset}_prott5.pkl`, `{dataset}_esm2.pkl`, ...) and passed to
    `build_tensors`. Normalising only at the config lookup, as the first version
    of these aliases did, left those paths pointing at
    `sahni_fragoza_prott5.pkl` -- a file that does not exist.
    """
    import argparse as _argparse
    try:
        return resolve_dataset(value)
    except KeyError as exc:
        raise _argparse.ArgumentTypeError(str(exc).strip('"')) from exc


_SEQ_CACHE: dict | None = None


class StaleCacheError(RuntimeError):
    """A cached prediction array or GCV result pkl is keyed to a row ordering
    that no longer exists.

    These artifacts are POSITIONAL: element i is the score for row i of
    whatever table was current when they were written. Once the dataset is
    re-derived -- the 090826 rebaseline changed every canonical row count --
    indexing them by the new ordering does not fail, it silently pairs each
    score with a different variant. This is the one definition every consumer
    imports, so the check cannot be skipped by a new caller and cannot
    silently diverge between two callers checking the same file two ways.
    """


def _gcv_row_count(detailed_results: dict) -> tuple[int, str]:
    """One seed's total row count from a `*_detailed_results.pkl`, and which
    key it was counted from.

    Schema-aware: a method's own detailed_results counts rows via `preds`; a
    few older per-fold pkls (e.g. an ipTM-derived byproduct) carry no `preds`
    key at all, only `complex_ids` -- that is counted instead. Either way this
    is a count of ROWS, not a value comparison, so the schema difference does
    not change what is being asserted.
    """
    it0 = next(iter(detailed_results["iterations"].values()))
    first_fold = next(iter(it0["folds"].values()))
    key = "preds" if "preds" in first_fold["class_1"] else "complex_ids"
    n_total = sum(len(fold_data[f"class_{c}"][key])
                  for fold_data in it0["folds"].values() for c in (1, 2, 3))
    return n_total, key


def assert_gcv_pkl_fresh(detailed_results: dict, canonical_dataset: str, *,
                         pkl_name: str = "<gcv pkl>") -> None:
    """Raise `StaleCacheError` unless one seed's total row count matches the
    canonical row count for `canonical_dataset`.

    The one check every GCV-pkl consumer must run before trusting a
    `*_detailed_results.pkl` it did not just write. It replaces four
    independently hand-rolled versions (`biclass_sf_gcv.py`,
    `interface_analysis.py`, `plddt_stratification.py`,
    `protein_class_stratification.py`) that each checked a DIFFERENT thing --
    a full seed's total, or a specific fold's class_3 count against a
    separately-loaded `pair_test_classes` slice -- with one implementation of
    the check that actually matters: does this pkl's row count match the
    canonical table's row count. It needs nothing but the pkl itself and the
    canonical dataset name, so it is always available even to a caller that
    has not (yet, or ever) loaded fold_splits/pair_test_classes for its own
    purposes.
    """
    n_canonical = len(load_data(DATASET_CONFIGS[canonical_dataset]))
    n_total, key = _gcv_row_count(detailed_results)
    if n_total != n_canonical:
        raise StaleCacheError(
            f"{pkl_name}: one seed's total `{key}` count ({n_total}) does not "
            f"match the canonical {canonical_dataset} row count ({n_canonical}). "
            f"This pkl was built on a superseded row ordering and must be "
            f"regenerated against the canonical tables before this analysis "
            f"can be trusted.")


def load_gcv_detailed_results(pkl_path, canonical_dataset: str) -> dict:
    """Load a `*_detailed_results.pkl`, raising `StaleCacheError` if stale.

    Load-and-check in one call, for the common case: a caller that just wants
    verified-fresh contents, not the check as a separate step.
    """
    with open(pkl_path, "rb") as f:
        detailed_results = pickle.load(f)
    assert_gcv_pkl_fresh(detailed_results, canonical_dataset,
                         pkl_name=Path(pkl_path).name)
    return detailed_results


def load_positional_cache(path, n_expected: int, *, default_fill: float | None = None):
    """A positional (per-row) `.npy` array, or `default_fill` if absent.

    Raises `StaleCacheError` if the array exists but its length disagrees with
    `n_expected` (the canonical row count) -- a positional array cannot be
    re-aligned onto a new row ordering, so a length mismatch means it was
    built against an ordering that no longer exists. Replaces the two
    independent hand-rolled versions in `roc_plots.py` (`_load_cached`) and
    `biclass_sf_gcv.py` (`filter_baseline_predictions`'s inline checks).
    """
    path = Path(path)
    if not path.exists():
        if default_fill is None:
            return None
        return np.full(n_expected, default_fill)
    arr = np.load(path)
    if len(arr) != n_expected:
        raise StaleCacheError(
            f"{path.name} holds {len(arr)} values but the canonical table has "
            f"{n_expected} rows. This array is positional, so it cannot be "
            f"re-aligned -- it must be recomputed on the canonical rows before "
            f"it can be used.")
    return arr


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


def union_rows_across_datasets() -> pd.DataFrame:
    """Concatenated rows from all five canonical datasets.

    `sahni_fragoza_varchamp_all_mapped090826` looks like a pre-built superset
    but is a POOLED, conflict-resolved table: a row can be dropped during
    pooling even though the same (interactor, partner, mutation) is present
    and unconflicted in a smaller dataset. Consumers that need every triple
    the five datasets collectively test (AF3 structure requirements, MutPred2
    queries) must union the raw tables themselves rather than trust the
    pooled one alone -- see `prepare_af3_inputs.py`'s module docstring for the
    verification that this matters.
    """
    return pd.concat([load_data(cfg) for cfg in DATASET_CONFIGS.values()],
                     ignore_index=True)


def add_mutated_sequence(rows: pd.DataFrame) -> pd.DataFrame:
    """Append `mutated_sequence`: the interactor sequence with the variant applied.

    Not stored in the tables -- it is fully determined by `interactor_sequence`
    and `mutation`, and materialising it for every row would duplicate tens of MB.
    Only the embedding precomputes need it.

    Goes through `utils.mutations.apply`, which verifies the wild-type residue
    before substituting. This used to splice inline off the `position` COLUMN
    (`s[:p-1] + m[-1] + s[p:]`) with no check at all -- a fifth private copy of
    the operation `utils/mutations.py` exists to own, and the one place it
    mattered most, since a wrong sequence here propagates silently into every
    ProtT5/ESM-2/MINT/PPLM embedding cache. Verified 2026-09-10 to be
    byte-identical on all 53,239 rows of the five canonical datasets, so this is
    a guard rather than a change; it only bites if `position` and `mutation`
    ever disagree, which is exactly the case the old form could not see.
    """
    out = rows.copy()
    applied = [mutations.apply(s, m)
               for s, m in zip(out["interactor_sequence"], out["mutation"])]
    bad = [(i, m) for i, (m, a) in enumerate(zip(out["mutation"], applied))
           if a is None]
    if bad:
        raise ValueError(
            f"{len(bad)} row(s) have a mutation that does not fit their "
            f"interactor sequence (wild-type mismatch or position out of "
            f"range), e.g. row {bad[0][0]} mutation {bad[0][1]!r}. The "
            f"canonical tables are validated at build time, so this means the "
            f"rows and sequences.csv.gz are out of sync -- rebuild with "
            f"src/data_processing/training_sets/prepare_gcv_tables.py.")
    out["mutated_sequence"] = applied
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
    # Mask NaN in EITHER preds or labels -- the union of what the three former
    # implementations did (gcv_common masked preds only, mutpred_ppi_cv masked
    # nothing, swing_gcv masked both).
    #
    # This is DEFENSIVE, and as of 2026-09-10 it should never fire in GCV.
    # `mutpred_ppi_gcv.py` builds tensors with `require_complete=True`, which
    # RAISES on any row lacking a structure or an embedding rather than scoring
    # it NaN; and rows whose complex has no AlphaFold3 structure are now dropped
    # at dataset-build time (`af3_failed`, see
    # src/data_processing/annotate_af3_coverage.py), so they never reach here at
    # all. A NaN surviving to this point means a genuine defect -- a stale
    # embedding cache, or a table built without the af3_failed filter -- so the
    # count printed below is worth reading rather than ignoring.
    #
    # (An earlier version of this comment claimed mutpred_ppi_gcv "emits NaN by
    # design". It does not, and never did: see its own comment at the
    # build_tensors call, "No NaN, no shrunken denominator.")
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

        def _run_fold(fold, train_idx, test_idx):
            preds = fit_predict_fold(
                rows.iloc[train_idx].reset_index(drop=True),
                rows.iloc[test_idx].reset_index(drop=True),
                fold=fold, gcv_seed=gcv_seed,
                train_idx=train_idx, test_idx=test_idx)
            preds = np.asarray(preds, dtype=float)
            if len(preds) != len(test_idx):
                raise ValueError(f"fold {fold}: {len(preds)} predictions for "
                                 f"{len(test_idx)} test rows")
            print(f"  Fold {fold} done", flush=True)
            return preds

        # Folds are independent -- each trains its own model on its own split and
        # nothing is carried between them -- so they may run concurrently. This is
        # OFF by default (`fold_jobs=1`), which keeps the loop exactly as it was
        # for every method; only a caller that opts in sees any difference. SWING
        # retrains a Doc2Vec per fold in blind-test mode and is the reason this
        # exists. The threading backend is deliberate: gensim releases the GIL in
        # its training loop, and threads share the already-loaded frames instead
        # of pickling them to subprocesses. Results are collected in fold order,
        # so `all_preds` is assembled identically either way.
        fold_jobs = int(getattr(args, "fold_jobs", 1) or 1)
        if fold_jobs > 1:
            from joblib import Parallel, delayed
            fold_preds = Parallel(n_jobs=fold_jobs, backend="threading")(
                delayed(_run_fold)(f, tr, te) for f, tr, te in fold_splits)
        else:
            fold_preds = [_run_fold(f, tr, te) for f, tr, te in fold_splits]

        all_preds, all_labels = [], []
        for (fold, train_idx, test_idx), preds in zip(fold_splits, fold_preds):
            all_preds.extend(preds.tolist())
            all_labels.extend(labels[test_idx].tolist())

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


# ── grouping and splits ───────────────────────────────────────────────────────
#
# Recovered from `mutpred_ppi_cv.py` when that file was reduced to `train_fold`.
# They belong here: this is the shared data layer, and every CV/GCV entry point
# needs the same grouping or the folds stop being comparable between methods.

def cluster_sequences(sequences: list, identity: float = 0.5) -> list:
    """cd-hit cluster ids, one per input sequence, in input order."""
    import os
    import subprocess
    import tempfile

    from paths import cdhit_binary

    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".fasta") as f:
        fasta_path = f.name
        for i, seq in enumerate(sequences):
            f.write(f">seq{i}\n{seq}\n")

    out_path = fasta_path + "_clustered"
    result = subprocess.run(
        [cdhit_binary(), "-i", fasta_path, "-o", out_path, "-c", str(identity),
         "-n", "3"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        # Returning [] here would surface much later as an opaque IndexError,
        # since these clusters are the GroupKFold groups.
        raise RuntimeError(
            f"cd-hit failed (exit {result.returncode}):\n{result.stderr.decode()}")

    clstr_path = out_path + ".clstr"
    cluster_map: dict[int, int] = {}
    cluster_id = 0
    with open(clstr_path) as f:
        for line in f:
            if line.startswith(">Cluster"):
                cluster_id = int(line.strip().split()[-1])
            else:
                idx = int(line.split(">seq")[1].split("...")[0])
                cluster_map[idx] = cluster_id

    for path in (fasta_path, out_path, clstr_path):
        if os.path.exists(path):
            os.remove(path)
    return [cluster_map[i] for i in range(len(sequences))]


def complex_clusters(rows, identity: float = 0.5) -> list:
    """GroupKFold groups for a canonical row table.

    Clusters on the FULL COMPLEX SEQUENCE -- interactor followed by partner --
    never on the interactor alone. Every published split was built that way, and
    grouping on one chain would let a near-duplicate complex straddle the
    train/test boundary.
    """
    complexes = [f"{a}{b}" for a, b in
                 zip(rows["interactor_sequence"], rows["partner_sequence"])]
    return cluster_sequences(complexes, identity=identity)


def make_fold_splits(clusters: list, gcv_seed: int, n_splits: int = 10) -> list:
    """[(fold, train_idx, test_idx)] grouped so a cluster never straddles a fold."""
    from sklearn.model_selection import GroupKFold

    # Labels are cast to str for the same reason as in `prepare_gcv_tables`:
    # GroupKFold(shuffle=True) orders unique labels to assign them to folds, and
    # string vs integer ordering ("10" < "2" but 10 > 2) yields a different --
    # equally valid, but different -- partition from the same seed.
    kf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=gcv_seed)
    groups = [str(c) for c in clusters]
    return [(fold, tr, te) for fold, (tr, te)
            in enumerate(kf.split(range(len(groups)), groups=groups))]


def compute_pair_test_classes(pairs: list, fold_splits: list) -> np.ndarray:
    """C1/C2/C3 per test sample, concatenated in fold order (fold 0 first).

    C1: both proteins seen in training; C2: one seen; C3: neither.

    `pairs` is an explicit list of `(interactor, partner)` tuples. It is never
    re-derived by splitting a `complex_id`: separators differ per source and an
    isoform accession contains the same `-` the composite uses, so 261 of 2,785
    ids in the sahni_fragoza reference would split wrongly.
    """
    classes: list[int] = []
    for _fold, train_idx, test_idx in fold_splits:
        train_proteins: set = set()
        for i in train_idx:
            a, b = pairs[i]
            train_proteins.add(a)
            train_proteins.add(b)
        for i in test_idx:
            a, b = pairs[i]
            a_seen, b_seen = a in train_proteins, b in train_proteins
            classes.append(1 if (a_seen and b_seen) else 2 if (a_seen or b_seen) else 3)
    return np.array(classes, dtype=np.int64)

def compute_blind_test_classes(train_pairs: list, test_pairs: list) -> np.ndarray:
    """C1/C2/C3 for a fixed train/test blind test (not k-fold).

    Same protein-overlap rule as `compute_pair_test_classes` -- C1: both
    proteins seen in `train_pairs`, C2: one, C3: neither -- generalized to two
    separate row sets instead of fold indices into one shared table. This is
    what the VarChAMP blind test uses: train on `sahni_fragoza_mapped090826`,
    predict on all of `varchamp_all_mapped090826`, class each test row by
    whether its interactor/partner appeared anywhere in the training pairs.

    Trained methods (MutPred-PPI, eSIG-Net, MINT, PPLM, SWING) use this.
    SKEMPI-pretrained methods (SAAMBE-3D, MutPPI, MutPPI+) use
    `skempi_test_class` instead -- their training set is SKEMPI, not
    Sahni+Fragoza. MutPred2 is partner-agnostic and gets a constant class
    (see `STRATIFICATION_INDEPENDENT_METHODS` in `blind_test_figures.py`).
    """
    train_proteins: set = set()
    for a, b in train_pairs:
        train_proteins.add(a)
        train_proteins.add(b)
    classes = []
    for a, b in test_pairs:
        a_seen, b_seen = a in train_proteins, b in train_proteins
        classes.append(1 if (a_seen and b_seen) else 2 if (a_seen or b_seen) else 3)
    return np.array(classes, dtype=np.int64)


_SKEMPI_TRAIN_UNIPROTS: set | None = None


def load_skempi_train_uniprots() -> set:
    """The SKEMPI 2.0 proteins SAAMBE-3D/MutPPI/MutPPI+ were pretrained on.

    These methods are not retrained per dataset, so their own training-set
    overlap is what defines their C1/C2/C3, in both GCV (roc_plots.py) and the
    VarChAMP blind test. Cached at module level -- every fold/row lookup in a
    GCV or blind-test run reads the same set.

    Derived from source by
    `src/data_processing/training_sets/prepare_skempi_reference.py`
    (SKEMPI 2.0 joined per chain against SIFTS), rather than being an opaque
    input. It used to be `results/gcv/SAAMBE_train_uniprots.npy` -- an INPUT
    living in a generated, gitignored directory, so no one cloning the repo
    could stratify these three methods at all. That file also held only 258
    accessions because its derivation kept only single-character chain groups,
    silently dropping all 122 multi-chain SKEMPI complexes (antibody H/L pairs
    and similar); the corrected derivation finds 342. Replaced 2026-09-10.
    """
    global _SKEMPI_TRAIN_UNIPROTS
    if _SKEMPI_TRAIN_UNIPROTS is None:
        path = ANNOTATIONS_DIR / "skempi_train_uniprots.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- build it with\n"
                f"  python src/data_processing/training_sets/prepare_skempi_reference.py")
        accs = set()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and line != "uniprot":
                    accs.add(line)
        _SKEMPI_TRAIN_UNIPROTS = accs
    return _SKEMPI_TRAIN_UNIPROTS


def skempi_test_class(interactor: str, partner: str, train_uniprots: set | None = None) -> int:
    """1 = both proteins in SKEMPI training, 2 = one, 3 = neither.

    `train_uniprots` holds parent accessions only, so the membership test is
    on `bare_accession` -- a lookup in a per-accession database that has no
    isoform entries. Pass `train_uniprots` explicitly to avoid re-loading the
    cache per row in a tight loop; omit it to use the shared cache.
    """
    if train_uniprots is None:
        train_uniprots = load_skempi_train_uniprots()
    n_seen = sum(bare_accession(p) in train_uniprots for p in (interactor, partner))
    return {2: 1, 1: 2, 0: 3}[n_seen]
