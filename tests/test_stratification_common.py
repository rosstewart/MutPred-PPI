"""The GCV fold-curve reconstruction shared by the stratified supplements.

This is the first test coverage for anything in `src/analysis/`, and it targets
the piece most worth pinning: the interleave that maps a fold's per-class
prediction buckets back onto canonical row order. Three scripts each carried
their own copy of it (`protein_class_stratification`, `plddt_stratification`,
`interface_analysis`); two had dropped the guards the third kept.

Everything here runs on synthetic fixtures -- the real inputs are multi-GB GCV
pickles that no clone has.
"""
import pickle

import numpy as np
import pytest

from analysis import stratification_common as sc


def _write_fixture(tmp_path, n_rows=40, n_folds=4, n_seeds=2, classes=None):
    """A self-consistent CV reference + GCV pkl for `n_rows` rows."""
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = pd.DataFrame({"row_index": range(n_rows),
                         "interactor": [f"P{i:05d}" for i in range(n_rows)],
                         "partner": [f"Q{i:05d}" for i in range(n_rows)]})
    rows_file = tmp_path / "rows.csv.gz"
    rows.to_csv(rows_file, index=False)

    if classes is None:
        classes = rng.integers(1, 4, size=n_rows)
    folds = np.arange(n_rows) % n_folds

    iterations = []
    for _seed in range(n_seeds):
        fold_entries = {}
        for f in range(n_folds):
            test_idx = np.where(folds == f)[0]
            buckets = {}
            for c in (1, 2, 3):
                sel = test_idx[classes[test_idx] == c]
                buckets[f"class_{c}"] = {
                    "preds": [float(i) for i in sel],      # pred == row_index
                    "labels": [int(i % 2) for i in sel],
                }
            fold_entries[f] = buckets
        iterations.append({"folds": fold_entries})

    gcv_pkl = tmp_path / "gcv.pkl"
    with open(gcv_pkl, "wb") as fh:
        pickle.dump({"iterations": iterations}, fh)

    cv = tmp_path / "cv"
    cv.mkdir()
    for seed in range(n_seeds):
        with open(cv / f"sahni_fragoza_train_fold_splits_{seed}.pkl", "wb") as fh:
            pickle.dump([(f, np.where(folds != f)[0], np.where(folds == f)[0])
                         for f in range(n_folds)], fh)
        ordered = np.concatenate([classes[np.where(folds == f)[0]]
                                  for f in range(n_folds)])
        np.save(cv / f"swing_train_pair_test_classes_{seed}.npy", ordered)
    return rows_file, gcv_pkl, cv, classes, folds


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    rows_file, gcv_pkl, cv, classes, folds = _write_fixture(tmp_path)
    monkeypatch.setattr(sc, "CV_DIR", str(cv))
    monkeypatch.setattr(sc, "load_gcv_detailed_results",
                        lambda path, dataset: pickle.load(open(path, "rb")))
    return rows_file, gcv_pkl, cv, classes, folds


def test_interleave_recovers_row_identity(fixture):
    """preds were set to row_index, so every row must come back to its own group."""
    rows_file, gcv_pkl, _cv, classes, _folds = fixture
    n = len(classes)
    groups = np.array(["even" if i % 2 == 0 else "odd" for i in range(n)], dtype=object)

    _curves, by_group = sc.stratified_fold_curves(
        groups, ["even", "odd"], min_n=1, n_seeds=2,
        rows_file=str(rows_file), gcv_results=str(gcv_pkl))

    for cls in (1, 2, 3):
        for grp in ("even", "odd"):
            for ridx in by_group[cls][grp]:
                assert classes[ridx] == cls
                assert groups[ridx] == grp


def test_every_row_is_assigned_exactly_once_per_seed(fixture):
    rows_file, gcv_pkl, _cv, classes, _ = fixture
    n = len(classes)
    groups = np.array(["g"] * n, dtype=object)
    _c, by_group = sc.stratified_fold_curves(
        groups, ["g"], min_n=1, n_seeds=2,
        rows_file=str(rows_file), gcv_results=str(gcv_pkl))
    seen = set()
    for cls in (1, 2, 3):
        seen |= by_group[cls]["g"]
    assert seen == set(range(n))


def test_rows_outside_the_group_list_are_ignored(fixture):
    """`None` (unknown) rows must never be selected, not silently bucketed."""
    rows_file, gcv_pkl, _cv, classes, _ = fixture
    n = len(classes)
    groups = np.array(["known" if i < 10 else None for i in range(n)], dtype=object)
    _c, by_group = sc.stratified_fold_curves(
        groups, ["known"], min_n=1, n_seeds=1,
        rows_file=str(rows_file), gcv_results=str(gcv_pkl))
    picked = set().union(*(by_group[c]["known"] for c in (1, 2, 3)))
    assert picked <= set(range(10))


def test_row_index_must_be_positional(tmp_path):
    import pandas as pd
    bad = tmp_path / "bad.csv.gz"
    pd.DataFrame({"row_index": [5, 6, 7]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="row_index is not 0"):
        sc.load_canonical_rows(str(bad))


def test_fold_coverage_mismatch_is_named(fixture):
    """A CV reference that does not cover the table must not index silently."""
    rows_file, gcv_pkl, _cv, classes, _ = fixture
    groups = np.array(["g"] * (len(classes) - 1), dtype=object)   # wrong length
    with pytest.raises(ValueError, match="row_groups has"):
        sc.stratified_fold_curves(groups, ["g"], rows_file=str(rows_file),
                                  gcv_results=str(gcv_pkl))


def test_stale_fold_bucket_raises_stalecacheerror(tmp_path, monkeypatch):
    """A fold whose cached buckets do not match its test size is named, not IndexError."""
    rows_file, gcv_pkl, cv, classes, _ = _write_fixture(tmp_path)
    with open(gcv_pkl, "rb") as fh:
        data = pickle.load(fh)
    data["iterations"][0]["folds"][0]["class_1"]["preds"].append(999.0)   # one too many
    with open(gcv_pkl, "wb") as fh:
        pickle.dump(data, fh)
    monkeypatch.setattr(sc, "CV_DIR", str(cv))
    monkeypatch.setattr(sc, "load_gcv_detailed_results",
                        lambda path, dataset: pickle.load(open(path, "rb")))
    groups = np.array(["g"] * len(classes), dtype=object)
    with pytest.raises(sc.StaleCacheError, match="internally inconsistent"):
        sc.stratified_fold_curves(groups, ["g"], min_n=1, n_seeds=1,
                                  rows_file=str(rows_file), gcv_results=str(gcv_pkl))


def test_mean_curve_is_pinned_at_both_ends():
    curves = [np.linspace(0, 1, len(sc.FPR_GRID)) for _ in range(3)]
    mean, sem, area, n = sc.mean_curve_and_auc(curves)
    assert n == 3 and mean[0] == 0.0 and mean[-1] == 1.0
    assert 0.0 <= area <= 1.0
    assert np.allclose(sem, 0.0)


def test_mean_curve_handles_no_folds():
    mean, sem, area, n = sc.mean_curve_and_auc([])
    assert mean is None and sem is None and n == 0 and np.isnan(area)


# ── class_roc_auc: one NaN policy for every one-off ROC ────────────────────────

def test_class_roc_auc_masks_nan_in_preds():
    from analysis.gcv_curves import class_roc_auc
    preds = np.array([0.1, 0.9, np.nan, 0.8, 0.2])
    labels = np.array([0, 1, 1, 1, 0])
    fpr, tpr, a = class_roc_auc(preds, labels)
    # the NaN row is excluded; remaining 4 rows are perfectly separable
    assert a == 1.0


def test_class_roc_auc_masks_nan_in_labels_too():
    from analysis.gcv_curves import class_roc_auc
    preds = np.array([0.1, 0.9, 0.5, 0.8, 0.2])
    labels = np.array([0, 1, np.nan, 1, 0])
    _, _, a = class_roc_auc(preds, labels)
    assert a == 1.0


def test_class_roc_auc_no_curve_when_one_label_class():
    from analysis.gcv_curves import class_roc_auc
    fpr, tpr, a = class_roc_auc(np.array([0.1, 0.2]), np.array([1, 1]))
    assert fpr is None and tpr is None and np.isnan(a)


def test_class_roc_auc_no_curve_when_all_nan():
    from analysis.gcv_curves import class_roc_auc
    fpr, tpr, a = class_roc_auc(np.array([np.nan, np.nan]), np.array([0, 1]))
    assert fpr is None and np.isnan(a)
