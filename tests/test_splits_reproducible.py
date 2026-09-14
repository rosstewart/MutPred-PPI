"""The shipped GCV splits tables must be exactly regenerable.

`{dataset}_splits.csv.gz` assigns every row to a test fold per seed, and every
`detailed_results.pkl` in the repo is joined to it BY ROW INDEX. So a regenerated
table that differs is not a cosmetic difference -- it silently re-labels which
predictions belong to which fold and class.

This nearly happened: `GroupKFold(shuffle=True)` orders the unique group labels
to assign them to folds, and the `cluster` column round-trips through read_csv as
int64 while the shipped tables were built from string labels. `np.unique` sorts
strings lexicographically ("10" < "2") but integers numerically, so the same
clusters and the same seed produced a different -- equally valid -- partition.
Re-running the producer would have replaced the tables and invalidated every GCV
figure, with nothing failing.

These tests regenerate and require an exact match.
"""
import numpy as np
import pandas as pd
import pytest

from data_processing.training_sets.prepare_gcv_tables import build_splits
from utils.gcv_common import DATASET_CONFIGS, TABLES, dataset_config, load_data

# smallest first; the big pooled table is slow and adds no coverage
DATASETS = ["sahni_only_mapped090826", "sahni_fragoza_mapped090826"]
N_SEEDS = 3


def _shipped(cfg):
    path = TABLES / cfg.splits_file
    if not path.exists():
        pytest.skip(f"{path} not present")
    return pd.read_csv(path)


@pytest.mark.parametrize("ds", DATASETS)
def test_regenerating_reproduces_the_shipped_table(ds):
    cfg = dataset_config(ds)
    shipped = _shipped(cfg)
    rows = load_data(cfg).reset_index(drop=True)
    fresh = build_splits(rows, n_seeds=N_SEEDS)
    for seed in range(N_SEEDS):
        a = shipped[shipped.seed == seed].sort_values("row_index")
        b = fresh[fresh.seed == seed].sort_values("row_index")
        assert len(a) == len(b) == len(rows)
        assert np.array_equal(a.test_fold.values, b.test_fold.values), (
            f"{ds} seed {seed}: regenerated folds differ from the shipped table "
            f"({int((a.test_fold.values != b.test_fold.values).sum())} rows). "
            f"A different partition invalidates every result joined to it.")
        assert np.array_equal(a.test_class.values, b.test_class.values), \
            f"{ds} seed {seed}: regenerated C1/C2/C3 differ"


@pytest.mark.parametrize("ds", DATASETS)
def test_integer_labels_would_give_a_different_partition(ds):
    """Negative control: the bug this guards against is real and silent.

    If int vs str made no difference, the test above would pass for the wrong
    reason and the cast could be removed without anything failing.
    """
    from sklearn.model_selection import GroupKFold
    rows = load_data(dataset_config(ds)).reset_index(drop=True)
    n = len(rows)
    folds = {}
    for kind, g in (("str", rows["cluster"].astype(str).values),
                    ("int", rows["cluster"].values)):
        arr = np.empty(n, dtype=int)
        kf = GroupKFold(n_splits=10, shuffle=True, random_state=0)
        for f, (_tr, te) in enumerate(kf.split(range(n), groups=g)):
            arr[te] = f
        folds[kind] = arr
    assert not np.array_equal(folds["str"], folds["int"]), (
        "int and str group labels gave the same folds -- the cast in "
        "build_splits would then be untested by the test above")


@pytest.mark.parametrize("ds", DATASETS)
def test_shipped_table_is_a_valid_grouped_partition(ds):
    """Coverage and the no-leakage guarantee, independent of how it was made."""
    cfg = dataset_config(ds)
    shipped = _shipped(cfg)
    rows = load_data(cfg).reset_index(drop=True)
    clusters = rows["cluster"].to_numpy()
    for seed in sorted(shipped.seed.unique())[:N_SEEDS]:
        a = shipped[shipped.seed == seed].sort_values("row_index")
        assert np.array_equal(a.row_index.values, np.arange(len(rows))), \
            f"{ds} seed {seed}: rows not covered exactly once"
        straddling = (pd.DataFrame({"c": clusters, "f": a.test_fold.values})
                      .groupby("c")["f"].nunique() > 1).sum()
        assert straddling == 0, \
            f"{ds} seed {seed}: {straddling} clusters straddle folds (leakage)"


def test_every_dataset_has_a_splits_table():
    missing = [n for n, cfg in DATASET_CONFIGS.items()
               if not (TABLES / cfg.splits_file).exists()]
    assert not missing, f"no splits table for: {missing}"


class TestOneSourceOfTruth:
    """Every GCV-derived figure must resolve folds through the same table.

    The chain is: `{dataset}_splits.csv.gz` -> `load_splits()` ->
    `datasets/cv_reference/` -> the figures that need per-row identity
    (biclass, the pLDDT / interface / protein-class stratifications,
    the reconstruction tables). The ROC comparison and ablation figures read
    fold structure straight out of each `detailed_results.pkl` and never touch
    the reference at all.

    `cv_reference` used to re-derive its own folds, which disagreed with the
    table in 894 of 900 fold-class cells -- silently, because the consumer
    zero-filled any fold whose row count did not line up.
    """

    @pytest.mark.parametrize("ds", DATASETS)
    def test_cv_reference_matches_load_splits(self, ds):
        import pickle
        from pathlib import Path
        from analysis.export_cv_reference import NAMING, OUT_DIR
        from utils.gcv_common import load_splits

        prefix, ptc_prefix = NAMING[ds]
        cfg = dataset_config(ds)
        for seed in range(N_SEEDS):
            fs_path = Path(OUT_DIR) / f"{prefix}fold_splits_{seed}.pkl"
            ptc_path = Path(OUT_DIR) / f"{ptc_prefix}pair_test_classes_{seed}.npy"
            if not (fs_path.exists() and ptc_path.exists()):
                pytest.skip(f"cv_reference not generated for {ds}")
            live_fs, live_ptc = load_splits(cfg, seed)
            ref_fs = pickle.load(open(fs_path, "rb"))
            ref_ptc = np.load(ptc_path)

            live_sorted = sorted(live_fs, key=lambda t: t[0])
            ref_sorted = sorted(ref_fs, key=lambda t: t[0])
            assert len(live_sorted) == len(ref_sorted)
            for (lf, _ltr, lte), (rf, _rtr, rte) in zip(live_sorted, ref_sorted):
                assert lf == rf
                assert np.array_equal(np.asarray(lte), np.asarray(rte)), (
                    f"{ds} seed {seed} fold {lf}: cv_reference test indices "
                    f"differ from load_splits -- predictions would be joined to "
                    f"the wrong rows")
            assert np.array_equal(np.asarray(live_ptc), ref_ptc), \
                f"{ds} seed {seed}: cv_reference test classes differ"

    @pytest.mark.parametrize("ds", DATASETS)
    def test_reference_row_table_matches_the_canonical_one(self, ds):
        from pathlib import Path
        from analysis.export_cv_reference import NAMING, OUT_DIR
        prefix, _ = NAMING[ds]
        p = Path(OUT_DIR) / f"{prefix}rows.csv.gz"
        if not p.exists():
            pytest.skip(f"cv_reference rows not generated for {ds}")
        ref = pd.read_csv(p)
        rows = load_data(dataset_config(ds)).reset_index(drop=True)
        assert len(ref) == len(rows)
        for col in ("interactor", "partner", "mutation"):
            assert np.array_equal(ref[col].astype(str).values,
                                  rows[col].astype(str).values), \
                f"{ds}: cv_reference '{col}' differs from the canonical table"
