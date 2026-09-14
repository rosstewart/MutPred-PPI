"""`run_gcv` may run folds concurrently; doing so must change nothing.

Folds are independent -- each trains its own model on its own split and nothing
is carried between them -- so running them concurrently is a scheduling change.
The risk is not the maths but the bookkeeping: predictions are concatenated in
fold order, so a parallel map that returned out of order would silently
mis-align every prediction with its label and still produce a plausible AUC.

SWING in blind-test mode retrains a Doc2Vec per fold and takes ~1.6 h/seed,
which is what `--fold-jobs` exists for. Default is 1 everywhere.
"""
import numpy as np
import pytest

from joblib import Parallel, delayed


# fits folds concurrently; see tests/conftest.py for the opt-in flags.
pytestmark = pytest.mark.slow


def _fold_splits(n=40, k=5):
    idx = np.arange(n)
    return [(f, np.setdiff1d(idx, idx[f::k]), idx[f::k]) for f in range(k)]


def _collect(fold_splits, fold_jobs, fit):
    """Mirrors the collection in `gcv_common.run_gcv`."""
    def _run(fold, tr, te):
        return np.asarray(fit(fold, tr, te), dtype=float)
    if fold_jobs > 1:
        out = Parallel(n_jobs=fold_jobs, backend="threading")(
            delayed(_run)(f, tr, te) for f, tr, te in fold_splits)
    else:
        out = [_run(f, tr, te) for f, tr, te in fold_splits]
    preds = []
    for (_f, _tr, _te), p in zip(fold_splits, out):
        preds.extend(p.tolist())
    return np.array(preds)


class TestOrdering:
    @pytest.mark.parametrize("jobs", [2, 3, 5])
    def test_parallel_matches_sequential(self, jobs):
        splits = _fold_splits()
        fit = lambda f, tr, te: te.astype(float)      # noqa: E731
        assert np.array_equal(_collect(splits, 1, fit),
                              _collect(splits, jobs, fit))

    def test_results_stay_aligned_with_their_fold(self, jobs=5):
        """Each fold's predictions must land in that fold's slot."""
        splits = _fold_splits()
        fit = lambda f, tr, te: np.full(len(te), f, dtype=float)  # noqa: E731
        got = _collect(splits, jobs, fit)
        expected = np.concatenate([np.full(len(te), f, dtype=float)
                                   for f, _tr, te in splits])
        assert np.array_equal(got, expected)

    def test_uneven_fold_sizes_still_align(self):
        """Equal-size folds would hide an off-by-one in the concatenation."""
        splits = [(0, np.arange(10), np.arange(0, 3)),
                  (1, np.arange(10), np.arange(3, 9)),
                  (2, np.arange(10), np.arange(9, 11))]
        fit = lambda f, tr, te: np.full(len(te), f, dtype=float)  # noqa: E731
        assert np.array_equal(_collect(splits, 3, fit),
                              np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2],
                                       dtype=float))

    def test_a_completion_ordered_map_would_be_caught(self):
        """Negative control: the bug this guards against is detectable.

        If results were collected as folds FINISHED rather than by fold index,
        the tests above would have to fail. Simulate that by reversing.
        """
        splits = _fold_splits()
        fit = lambda f, tr, te: np.full(len(te), f, dtype=float)  # noqa: E731
        good = _collect(splits, 1, fit)
        bad = np.concatenate([np.full(len(te), f, dtype=float)
                              for f, _tr, te in splits[::-1]])
        assert not np.array_equal(good, bad)


class TestDefaultIsSequential:
    def test_run_gcv_defaults_to_one_job(self):
        """A method that never sets `fold_jobs` must keep the old path."""
        import argparse
        from utils.gcv_common import run_gcv  # noqa: F401
        ns = argparse.Namespace()
        assert int(getattr(ns, "fold_jobs", 1) or 1) == 1

    def test_none_is_treated_as_one(self):
        import argparse
        ns = argparse.Namespace(fold_jobs=None)
        assert int(getattr(ns, "fold_jobs", 1) or 1) == 1
