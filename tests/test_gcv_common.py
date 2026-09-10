"""Tests for the shared staleness-detection layer in utils/gcv_common.py.

`StaleCacheError`, `assert_gcv_pkl_fresh`, `load_gcv_detailed_results` and
`load_positional_cache` are the ONE implementation every GCV-pkl / positional
-cache consumer must use, replacing what used to be four independently
hand-rolled checks (biclass_sf_gcv.py, interface_analysis.py,
plddt_stratification.py, protein_class_stratification.py) plus two more for
positional npy arrays (roc_plots.py, biclass_sf_gcv.py).
"""
import numpy as np
import pytest

from utils.gcv_common import (
    StaleCacheError,
    _gcv_row_count,
    assert_gcv_pkl_fresh,
    load_gcv_detailed_results,
    load_positional_cache,
)


def _make_detailed_results(n_per_class: dict[int, int], key: str = "preds") -> dict:
    """A minimal one-seed, one-fold detailed_results dict."""
    folds = {}
    for cls, n in n_per_class.items():
        folds.setdefault(0, {})[f"class_{cls}"] = {key: list(range(n))}
    return {"iterations": {0: {"folds": folds}}}


# -- _gcv_row_count: schema awareness ---------------------------------------------

def test_gcv_row_count_sums_preds():
    dr = _make_detailed_results({1: 3, 2: 4, 3: 5})
    n, key = _gcv_row_count(dr)
    assert n == 12
    assert key == "preds"


def test_gcv_row_count_falls_back_to_complex_ids():
    dr = _make_detailed_results({1: 2, 2: 2, 3: 2}, key="complex_ids")
    n, key = _gcv_row_count(dr)
    assert n == 6
    assert key == "complex_ids"


# -- assert_gcv_pkl_fresh: the core check ------------------------------------------

def _patch_canonical_count(monkeypatch, dataset_name: str, n: int):
    """Make DATASET_CONFIGS[dataset_name] resolvable and load_data(...) return
    something of length n, without touching real canonical tables."""
    import utils.gcv_common as gcv_common

    monkeypatch.setitem(gcv_common.DATASET_CONFIGS, dataset_name, object())
    monkeypatch.setattr(gcv_common, "load_data", lambda cfg: list(range(n)))


def test_assert_gcv_pkl_fresh_passes_when_counts_match(monkeypatch):
    _patch_canonical_count(monkeypatch, "fake_dataset", 12)
    dr = _make_detailed_results({1: 3, 2: 4, 3: 5})
    assert_gcv_pkl_fresh(dr, "fake_dataset", pkl_name="fake.pkl")  # must not raise


def test_assert_gcv_pkl_fresh_raises_when_counts_disagree(monkeypatch):
    _patch_canonical_count(monkeypatch, "sahni_fragoza_mapped090826", 6219)
    dr = _make_detailed_results({1: 1000, 2: 1000, 3: 3894})  # sums to 5894, stale
    with pytest.raises(StaleCacheError, match="5894"):
        assert_gcv_pkl_fresh(dr, "sahni_fragoza_mapped090826", pkl_name="stale.pkl")


def test_assert_gcv_pkl_fresh_error_names_the_pkl_and_dataset(monkeypatch):
    _patch_canonical_count(monkeypatch, "some_dataset", 100)
    dr = _make_detailed_results({1: 1, 2: 1, 3: 1})
    with pytest.raises(StaleCacheError) as exc_info:
        assert_gcv_pkl_fresh(dr, "some_dataset", pkl_name="MyMethod_detailed_results.pkl")
    msg = str(exc_info.value)
    assert "MyMethod_detailed_results.pkl" in msg
    assert "some_dataset" in msg


# -- load_gcv_detailed_results: load + check in one call ---------------------------

def test_load_gcv_detailed_results_round_trips_and_checks(tmp_path, monkeypatch):
    import pickle

    dr = _make_detailed_results({1: 2, 2: 2, 3: 2})
    pkl_path = tmp_path / "fake_detailed_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(dr, f)

    _patch_canonical_count(monkeypatch, "fake_dataset", 6)
    loaded = load_gcv_detailed_results(pkl_path, "fake_dataset")
    assert loaded == dr


def test_load_gcv_detailed_results_raises_on_stale_pkl(tmp_path, monkeypatch):
    import pickle

    dr = _make_detailed_results({1: 2, 2: 2, 3: 2})  # 6 rows
    pkl_path = tmp_path / "stale_detailed_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(dr, f)

    _patch_canonical_count(monkeypatch, "sahni_fragoza_mapped090826", 6219)
    with pytest.raises(StaleCacheError):
        load_gcv_detailed_results(pkl_path, "sahni_fragoza_mapped090826")


# -- load_positional_cache -----------------------------------------------------------

def test_load_positional_cache_missing_returns_none_by_default(tmp_path):
    assert load_positional_cache(tmp_path / "absent.npy", n_expected=10) is None


def test_load_positional_cache_missing_returns_fill_value(tmp_path):
    arr = load_positional_cache(tmp_path / "absent.npy", n_expected=5, default_fill=0.5)
    assert arr.shape == (5,)
    assert np.all(arr == 0.5)


def test_load_positional_cache_correct_length_loads(tmp_path):
    path = tmp_path / "cache.npy"
    np.save(path, np.arange(10))
    arr = load_positional_cache(path, n_expected=10)
    assert np.array_equal(arr, np.arange(10))


def test_load_positional_cache_wrong_length_raises(tmp_path):
    path = tmp_path / "stale_cache.npy"
    np.save(path, np.arange(5894))
    with pytest.raises(StaleCacheError, match="5894"):
        load_positional_cache(path, n_expected=6219)
