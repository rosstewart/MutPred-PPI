"""Tests for utils/blind_test_common.py -- the one shared blind-test array
loader, replacing what varchamp_blind_test.py and restratify_skempi_methods.py
each hand-rolled with no internal-consistency check at all.
"""
import numpy as np
import pytest

from utils.blind_test_common import load_class_arrays
from utils.gcv_common import StaleCacheError


def _write_triple(tmp_path, method, cls, n_preds, n_labels, n_vt_ids=None):
    np.save(tmp_path / f"{method}_c{cls}_preds.npy", np.arange(n_preds, dtype=float))
    np.save(tmp_path / f"{method}_c{cls}_labels.npy", np.arange(n_labels))
    if n_vt_ids is not None:
        np.save(tmp_path / f"{method}_c{cls}_vt_ids.npy",
               np.array([f"P{i} A1V" for i in range(n_vt_ids)], dtype=object))


def test_returns_none_when_missing(tmp_path):
    assert load_class_arrays("SomeMethod", 1, tmp_path) is None


def test_loads_consistent_triple(tmp_path):
    _write_triple(tmp_path, "M", 1, 5, 5, 5)
    result = load_class_arrays("M", 1, tmp_path)
    assert result is not None
    preds, labels, vt_ids = result
    assert len(preds) == len(labels) == len(vt_ids) == 5


def test_raises_when_preds_and_labels_disagree(tmp_path):
    _write_triple(tmp_path, "M", 1, 5, 4, 5)
    with pytest.raises(StaleCacheError):
        load_class_arrays("M", 1, tmp_path)


def test_raises_when_vt_ids_disagrees(tmp_path):
    _write_triple(tmp_path, "M", 1, 5, 5, 3)
    with pytest.raises(StaleCacheError):
        load_class_arrays("M", 1, tmp_path)


def test_require_vt_ids_false_allows_missing_vt_ids(tmp_path):
    _write_triple(tmp_path, "M", 1, 5, 5)  # no vt_ids file
    result = load_class_arrays("M", 1, tmp_path, require_vt_ids=False)
    assert result is not None
    preds, labels, vt_ids = result
    assert vt_ids is None


def test_require_vt_ids_true_returns_none_if_vt_ids_missing(tmp_path):
    _write_triple(tmp_path, "M", 1, 5, 5)  # no vt_ids file
    assert load_class_arrays("M", 1, tmp_path, require_vt_ids=True) is None


def test_partial_files_returns_none(tmp_path):
    # Only preds written -- labels missing entirely.
    np.save(tmp_path / "M_c1_preds.npy", np.arange(5, dtype=float))
    assert load_class_arrays("M", 1, tmp_path) is None
