"""Tests for utils/legacy_guard.py -- the pre-090826 input rejection gate.

Every migrated comparator runner (SAAMBE-3D, MutPPI, the VarChAMP blind test,
MutPred2 import) calls `reject_legacy` on its resolved inputs. These tests
pin the specific retired artifacts that motivated the guard so a future
change cannot silently start accepting them again.
"""
import time

import pytest

from paths import REPO_ROOT
from utils.legacy_guard import (
    LegacyInputError, reject_legacy, reject_legacy_dataset_name)


@pytest.fixture
def in_repo_tmp_path(tmp_path_factory):
    """A tmp dir inside REPO_ROOT, since reject_legacy treats out-of-tree
    paths as suspicious (that check is itself under test elsewhere)."""
    d = REPO_ROOT / "tests" / "_scratch"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def test_canonical_table_passes(in_repo_tmp_path):
    p = in_repo_tmp_path / "sahni_fragoza_mapped090826_rows.csv.gz"
    p.write_bytes(b"")
    reject_legacy(p)  # must not raise


@pytest.mark.parametrize("name", [
    "sfvcfp_rows.csv.gz",
    "training_data_internal.csv",
    "training_data.csv",
    "sahni_fragoza_all_vt_ids_and_labels.txt",
    "sahni_fragoza_all_vt_ids_0.pkl",
    "fold_splits.pkl",
    "some_export.mat",
    "some_export.pos",
    "some_export.neg",
    "some_export.labels",
    "some_export.vt_ids",
])
def test_retired_filenames_raise(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"")
    with pytest.raises(LegacyInputError):
        reject_legacy(p)


def test_path_outside_repo_and_data_root_raises():
    with pytest.raises(LegacyInputError):
        reject_legacy("/home/rcstewart/gnn/ppi_interaction_loss/cv_splits/x.pkl")
    with pytest.raises(LegacyInputError):
        reject_legacy("/home/rcstewart/ppi_lossgain/2026/mutppi/benchmark/training_data.csv")


def test_stale_mtime_raises(tmp_path):
    p = tmp_path / "some_canonical_looking_rows.csv.gz"
    p.write_bytes(b"")
    old = time.mktime(time.strptime("2026-01-01", "%Y-%m-%d"))
    import os
    os.utime(p, (old, old))
    with pytest.raises(LegacyInputError):
        reject_legacy(p)


def test_fresh_mtime_passes(in_repo_tmp_path):
    p = in_repo_tmp_path / "some_canonical_looking_rows.csv.gz"
    p.write_bytes(b"")
    reject_legacy(p)  # just-written, must not raise


@pytest.mark.parametrize("name", [
    "sahni_fragoza_varchamp_full_pooled",
    "sahni_fragoza_varchamp1p_cava",
    "sahni_fragoza_varchamp2026",
    "sahni_varchamp1p_cava",
    "sahni_fragoza_varchamp_pooled",
])
def test_retired_dataset_tokens_raise(name):
    with pytest.raises(LegacyInputError):
        reject_legacy_dataset_name(name)


@pytest.mark.parametrize("name", [
    "sahni_only_mapped090826",
    "fragoza_only_mapped090826",
    "sahni_fragoza_mapped090826",
    "varchamp_all_mapped090826",
    "sahni_fragoza_varchamp_all_mapped090826",
])
def test_canonical_dataset_names_pass(name):
    reject_legacy_dataset_name(name)  # must not raise
