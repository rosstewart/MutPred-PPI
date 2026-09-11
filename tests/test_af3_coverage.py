"""Rows with no AlphaFold3 structure must never reach the GCV layer.

A complex absent from the canonical structure manifest has no contact graph, so
every structure-based method scores it NaN. `annotate_af3_coverage.py` marks
those rows `af3_failed = True` in the mapping CSVs (where they are kept for
provenance) and `prepare_gcv_tables.py` drops them **before** assigning
`row_index` -- which is what keeps `row_index` a contiguous 0..n-1 positional
key for the splits table and every downstream cache.
"""
import csv
from pathlib import Path

import pandas as pd
import pytest

from data_processing import annotate_af3_coverage as ac

MANIFEST = Path("datasets/af3_structures_canonical/manifest.csv")
MAPPING = Path("datasets/source_mapping/datasets")
TRAINING_EVAL = Path("datasets/training_eval")

needs_data = pytest.mark.skipif(not (MANIFEST.exists() and MAPPING.is_dir()),
                                reason="structure manifest / mapping CSVs absent")


def test_pairs_are_compared_unordered(tmp_path):
    """(A,B) and (B,A) are one complex; orientation is resolved at load time."""
    man = tmp_path / "manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain_a_accession", "chain_b_accession"])
        w.writerow(["P00001", "Q00002"])
    folded = ac.folded_pairs(man)
    assert ("P00001", "Q00002") in folded
    assert tuple(sorted(("Q00002", "P00001"))) in folded


def test_annotation_is_additive(tmp_path):
    """It adds one column and changes nothing else."""
    man = tmp_path / "manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain_a_accession", "chain_b_accession"])
        w.writerow(["P1", "Q1"])
    root = tmp_path / "datasets"
    root.mkdir()
    before = pd.DataFrame({"interactor": ["P1", "P2"], "partner": ["Q1", "Q2"],
                           "mutation": ["A1V", "C2D"]})
    before.to_csv(root / "d.csv", index=False)

    ac.annotate(dry_run=False, manifest=man, root=root)
    after = pd.read_csv(root / "d.csv")

    assert list(after["af3_failed"]) == [False, True]
    assert after.drop(columns=["af3_failed"]).equals(before)


def test_dry_run_does_not_write(tmp_path):
    man = tmp_path / "manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain_a_accession", "chain_b_accession"])
    root = tmp_path / "datasets"
    root.mkdir()
    pd.DataFrame({"interactor": ["P1"], "partner": ["Q1"]}).to_csv(root / "d.csv", index=False)
    ac.annotate(dry_run=True, manifest=man, root=root)
    assert "af3_failed" not in pd.read_csv(root / "d.csv").columns


@needs_data
def test_mapping_csvs_carry_the_flag():
    for path in ac.mapping_csvs(MAPPING):
        cols = pd.read_csv(path, nrows=0).columns
        if {"interactor", "partner"} <= set(cols):
            assert "af3_failed" in cols, f"{path} not annotated"


@pytest.mark.skipif(not TRAINING_EVAL.is_dir(), reason="training_eval absent")
def test_gcv_tables_contain_no_unfolded_pair():
    """The load-bearing assertion: nothing in the GCV layer lacks a structure."""
    if not MANIFEST.exists():
        pytest.skip("manifest absent")
    folded = ac.folded_pairs(MANIFEST)
    from utils.gcv_common import DATASET_CONFIGS, load_data
    for name, cfg in DATASET_CONFIGS.items():
        if not (TRAINING_EVAL / cfg.rows_file).exists():
            pytest.skip(f"{cfg.rows_file} not built")
        df = load_data(cfg)
        missing = [(a, b) for a, b in zip(df["interactor"], df["partner"])
                   if tuple(sorted((a, b))) not in folded]
        assert not missing, (f"{name}: {len(missing)} row(s) reference a complex "
                             f"with no AF3 structure, e.g. {missing[:3]}")


@pytest.mark.skipif(not TRAINING_EVAL.is_dir(), reason="training_eval absent")
def test_row_index_is_contiguous_after_filtering():
    from utils.gcv_common import DATASET_CONFIGS, load_data
    for cfg in DATASET_CONFIGS.values():
        if not (TRAINING_EVAL / cfg.rows_file).exists():
            pytest.skip(f"{cfg.rows_file} not built")
        df = load_data(cfg)
        assert list(df.index) == list(range(len(df))), \
            f"{cfg.name}: row_index is not 0..n-1 after dropping af3_failed rows"


# ── input completeness ────────────────────────────────────────────────────────
#
# With structure-less rows dropped at dataset-build time, every remaining row
# should have a contact graph AND all three ProtT5 embeddings (wild-type,
# mutant, partner). `mutpred_ppi_gcv.py` relies on exactly this: it builds
# tensors with `require_complete=True`, which RAISES rather than scoring an
# incomplete row NaN. So a gap here is not a silent quality loss, it is a
# crash in the GCV.

@pytest.mark.skipif(not TRAINING_EVAL.is_dir(), reason="training_eval absent")
def test_every_row_has_all_three_prott5_embeddings():
    import pickle
    from utils.gcv_common import DATASET_CONFIGS, load_data
    for name, cfg in DATASET_CONFIGS.items():
        cache_path = TRAINING_EVAL / f"{name}_prott5.pkl"
        if not (TRAINING_EVAL / cfg.rows_file).exists() or not cache_path.exists():
            pytest.skip(f"{name} tables or ProtT5 cache not built")
        df = load_data(cfg)
        with open(cache_path, "rb") as fh:
            cache = pickle.load(fh)
        keys = set(cache)
        missing_wt = ((set(df["interactor"]) | set(df["partner"])) - keys)
        missing_mut = {f"{a}_{m}" for a, m in zip(df["interactor"], df["mutation"])} - keys
        assert not missing_wt, f"{name}: {len(missing_wt)} wild-type/partner embeddings missing"
        assert not missing_mut, (
            f"{name}: {len(missing_mut)} MUTANT embeddings missing, e.g. "
            f"{sorted(missing_mut)[:3]} -- top up with "
            f"src/data_processing/precompute_prott5_datasets.py --dataset {name}")
