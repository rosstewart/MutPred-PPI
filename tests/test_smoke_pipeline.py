"""End-to-end pipeline smoke test: cif -> graph -> store -> load_data().

Not a full GCV fold -- that needs a trained model, which is out of scope for
a fast, no-GPU test. This exercises the whole DATA path instead: parsing a
real AF3 structure, building its contact graph, storing/retrieving it
content-addressed, and joining a canonical row table onto its own sequences
exactly the way `utils.gcv_common.load_data` does for every GCV runner.

Fixtures are two real AF3 model outputs (~65 KB gzipped each), taken from the
archived training-quickstart example (archive/dead_scripts_20260909/).
"""
from pathlib import Path

import pandas as pd

from contact_graphs import ContactGraphStore, contact_graph_from_structure
from utils import gcv_common

FIXTURES = Path(__file__).parent / "fixtures" / "structures"


def test_smoke_cif_to_graph_to_store(tmp_path):
    fixture = FIXTURES / "fold_q8tcd5_v9hwf3_model_0.cif.gz"
    result = contact_graph_from_structure(fixture)
    assert result is not None
    seq_a, seq_b, edge_index = result
    assert len(seq_a) > 0 and len(seq_b) > 0
    assert edge_index.shape[0] == 2

    store_path = tmp_path / "contact_graphs.h5"
    store = ContactGraphStore(store_path, "w")
    store.put(seq_a, seq_b, edge_index, source=str(fixture))
    store.close()

    store = ContactGraphStore(store_path, "r")
    hit = store.get(seq_a, seq_b)
    assert hit is not None
    ei, n, is_first = hit
    assert n == len(seq_a) + len(seq_b)
    assert is_first

    edge_index_sym = store.load_edge_index(interactor=seq_a, partner=seq_b)
    assert edge_index_sym is not None
    assert edge_index_sym.max() < n
    store.close()


def test_smoke_second_fixture_produces_a_different_graph(tmp_path):
    """Sanity check that the two committed fixtures are genuinely different
    structures, not accidental duplicates."""
    r1 = contact_graph_from_structure(FIXTURES / "fold_q8tcd5_v9hwf3_model_0.cif.gz")
    r2 = contact_graph_from_structure(FIXTURES / "fold_q96s44_q9y3c4_model_0.cif.gz")
    assert r1 is not None and r2 is not None
    assert (r1[0], r1[1]) != (r2[0], r2[1])


def test_smoke_graph_to_canonical_row_table_to_load_data(tmp_path, monkeypatch):
    """The join every GCV runner performs: canonical rows (interactor, partner,
    mutation, perturbed) joined to sequences by accession, producing exactly
    the columns `PREDICTOR_COLS` names."""
    fixture = FIXTURES / "fold_q8tcd5_v9hwf3_model_0.cif.gz"
    seq_a, seq_b, _ = contact_graph_from_structure(fixture)

    mapped_dir = tmp_path / "mapped090826"
    mapped_dir.mkdir()

    rows = pd.DataFrame({
        "row_index": [0, 1],
        "interactor": ["FIXTURE_A", "FIXTURE_A"],
        "partner": ["FIXTURE_B", "FIXTURE_B"],
        "mutation": ["M1V", "K2A"],
        "perturbed": [0, 1],
    }).set_index("row_index")
    rows.to_csv(mapped_dir / "smoke_test_mapped090826_rows.csv.gz", compression="gzip")

    seqs = pd.DataFrame({"accession": ["FIXTURE_A", "FIXTURE_B"], "sequence": [seq_a, seq_b]})
    seqs.to_csv(mapped_dir / "sequences.csv.gz", index=False, compression="gzip")

    monkeypatch.setattr(gcv_common, "TABLES", mapped_dir)
    monkeypatch.setattr(gcv_common, "_SEQ_CACHE", None)

    cfg = gcv_common.DatasetConfig(
        "smoke_test_mapped090826", "smoke_test_mapped090826_rows.csv.gz", "unused_splits.csv.gz")
    df = gcv_common.load_data(cfg)

    assert len(df) == 2
    assert df.loc[0, "interactor_sequence"] == seq_a
    assert df.loc[1, "partner_sequence"] == seq_b
    assert df["perturbed"].tolist() == [0, 1]
    for col in gcv_common.PREDICTOR_COLS:
        assert col in df.columns


def test_smoke_load_data_raises_on_missing_sequence(tmp_path, monkeypatch):
    """load_data() must error, not silently drop, a row whose accession has no
    sequence -- this is the boundary check the module docstring promises."""
    mapped_dir = tmp_path / "mapped090826"
    mapped_dir.mkdir()

    rows = pd.DataFrame({
        "row_index": [0],
        "interactor": ["UNKNOWN_ACCESSION"],
        "partner": ["ALSO_UNKNOWN"],
        "mutation": ["M1V"],
        "perturbed": [0],
    }).set_index("row_index")
    rows.to_csv(mapped_dir / "smoke_test_mapped090826_rows.csv.gz", compression="gzip")

    seqs = pd.DataFrame({"accession": [], "sequence": []})
    seqs.to_csv(mapped_dir / "sequences.csv.gz", index=False, compression="gzip")

    monkeypatch.setattr(gcv_common, "TABLES", mapped_dir)
    monkeypatch.setattr(gcv_common, "_SEQ_CACHE", None)

    cfg = gcv_common.DatasetConfig(
        "smoke_test_mapped090826", "smoke_test_mapped090826_rows.csv.gz", "unused_splits.csv.gz")
    try:
        gcv_common.load_data(cfg)
        assert False, "expected KeyError for missing sequences"
    except KeyError:
        pass
