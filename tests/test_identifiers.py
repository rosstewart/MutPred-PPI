"""Tests for utils/identifiers.py -- the two composite identifiers.

`__` joins protein pairs, a space joins protein+variant. Both round-trip
exactly because neither delimiter can occur inside the things they join.
"""
import pytest

from utils.identifiers import (
    bare_accession,
    is_pair_id,
    is_variant_id,
    pair_id,
    split_legacy_pair,
    split_pair_id,
    split_variant_id,
    variant_id,
)


# -- pair_id / split_pair_id ----------------------------------------------------

def test_pair_id_basic():
    assert pair_id("O14787-2", "Q13207") == "O14787-2__Q13207"


def test_pair_id_round_trip():
    assert split_pair_id(pair_id("A", "B")) == ("A", "B")


def test_pair_id_round_trip_isoforms():
    assert split_pair_id(pair_id("O14787-2", "Q13207-1")) == ("O14787-2", "Q13207-1")


def test_pair_id_preserves_order():
    # Not a sorted key -- pair_id(a, b) != pair_id(b, a) in general.
    assert pair_id("A", "B") != pair_id("B", "A")


def test_pair_id_rejects_separator_in_input():
    with pytest.raises(ValueError):
        pair_id("A__weird", "B")


def test_split_pair_id_rejects_non_pair():
    with pytest.raises(ValueError):
        split_pair_id("no_separator_here")


def test_split_pair_id_rejects_multiple_separators():
    with pytest.raises(ValueError):
        split_pair_id("A__B__C")


def test_is_pair_id():
    assert is_pair_id("A__B")
    assert not is_pair_id("A_B")
    assert not is_pair_id("A__B__C")


# -- variant_id / split_variant_id ----------------------------------------------

def test_variant_id_basic():
    assert variant_id("P25054", "S305R") == "P25054 S305R"


def test_variant_id_round_trip():
    assert split_variant_id(variant_id("P25054", "S305R")) == ("P25054", "S305R")


def test_variant_id_rejects_whitespace_in_input():
    with pytest.raises(ValueError):
        variant_id("P25054 extra", "S305R")
    with pytest.raises(ValueError):
        variant_id("P25054", "S305R\t")


def test_split_variant_id_rejects_non_pair():
    with pytest.raises(ValueError):
        split_variant_id("just_one_token")
    with pytest.raises(ValueError):
        split_variant_id("three tokens here")


def test_is_variant_id():
    assert is_variant_id("P25054 S305R")
    assert not is_variant_id("P25054")
    assert not is_variant_id("a b c")


# -- delimiters never collide ----------------------------------------------------

def test_pair_and_variant_delimiters_are_distinct():
    # A pair_id contains no whitespace, so it is never mistaken for a variant_id.
    p = pair_id("O14787-2", "Q13207")
    assert not is_variant_id(p)
    # A variant_id contains no "__", so it is never mistaken for a pair_id.
    v = variant_id("P25054", "S305R")
    assert not is_pair_id(v)


# -- bare_accession ---------------------------------------------------------------

def test_bare_accession_strips_isoform_suffix():
    assert bare_accession("O14787-2") == "O14787"


def test_bare_accession_no_op_on_bare_input():
    assert bare_accession("O14787") == "O14787"


# -- split_legacy_pair (heuristic, pre-090826 artifacts only) --------------------

def test_split_legacy_pair_uniprot_hyphen():
    assert split_legacy_pair("P35609-P29373") == ("P35609", "P29373")


def test_split_legacy_pair_refseq_prefix():
    acc, gene = split_legacy_pair("NP_005190_KRTAP10-7")
    assert acc == "NP_005190"
    assert gene == "KRTAP10-7"


def test_split_legacy_pair_isoform_in_first_id():
    # "Q8WWY3-1-Q9P286": three hyphen-delimited parts, isoform suffix is numeric.
    assert split_legacy_pair("Q8WWY3-1-Q9P286") == ("Q8WWY3-1", "Q9P286")


def test_split_legacy_pair_underscore_delimited():
    assert split_legacy_pair("A_B") == ("A", "B")


# -- exhaustive round-trip over real canonical data (plan verification §8) ------

def test_pair_id_round_trips_every_canonical_accession(fixtures_dir):
    """Every (interactor, partner) in the canonical tables round-trips through
    pair_id/split_pair_id exactly, isoforms included -- the guarantee the whole
    __ delimiter choice exists to make.
    """
    import glob

    import pandas as pd

    from paths import DATASETS_DIR

    csvs = glob.glob(str(DATASETS_DIR / "cv_reference" / "*_rows.csv.gz"))
    if not csvs:
        pytest.skip("no canonical row tables available in this environment")

    checked = 0
    for path in csvs:
        df = pd.read_csv(path, usecols=lambda c: c in ("interactor", "partner", "mutation"))
        for interactor, partner in zip(df["interactor"].astype(str), df["partner"].astype(str)):
            assert split_pair_id(pair_id(interactor, partner)) == (interactor, partner)
            checked += 1
    assert checked > 0

