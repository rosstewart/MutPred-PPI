"""Tests for utils/mutations.py -- parse, base conversion, and apply.

Covers the three policy decisions the module's own docstring calls out:
alphabet (accepts lowercase, normalises), out-of-range positions (raise, never
silently index from the end), and apply()'s None-on-mismatch contract (never
a silent no-op that looks like a wild-type sequence).
"""
import pytest

from utils.mutations import (
    MUTATION_RE,
    apply,
    index,
    is_mutation,
    parse,
    position,
    to_one_based,
    to_zero_based,
)


# -- is_mutation -------------------------------------------------------------------

def test_is_mutation_valid():
    assert is_mutation("A123V")


def test_is_mutation_invalid():
    assert not is_mutation("not a mutation")
    assert not is_mutation("123")
    assert not is_mutation("")


# -- parse ---------------------------------------------------------------------------

def test_parse_basic():
    assert parse("A123V") == ("A", 123, "V")


def test_parse_normalises_lowercase():
    # Policy: accept lowercase, normalise to upper -- rejecting a real row for
    # a cosmetic reason is worse than normalising it.
    assert parse("a123v") == ("A", 123, "V")


def test_parse_rejects_unparseable():
    with pytest.raises(ValueError):
        parse("not a mutation")


def test_parse_rejects_non_positive_position():
    # Policy: pos < 1 in a 1-based string is a bug upstream, not a position to
    # silently reinterpret (the old bug: pos - 1 = -1 indexed from the end).
    with pytest.raises(ValueError):
        parse("A0V")


# -- index / position ------------------------------------------------------------------

def test_index_is_position_minus_one():
    assert index("A123V") == 122


def test_index_rejects_zero_position():
    with pytest.raises(ValueError):
        index("A0V")


def test_position_does_not_convert():
    # position() assumes no base -- it must NOT subtract 1, unlike index().
    assert position("A123V") == 123
    assert position("A0V") == 0  # legal for position(); parse()/index() reject it


# -- to_zero_based / to_one_based -----------------------------------------------------

def test_to_zero_based():
    assert to_zero_based("A123V") == "A122V"


def test_to_one_based():
    assert to_one_based("A122V") == "A123V"


def test_to_one_based_accepts_position_zero():
    # Legal in a 0-based string; parse() would reject it, so to_one_based must
    # not route through parse().
    assert to_one_based("A0V") == "A1V"


def test_to_one_based_to_zero_based_round_trip():
    assert to_zero_based(to_one_based("A0V")) == "A0V"


def test_to_one_based_rejects_unparseable():
    with pytest.raises(ValueError):
        to_one_based("garbage")


def test_to_zero_based_rejects_unparseable():
    with pytest.raises(ValueError):
        to_zero_based("garbage")


# -- apply ------------------------------------------------------------------------------

def test_apply_basic():
    assert apply("MATVAL", "A2T") == "MTTVAL"


def test_apply_returns_none_on_wt_mismatch():
    # sequence[1] is 'A', not 'V' -- the mutation does not match.
    assert apply("MATVAL", "V2T") is None


def test_apply_returns_none_when_position_out_of_range():
    assert apply("MAT", "A10V") is None


def test_apply_never_silently_returns_unchanged_sequence():
    # The bug this contract exists to prevent: a caller must be able to tell
    # "did not apply" apart from "applied and happened to be a no-op" -- the
    # latter is impossible here since wt != mt is not enforced, but a mismatch
    # must never come back as the original sequence.
    original = "MATVAL"
    result = apply(original, "Q2T")  # wrong wt at position 2
    assert result is None
    assert result != original


# -- exhaustive round-trip over real canonical mutations (plan verification) -----------

def test_base_conversion_round_trips_every_canonical_mutation():
    import glob

    import pandas as pd

    from paths import DATASETS_DIR

    csvs = glob.glob(str(DATASETS_DIR / "cv_reference" / "*_rows.csv.gz"))
    if not csvs:
        pytest.skip("no canonical row tables available in this environment")

    checked = 0
    for path in csvs:
        df = pd.read_csv(path, usecols=lambda c: c == "mutation")
        for m in df["mutation"].astype(str):
            if not is_mutation(m):
                continue
            assert to_one_based(to_zero_based(m)) == m
            assert index(m) == position(m) - 1
            checked += 1
    assert checked > 0
