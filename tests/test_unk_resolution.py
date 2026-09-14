"""Structures with residues AlphaFold3 could not model must still resolve.

AF3 writes a residue it cannot build as `UNK`, which maps to "X". Selenoproteins
are the common case: TXNRD1 (Q16881), TXNRD2 (Q9NNW7) and GPX1 (P07203) each
carry a selenocysteine that AF3 emits as UNK, so the structure reads `...AGCXG`
where the canonical sequence reads `...AGCUG`. Lengths agree; one character does
not.

An exact dict lookup drops the entire complex on that one character. Five
structures were lost this way when the canonical set was rebuilt, four of them
load-bearing for neurodev rows -- and silently, as an "unresolved sequence"
count rather than an error.

The rule: exact match first; on a miss, "X" is a wildcard against canonical
sequences of the same length, and the match must be UNIQUE.
"""
import pytest

from data_processing.canonicalize_structures import _resolve_with_unk


def _index(m):
    by_len = {}
    for s, a in m.items():
        by_len.setdefault(len(s), []).append((s, a))
    return by_len


MAP = {"ABCUG": "P1", "ABCDE": "P2", "AAAAA": "P3", "ABCUGZZ": "P4"}
BY_LEN = _index(MAP)


class TestResolution:
    def test_exact_match_wins(self):
        assert _resolve_with_unk("ABCUG", MAP, BY_LEN) == "P1"

    def test_unk_position_matches_the_canonical_residue(self):
        """The selenocysteine case: X stands in for U."""
        assert _resolve_with_unk("ABCXG", MAP, BY_LEN) == "P1"

    def test_length_must_still_agree(self):
        assert _resolve_with_unk("ABCXGZ", MAP, BY_LEN) is None

    def test_non_x_mismatches_are_not_tolerated(self):
        """Only the unmodelled position is wild; everything else is exact."""
        assert _resolve_with_unk("ABCUQ", MAP, BY_LEN) is None

    def test_unknown_sequence_returns_none(self):
        assert _resolve_with_unk("ZZZZZ", MAP, BY_LEN) is None

    def test_a_sequence_without_x_takes_no_fallback(self):
        """No X means nothing to be wild about -- must not fuzzy-match."""
        assert _resolve_with_unk("ABCDZ", MAP, BY_LEN) is None


class TestAmbiguityIsRefused:
    def test_two_candidates_differing_only_at_x_are_refused(self):
        """Guessing here would silently mis-assign a whole complex."""
        m = {"ABCUG": "P1", "ABCDG": "P2"}
        assert _resolve_with_unk("ABCXG", m, _index(m)) is None

    def test_one_candidate_among_several_same_length_is_accepted(self):
        m = {"ABCUG": "P1", "QQQQQ": "P2", "ZZZZZ": "P3"}
        assert _resolve_with_unk("ABCXG", m, _index(m)) == "P1"

    def test_all_x_is_refused_when_several_share_the_length(self):
        m = {"ABCUG": "P1", "QQQQQ": "P2"}
        assert _resolve_with_unk("XXXXX", m, _index(m)) is None


class TestRealSelenoproteins:
    """The three accessions that actually triggered this."""

    @pytest.mark.parametrize("acc,canon,struct", [
        ("Q9NNW7", "MAAMTGCUG", "MAAMTGCXG"),
        ("Q16881", "MGCAAGCUG", "MGCAAGCXG"),
        ("P07203", "MCAAGPSCU", "MCAAGPSCX"),
    ])
    def test_sec_written_as_unk_still_resolves(self, acc, canon, struct):
        m = {canon: acc}
        assert _resolve_with_unk(struct, m, _index(m)) == acc
