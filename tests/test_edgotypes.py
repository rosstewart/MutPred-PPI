"""Tests for the edgotype definition and the variant-group storage contract.

`analysis.edgotypes` replaced a pair of parallel files (an edgotype array and a
per-variant score list) that had to be kept index-aligned by hand. These tests
pin the three properties that pairing failed to guarantee: the views stay
aligned by construction, the stored table carries no baked-in threshold, and the
manuscript's partner-coverage rule is conditional on what BioGRID actually
lists for a variant.
"""
import numpy as np
import pandas as pd
import pytest

from analysis import edgotypes
from analysis.classify_variant_dbs import build_arrays


def _group(rows):
    return edgotypes.EdgotypeGroup(
        name="t", table=pd.DataFrame(rows, columns=edgotypes.COLUMNS))


class TestClassify:
    def test_all_disrupted_is_quasi_null(self):
        assert edgotypes.classify([0.9, 0.8]) == "Quasi-null"

    def test_none_disrupted_is_quasi_wild_type(self):
        assert edgotypes.classify([0.1, 0.2]) == "Quasi-wild-type"

    def test_mixed_is_edgetic(self):
        assert edgotypes.classify([0.9, 0.1]) == "Edgetic"

    def test_threshold_is_strict(self):
        # exactly at the threshold counts as preserved
        assert edgotypes.classify([0.5]) == "Quasi-wild-type"

    def test_single_partner_can_never_be_edgetic(self):
        # Edgetic needs disagreement between partners, so coverage -- not the
        # variant -- decides whether the class is reachable at all.
        for s in (0.0, 0.4, 0.6, 1.0):
            assert edgotypes.classify([s]) in ("Quasi-null", "Quasi-wild-type")


class TestGroupViews:
    def test_views_are_aligned_and_ordered(self):
        g = _group([
            ("U2", "C2D", "P1", 0.9, 2), ("U2", "C2D", "P2", 0.1, 2),
            ("U1", "A1B", "P1", 0.9, 1),
        ])
        assert g.variants == [("U1", "A1B"), ("U2", "C2D")]
        assert len(g.classify()) == len(g.scores_by_variant()) == len(g) == 2
        assert list(g.classify()) == ["Quasi-null", "Edgetic"]
        assert list(g.n_partners()) == [1, 2]

    def test_threshold_is_applied_on_read_not_stored(self):
        g = _group([("U1", "A1B", "P1", 0.9, 1)])
        assert g.classify(0.5)[0] == "Quasi-null"
        assert g.classify(0.95)[0] == "Quasi-wild-type"

    def test_counts_cover_every_edgotype(self):
        g = _group([("U1", "A1B", "P1", 0.9, 1)])
        assert set(g.counts()) == set(edgotypes.EDGOTYPES)
        assert sum(g.counts().values()) == len(g)


class TestRoundTrip:
    def test_save_then_load_preserves_everything(self, tmp_path):
        table = pd.DataFrame(
            [("U1", "A1B", "P1", 0.9, 4), ("U1", "A1B", "P2", 0.1, 4)],
            columns=edgotypes.COLUMNS)
        edgotypes.save_group(tmp_path / "db", "grp", table)
        g = edgotypes.load_group(tmp_path, "db", "grp")
        assert list(g.classify()) == ["Edgetic"]
        assert sorted(g.table["n_biogrid_partners"].unique()) == [4]

    def test_missing_group_is_none_not_an_error(self, tmp_path):
        assert edgotypes.load_group(tmp_path, "db", "absent") is None

    def test_save_rejects_a_table_missing_columns(self, tmp_path):
        with pytest.raises(ValueError, match="missing"):
            edgotypes.save_group(tmp_path, "grp",
                                 pd.DataFrame({"uniprot": ["U1"]}))


class TestPartnerCoverageRule:
    """The manuscript analyses a variant only when enough of its partners were
    scored, and only when BioGRID lists that many for it to begin with."""

    GROUPED = {
        ("U1", "A1B"): {"P1": 0.9},
        ("U2", "C2D"): {"P1": 0.9, "P2": 0.1},
        ("U3", "E3F"): {"P1": 0.2, "P2": 0.9, "P3": 0.4, "P4": 0.1},
    }
    SUBSET = {("U1", "A1B", "P1"), ("U2", "C2D", "P1"), ("U3", "E3F", "P1")}
    # What BioGRID lists, which is not what was scored.
    COUNTS = {("U1", "A1B"): 1, ("U2", "C2D"): 5, ("U3", "E3F"): 5}

    def _variants(self, table):
        return set(map(tuple, table[["uniprot", "variant"]].drop_duplicates().values))

    def test_default_keeps_every_scored_variant(self):
        t = build_arrays(self.GROUPED, self.SUBSET, min_partners=1,
                         biogrid_counts=self.COUNTS)
        assert self._variants(t) == {("U1", "A1B"), ("U2", "C2D"), ("U3", "E3F")}

    def test_sparse_variant_is_processed_normally(self):
        # BioGRID lists one partner for U1, so it is kept even at a threshold of 3.
        t = build_arrays(self.GROUPED, self.SUBSET, min_partners=3,
                         biogrid_counts=self.COUNTS)
        assert ("U1", "A1B") in self._variants(t)

    def test_well_connected_but_poorly_scored_variant_is_dropped(self):
        # BioGRID lists five partners for U2 but only two were scored.
        t = build_arrays(self.GROUPED, self.SUBSET, min_partners=3,
                         biogrid_counts=self.COUNTS)
        assert ("U2", "C2D") not in self._variants(t)
        assert ("U3", "E3F") in self._variants(t)

    def test_biogrid_count_is_persisted_not_the_scored_count(self):
        t = build_arrays(self.GROUPED, self.SUBSET, min_partners=1,
                         biogrid_counts=self.COUNTS)
        u2 = t[t["uniprot"] == "U2"]
        assert len(u2) == 2                                  # two scored
        assert set(u2["n_biogrid_partners"]) == {5}          # five known

    def test_subset_selects_variants_not_partners(self):
        """A group file listing one partner must not discard the others.

        Group membership is a property of the variant; the partner universe is
        the BioGRID direct-binding interactome, enforced when the rows table was
        built. Filtering partners by the subset as well silently dropped scored
        direct-binding edges.
        """
        t = build_arrays(self.GROUPED, self.SUBSET, min_partners=1,
                         biogrid_counts=self.COUNTS)
        u3 = t[t["uniprot"] == "U3"]
        assert len(u3) == 4, "all scored partners kept, not just the subset's one"

    def test_variant_absent_from_the_group_is_excluded(self):
        t = build_arrays(self.GROUPED, {("U1", "A1B", "P1")}, min_partners=1,
                         biogrid_counts=self.COUNTS)
        assert self._variants(t) == {("U1", "A1B")}
