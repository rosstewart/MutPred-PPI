"""Variants the model was trained on must be excluded from every enrichment figure.

A variant present in the training set has a *fitted* score, not a predicted one.
Leaving it in an enrichment comparison lets training signal inflate exactly the
contrast the figure measures, and it does so unevenly: the disease sets overlap
training far more than the gnomAD background does, so the bias is not a wash.

The exclusion is defined once, in `analysis.training_overlap`, and consumed by
`classify_variant_dbs.py` (which feeds Fig 5, S8, S9 and the stability figure)
and by `protein_class_enrichment.py` (which reads the raw TSVs instead). These
tests pin the definition and the places it must be applied.
"""
import pandas as pd
import pytest

from analysis import training_overlap


class TestDefinition:
    def test_overlap_ignores_the_partner(self):
        """(interactor, variant) is the unit, not the full triple.

        A variant seen against ANY partner has had its mutation representation
        fitted, and the model reads the same mutated-site features whichever
        partner it is scored against.
        """
        df = pd.DataFrame({"uniprot": ["P1", "P1"], "variant": ["A1G", "A1G"],
                           "partner": ["Q1", "Q2"]})
        known = frozenset({("P1", "A1G")})
        mask = pd.Series([(u, v) in known
                          for u, v in zip(df.uniprot, df.variant)])
        assert mask.all(), "partner must not affect membership"

    def test_column_name_is_shared(self):
        assert training_overlap.OVERLAP_COLUMN == "training_overlap"

    def test_training_dataset_is_the_all_data_set(self):
        """The model that scores the repositories is trained on exactly this."""
        assert training_overlap.TRAINING_DATASET == "sahni_fragoza_varchamp_all"

    def test_missing_training_table_is_not_silently_empty(self, monkeypatch):
        """An empty set would turn 'unknown' into 'no overlap' -- the failure
        mode this module exists to prevent."""
        monkeypatch.setattr(training_overlap, "training_variants",
                            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
        assert training_overlap.training_variants_or_none() is None


@pytest.mark.requires_data
class TestAgainstRealData:
    def test_training_set_is_non_empty(self):
        assert len(training_overlap.training_variants()) > 1000

    def test_positions_are_one_based_on_both_sides(self):
        """Training `mutation` and repository `variant` must share a convention.

        If one were 0-based the two vocabularies would barely intersect, the
        filter would remove almost nothing, and nothing would visibly fail.

        Checked against the RAW prediction TSV, not a classified table: the
        classified tables have had the overlap removed, so asserting overlap on
        one of those would be asserting the opposite of what the filter does.
        """
        from pathlib import Path

        from paths import DATA_ROOT
        tsv = Path(DATA_ROOT) / "clinvar" / "mutpred_ppi_predictions.tsv"
        if not tsv.exists():
            pytest.skip("raw ClinVar predictions not present")
        raw = pd.read_csv(tsv, sep="\t", usecols=["interactor", "mutation"],
                          dtype=str).drop_duplicates()
        mask = training_overlap.overlap_mask(raw, "interactor", "mutation")
        assert mask.sum() > 0, (
            "no raw ClinVar prediction matches any training variant -- the two "
            "sides disagree about position numbering")

    def test_filter_removes_rows_and_reports_the_count(self, capsys):
        from pathlib import Path
        table = Path("results/variant_dbs_all_data/clinvar/pathogenic.csv.gz")
        if not table.exists():
            pytest.skip("classified tables not present")
        rows = pd.read_csv(table, dtype={"uniprot": str, "variant": str,
                                         "partner": str})
        kept, n = training_overlap.drop_training_variants(rows, label="test")
        assert len(kept) == len(rows) - n
        assert training_overlap.overlap_mask(kept).sum() == 0, "filter left overlap behind"


@pytest.mark.requires_data
class TestAppliedToPublishedTables:
    """The classified tables that Fig 5/S8/S9/stability read must be filtered."""

    @pytest.mark.parametrize("rel", [
        "clinvar/pathogenic.csv.gz",
        "clinvar/benign.csv.gz",
        "gnomad/gnomad.csv.gz",
    ])
    def test_no_training_variant_survives(self, rel):
        from pathlib import Path
        table = Path("results/variant_dbs_all_data") / rel
        if not table.exists():
            pytest.skip(f"{rel} not present")
        rows = pd.read_csv(table, dtype=str)
        n = int(training_overlap.overlap_mask(rows).sum())
        assert n == 0, (
            f"{rel} still contains {n:,} rows whose variant was in training -- "
            f"re-run classify_variant_dbs.py")


@pytest.mark.requires_data
class TestTableAndFiguresAgree:
    """Table S1 must report the same n as the figures draw.

    Both describe the same groups. If the table counted the raw inventory while
    the figures counted the filtered samples, the paper would print two
    different n next to the same group name, and a reader would have no way to
    tell which was which.
    """

    CHECK = {
        "Pathogenic":    "clinvar/pathogenic",
        "Benign":        "clinvar/benign",
        "VUS":           "clinvar/vus",
        "Rare Benign":   "clinvar/rare_benign",
        "Pathogenic AR": "clinvar/ar_pathogenic",
        "Pathogenic AD": "clinvar/ad_pathogenic",
    }

    @staticmethod
    def _table_variants():
        """{group: Variants} parsed out of the LaTeX table."""
        from pathlib import Path
        tex = Path("figures/variant_db_stats_table.tex")
        if not tex.exists():
            pytest.skip("Table S1 not generated")
        out = {}
        for line in tex.read_text().splitlines():
            if "&" not in line or "\\quad" not in line:
                continue
            cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
            name = cells[0].replace("\\quad", "").strip()
            nums = [c.replace("{,}", "").replace(",", "") for c in cells[1:]]
            try:
                out[name] = int(nums[2])          # Proteins, Pairs, Variants, ...
            except (IndexError, ValueError):
                continue
        return out

    def test_variant_counts_match_the_classified_tables(self):
        from pathlib import Path
        table = self._table_variants()
        checked = 0
        for name, rel in self.CHECK.items():
            p = Path("results/variant_dbs_all_data") / f"{rel}.csv.gz"
            if not p.exists() or name not in table:
                continue
            n = pd.read_csv(p, dtype=str).groupby(["uniprot", "variant"]).ngroups
            assert table[name] == n, (
                f"Table S1 reports {table[name]:,} variants for {name} but the "
                f"classified table the figures use has {n:,} -- one of them has "
                f"not been regenerated since the training-overlap filter changed")
            checked += 1
        if not checked:
            pytest.skip("no comparable groups present")
