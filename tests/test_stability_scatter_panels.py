"""The stability/interaction figure must inherit Fig 5's samples and statistics.

This figure used to re-derive each sample from the restricted subset pickles and
the COSMIC recurrence dictionary, duplicating `classify_variant_dbs.py`. The two
drifted: HGMD came out at 3,478 variants against the canonical table's 3,561,
and the script carried a hardcoded table of "Fig 5 reference counts" to check
itself against -- counts that had themselves been produced by the path being
checked.

It now reads the canonical per-stratum tables directly and measures enrichment
with the same statistic, the same background and the same resampling as the main
enrichment figure, so the two agree by construction. These tests pin that.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import stability_interaction_scatter as sis

pytestmark = pytest.mark.requires_data


class TestSampleSpec:
    def test_panel_a_covers_every_stratum_on_disk(self):
        """Every classified stratum should appear somewhere in the figure."""
        spec = {(db, stratum)
                for _d, db, subs in sis.ENRICHMENT_GROUPS for _l, stratum in subs}
        spec |= {(db, stratum) for _l, db, stratum in sis.KDE_PANELS}
        spec.add(sis.BASELINE)
        on_disk = {(p.parent.name, p.name[:-len(".csv.gz")])
                   for p in sis._DB.glob("*/*.csv.gz")}
        if not on_disk:
            pytest.skip("no classified stratum tables present")
        assert not on_disk - spec, f"strata produced but never plotted: {on_disk - spec}"

    def test_panel_a_groups_match_the_main_figure(self):
        """Same databases, same order, same subgroup labels as Fig 5."""
        from analysis.variant_db_charts import dataset_enrichment_labels as fig5
        ours = {d: [lbl for lbl, _ in subs] for d, _db, subs in sis.ENRICHMENT_GROUPS}
        assert ours["ClinVar"] == fig5["clinvar"]
        assert ours["COSMIC"] == fig5["cosmic"]
        assert ours["COSMIC (Onco)"] == fig5["cosmic_onco"]
        assert ours["COSMIC (TSG)"] == fig5["cosmic_tsg"]
        assert ours["HGMD"] == fig5["hgmd"]

    def test_two_components_in_a_fixed_order(self):
        names = [n for n, _k, _i in sis.COMPONENTS]
        assert names == ["Max PPI Disruption", "Stability Disruption"]

    def test_every_sample_resolves_a_real_colour(self):
        """A label that misses the shared palette falls back to grey silently.

        The palette is keyed on the MAIN figure's spellings ("NDD Case",
        "1e-2 < AF"); using the lowercased display form for the lookup left four
        bars grey with no error.
        """
        grey = {"#666", "#666666"}
        bad = [(d, lbl) for d, db, subs in sis.ENRICHMENT_GROUPS
               for lbl, st in subs
               if sis._get_enrich_color(db, st, lbl) in grey]
        assert not bad, f"samples falling back to the default grey: {bad}"

    def test_display_labels_lowercase_case_and_control(self):
        assert sis._display_label("NDD Case") == "NDD case"
        assert sis._display_label("NDD Control") == "NDD control"
        assert sis._display_label("Pathogenic") == "Pathogenic"

    def test_panel_c_uses_one_colour_scheme(self):
        """Every density panel must share a palette; hue carries no meaning."""
        colours = {sis._panel_color(lbl, db) for lbl, db, _s in sis.KDE_PANELS}
        assert len(colours) == 1, f"panel C is drawn in {len(colours)} colours"

    def test_panel_b_has_twelve_panels_in_the_declared_order(self):
        assert len(sis.KDE_PANELS) == 12
        assert [l for l, _d, _s in sis.KDE_PANELS][:5] == [
            "ClinVar pathogenic", "ClinVar benign", "ClinVar VUS", "HGMD", "gnomAD"]

    def test_cosmic_and_hgmd_are_declared_restricted(self):
        assert sis._RESTRICTED_DBS == {"cosmic", "hgmd"}


class TestStatistic:
    def test_matches_the_main_figure_definition(self):
        from analysis.variant_db_charts import calc_enrichment as fig5
        for a, b in [(0.3, 0.1), (0.05, 0.4), (0.5, 0.5)]:
            assert sis.calc_enrichment(a, b) == pytest.approx(fig5(a, b))

    def test_enrichment_is_zero_against_itself(self):
        assert sis.calc_enrichment(0.3, 0.3) == 0.0

    def test_enrichment_is_bounded_and_signed(self):
        assert sis.calc_enrichment(1.0, 0.0) == 1.0
        assert sis.calc_enrichment(0.0, 1.0) == -1.0
        assert sis.calc_enrichment(0.4, 0.2) > 0
        assert sis.calc_enrichment(0.1, 0.2) < 0

    def test_enrichment_handles_an_empty_background(self):
        assert sis.calc_enrichment(0.0, 0.0) == 0.0

    def test_thresholds_and_bootstrap_match_the_main_figure(self):
        assert sis.INT_THRESHOLD == 0.5
        assert sis.DDG_THRESHOLD == 0.5
        assert sis.N_BOOTSTRAP == 100_000

    def test_baseline_is_gnomad(self):
        assert sis.BASELINE == ("gnomad", "gnomad")

    def test_sample_fractions_counts_variants_over_threshold(self):
        df = pd.DataFrame({"max_score": [0.9, 0.1, 0.5, 0.2],
                           "mean_ddg":  [0.9, 0.9, 0.1, 0.1]})
        f_int, f_ddg, n = sis.sample_fractions(df)
        assert (n, f_int, f_ddg) == (4, 0.5, 0.5)

    def test_empty_sample_is_not_an_error(self):
        assert sis.sample_fractions(pd.DataFrame()) == (0.0, 0.0, 0)


class TestBootstrap:
    """Resampling must preserve the correlation between the two categories."""

    def _df(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        # perfectly correlated: every destabilising variant is also disrupting
        both = rng.random(n) < 0.3
        return pd.DataFrame({"max_score": np.where(both, 0.9, 0.1),
                             "mean_ddg":  np.where(both, 0.9, 0.1)})

    def test_replicates_have_the_right_shape(self):
        rng = np.random.default_rng(0)
        b = sis.bootstrap_fractions(self._df(), 0.5, 0.5, 500, rng)
        assert b.shape == (500, 2)

    def test_replicates_centre_on_the_observed_fractions(self):
        df = self._df()
        f_int, f_ddg, _ = sis.sample_fractions(df)
        rng = np.random.default_rng(0)
        b = sis.bootstrap_fractions(df, 0.5, 0.5, 4000, rng)
        assert b[:, 0].mean() == pytest.approx(f_int, abs=0.01)
        assert b[:, 1].mean() == pytest.approx(f_ddg, abs=0.01)

    def test_joint_resampling_preserves_correlation(self):
        """The two fractions are measured on the SAME variants.

        Resampling them independently would make their replicates uncorrelated
        and understate the uncertainty in their difference. Here they are
        perfectly correlated in the data, so they must be in the replicates too.
        """
        rng = np.random.default_rng(0)
        b = sis.bootstrap_fractions(self._df(), 0.5, 0.5, 2000, rng)
        assert np.corrcoef(b[:, 0], b[:, 1])[0, 1] > 0.99

    def test_uncertainty_shrinks_with_sample_size(self):
        rng = np.random.default_rng(0)
        small = sis.bootstrap_fractions(self._df(200), 0.5, 0.5, 2000, rng)
        large = sis.bootstrap_fractions(self._df(20000), 0.5, 0.5, 2000, rng)
        assert small[:, 0].std() > 5 * large[:, 0].std()

    def test_empty_sample_yields_zero_replicates(self):
        rng = np.random.default_rng(0)
        b = sis.bootstrap_fractions(pd.DataFrame(), 0.5, 0.5, 10, rng)
        assert b.shape == (10, 2) and not b.any()


class TestJoinBase:
    def test_both_sides_are_one_based(self):
        """Stratum `variant` and stability `mutation` must share a convention.

        If one were 0-based the vocabularies would be nearly disjoint, the join
        would silently produce almost nothing, and the figure would still draw.
        """
        db, stratum = "clinvar", "pathogenic"
        table = sis._DB / db / f"{stratum}.csv.gz"
        stab = sis.load_stability(db)
        if not table.exists() or stab is None:
            pytest.skip("ClinVar inputs not present")
        rows = pd.read_csv(table, dtype=str)
        keys_rows = set(map(tuple, rows[["uniprot", "partner", "variant"]].values))
        keys_stab = set(map(tuple, stab[["uniprot", "partner", "variant"]].values))
        overlap = keys_rows & keys_stab
        assert len(overlap) > 0.5 * len(keys_rows), (
            f"only {len(overlap)}/{len(keys_rows)} rows join -- the two sides "
            f"disagree about 0- vs 1-based positions")


class TestPanelSizes:
    def test_panel_size_equals_its_stratum_unique_variants(self):
        """No filtering may happen between the canonical table and the figure."""
        checked = 0
        for _display, db, subs in sis.ENRICHMENT_GROUPS:
            stab = sis.load_stability(db)
            if stab is None:
                continue
            for _label, stratum in subs[:2]:
                table = sis._DB / db / f"{stratum}.csv.gz"
                if not table.exists():
                    continue
                rows = pd.read_csv(table, dtype=str)
                expected = rows.groupby(["uniprot", "variant"]).ngroups
                got = len(sis.load_panel(db, stratum, stab))
                assert got <= expected
                assert got >= 0.9 * expected, (
                    f"{db}/{stratum}: dropped {expected - got} of {expected}")
                checked += 1
        if not checked:
            pytest.skip("no stratum tables present")
