"""The Sahni+Fragoza demonstration tier must never be mistaken for the real one.

Users without the unpublished VarChAMP measurements cannot train the all-data
model, so `reproduce_all_figures.py` falls back to a Sahni+Fragoza model that
lets them run variant-repository inference end to end. That fallback relaxes a
rule the codebase otherwise enforces -- "only the all-data model scores a
variant repository" -- so the relaxation has to stay narrow.

This is not a hypothetical risk. `results/variant_dbs/` (now under archive/)
was produced by scoring the repositories with a Sahni+Fragoza-only model, sat
next to the correct `results/variant_dbs_all_data/`, and
`stability_interaction_scatter.py` had to be taught which of the two to read.

The properties pinned here:
  * the default tier is still guarded, and still named all_data;
  * choosing the fallback requires saying so explicitly;
  * the fallback writes to a different file, so neither can overwrite the other;
  * figures from the fallback carry a visible notice.
"""
import argparse

import pytest

from variant_db_inference import run_variant_db_inference as rvdi


class TestDefaultTierIsGuarded:
    def test_default_is_all_data(self):
        assert rvdi.DEFAULT_MODEL_TIER == "all_data"

    def test_guard_still_raises_without_the_all_data_checkpoint(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="MutPred-PPI.pt"):
            rvdi.assert_all_data_model(str(tmp_path))

    def test_guard_message_points_at_the_fallback(self, tmp_path):
        """A user who cannot train the all-data model must be told what to do."""
        with pytest.raises(FileNotFoundError) as exc:
            rvdi.assert_all_data_model(str(tmp_path))
        msg = str(exc.value)
        assert "--model-tier sahni_fragoza" in msg
        assert "NOT the published" in msg or "not the published" in msg.lower()


class TestTierResolution:
    def test_all_data_tier_has_no_filename_suffix(self):
        _, suffix = rvdi.resolve_model_tier("all_data", None)
        assert suffix == ""

    def test_fallback_tier_writes_a_different_file(self):
        """Output paths must differ, or one tier silently overwrites the other."""
        _, all_data = rvdi.resolve_model_tier("all_data", None)
        _, fallback = rvdi.resolve_model_tier("sahni_fragoza", None)
        assert fallback and fallback != all_data

    def test_tiers_resolve_to_different_directories(self):
        a, _ = rvdi.resolve_model_tier("all_data", None)
        b, _ = rvdi.resolve_model_tier("sahni_fragoza", None)
        assert a != b
        assert b.name == "sahni_fragoza"

    def test_explicit_models_dir_overrides_the_tier_directory(self, tmp_path):
        d, _ = rvdi.resolve_model_tier("all_data", str(tmp_path))
        assert d == tmp_path

    def test_only_the_two_known_tiers_exist(self):
        assert set(rvdi.MODEL_TIERS) == {"all_data", "sahni_fragoza"}


class TestStamping:
    def test_notice_names_the_model_and_disclaims_the_numbers(self):
        from analysis.plot_style import DEMO_TIER_NOTICE
        assert "Sahni+Fragoza" in DEMO_TIER_NOTICE
        assert "NOT" in DEMO_TIER_NOTICE

    def test_demo_stamp_writes_text_onto_the_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from analysis.plot_style import DEMO_TIER_NOTICE, demo_stamp
        fig = plt.figure()
        before = len(fig.texts)
        demo_stamp(fig)
        assert len(fig.texts) == before + 1
        assert DEMO_TIER_NOTICE in fig.texts[-1].get_text()
        plt.close(fig)

    def test_demo_stamp_is_a_noop_for_the_published_tier(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from analysis.plot_style import demo_stamp
        fig = plt.figure()
        demo_stamp(fig, "")
        assert not fig.texts
        plt.close(fig)


class TestFigureProducersAcceptTheFlag:
    @pytest.mark.parametrize("module", [
        "analysis.variant_db_charts",
        "analysis.stability_interaction_scatter",
    ])
    def test_producer_exposes_demo_tier(self, module):
        """Both variant-DB figure producers must be able to mark their output."""
        import importlib
        mod = importlib.import_module(module)
        src = open(mod.__file__).read()
        assert "--demo-tier" in src, f"{module} cannot stamp its figures"


class TestGuardChecksIdentityNotPresence:
    """The all-data guard must verify WHICH checkpoint it found.

    It used to check only that a file named `MutPred-PPI.pt` existed in
    `--models-dir`. Three directories hold a file with that name: `weights/`
    (all-data), `weights/sahni_fragoza/` (the demonstration tier) and
    `weights/blind_test/` (whatever the blind test trained last). A mistyped
    `--models-dir` therefore passed the guard and scored every repository with
    the wrong model, producing a results tree indistinguishable from the
    published one. That is how the archived `results/variant_dbs/` tree came to
    exist, and the guard could not detect it.
    """

    def _write(self, tmp_path, digest_source):
        import shutil
        d = tmp_path / "models"
        d.mkdir()
        shutil.copy(digest_source, d / "MutPred-PPI.pt")
        return d

    def test_known_hashes_do_not_collide(self):
        assert rvdi.ALL_DATA_SHA256 not in rvdi.KNOWN_OTHER_MODELS, (
            "the all-data checkpoint is also listed as a non-all-data model")
        assert len(set(rvdi.KNOWN_OTHER_MODELS)) == len(rvdi.KNOWN_OTHER_MODELS)

    @pytest.mark.requires_data
    def test_the_published_checkpoint_matches_the_pinned_hash(self):
        from paths import WEIGHTS_DIR

        primary = WEIGHTS_DIR / "MutPred-PPI.pt"
        if not primary.exists():
            pytest.skip("weights/MutPred-PPI.pt not present")
        assert rvdi._sha256(primary) == rvdi.ALL_DATA_SHA256, (
            "weights/MutPred-PPI.pt is not the checkpoint ALL_DATA_SHA256 pins. "
            "If the model was retrained deliberately, update the constant.")

    @pytest.mark.requires_data
    def test_the_published_checkpoint_is_the_all_data_one(self):
        """Pinned by hash, but also confirm it against its qualified sibling."""
        from paths import WEIGHTS_DIR
        from utils.gcv_common import dataset_name

        primary = WEIGHTS_DIR / "MutPred-PPI.pt"
        sibling = WEIGHTS_DIR / (
            f"MutPred-PPI_{dataset_name('sahni_fragoza_varchamp_all')}"
            f"_megascale_all_all.pt")
        if not (primary.exists() and sibling.exists()):
            pytest.skip("checkpoints not present")
        assert rvdi._sha256(primary) == rvdi._sha256(sibling), (
            "weights/MutPred-PPI.pt is not byte-identical to the all-data "
            "checkpoint it is supposed to be a copy of")

    @pytest.mark.requires_data
    @pytest.mark.parametrize("subdir", ["sahni_fragoza", "blind_test"])
    def test_pointing_at_another_tier_is_refused(self, tmp_path, subdir):
        """The scenario that produced the archived results/variant_dbs/ tree."""
        from paths import WEIGHTS_DIR

        candidates = sorted((WEIGHTS_DIR / subdir).glob("*.pt"))
        if not candidates:
            pytest.skip(f"weights/{subdir}/ has no checkpoints")
        d = self._write(tmp_path, candidates[0])
        with pytest.raises(ValueError, match="not the all-data model"):
            rvdi.assert_all_data_model(str(d))

    def test_an_unrecognised_checkpoint_warns_rather_than_raises(self, tmp_path, capsys):
        """A legitimate retrain must not be blocked."""
        d = tmp_path / "models"
        d.mkdir()
        (d / "MutPred-PPI.pt").write_bytes(b"not a real checkpoint")
        rvdi.assert_all_data_model(str(d))       # must not raise
        assert "not the published all-data checkpoint" in capsys.readouterr().err

    def test_a_missing_checkpoint_still_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            rvdi.assert_all_data_model(str(tmp_path))
