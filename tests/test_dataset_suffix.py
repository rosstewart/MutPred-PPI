"""The mapping-generation stamp must be changeable in one place.

`_mapped090826` is a date stamp for the mapping run that produced the canonical
tables. It appears in every dataset name and therefore in every filename derived
from one. When the data is remapped, the stamp changes -- and anything still
holding the old literal will either fail loudly or, far worse, quietly load an
array from the superseded generation.

So the resolution path derives its names from `legacy_guard.DATASET_SUFFIX`.
These tests reload the modules under a patched constant and require the derived
names to move with it; a re-introduced literal fails here rather than in a figure
six months later.

They do NOT claim the whole repo is literal-free -- docs, tests and one-off
scripts still spell it out. What is pinned is the path that RESOLVES METHOD
OUTPUT FILES.
"""
import json
import subprocess
import sys

import pytest

from utils.legacy_guard import DATASET_SUFFIX

FAKE = "_mapped999999"

# re-imports the package in a subprocess per case; see tests/conftest.py for the opt-in flags.
pytestmark = pytest.mark.slow

# Patching the stamp means re-importing the modules that derive names from it.
# Doing that in-process with importlib.reload rebinds their classes, so other
# tests that catch those exception types start failing -- the reload is run in a
# SUBPROCESS instead, which proves propagation and leaves this interpreter alone.
_PROBE = """
import json, sys
sys.path.insert(0, "src")
import utils.legacy_guard as lg
lg.DATASET_SUFFIX = %r                      # before anything imports it
import utils.gcv_common as g
import analysis.method_names as mn

# The consumers that used to spell the stamp out by hand.  Each must now
# derive it, so every one of these moves when lg.DATASET_SUFFIX moves.
import data_processing.training_sets.prepare_gcv_tables as pgt
import analysis.export_cv_reference as ecr
import analysis.generate_training_table as gtt
import evaluation.run_varchamp_blind_test as bt
import run_benchmarks as rb

print(json.dumps({
    "configs":  sorted(g.DATASET_CONFIGS),
    "rows":     sorted(c.rows_file for c in g.DATASET_CONFIGS.values()),
    "aliases":  g.DATASET_ALIASES,
    "short":    sorted(mn._SHORT_DATASET_NAMES),
    "derived": {
        "prepare_gcv_tables.DATASETS":  sorted(pgt.DATASETS),
        "prepare_gcv_tables.sources":   sorted(str(v) for v in pgt.DATASETS.values()),
        "export_cv_reference.NAMING":   sorted(ecr.NAMING),
        "generate_training_table":      [gtt._SAHNI, gtt._FRAGOZA, gtt._VARCHAMP,
                                         gtt._SF, gtt._SFVC],
        "blind_test":                   [bt.DEFAULT_TRAIN_DATASET,
                                         bt.SAHNI_ONLY_DATASET, bt.TEST_CFG.name],
        "run_benchmarks":               rb.GCV_DATASETS + [rb.ABLATION_DATASET],
    },
}))
""" % FAKE


@pytest.fixture(scope="module")
def remapped():
    out = subprocess.run([sys.executable, "-c", _PROBE],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


class TestSingleSource:
    def test_dataset_configs_follow_the_suffix(self, remapped):
        assert remapped["configs"], "no datasets configured"
        for name in remapped["configs"]:
            assert name.endswith(FAKE), name

    def test_row_filenames_follow_the_suffix(self, remapped):
        for f in remapped["rows"]:
            assert FAKE in f, f

    def test_aliases_resolve_to_the_new_stamp(self, remapped):
        for base, full in remapped["aliases"].items():
            assert not base.endswith(FAKE), "base names carry no stamp"
            assert full == base + FAKE

    def test_display_name_map_follows_the_suffix(self, remapped):
        assert remapped["short"]
        for full in remapped["short"]:
            assert full.endswith(FAKE), full

    def test_dataset_name_is_idempotent(self):
        from utils.gcv_common import dataset_name
        once = dataset_name("sahni_only")
        assert dataset_name(once) == once


class TestNoStaleFallback:
    def test_baseline_loader_refuses_an_unstamped_name(self):
        """The property that matters: no silent fallback to an unsuffixed file.

        Accepting both forms is what let MutPred2 load from a short-key filename
        while SAAMBE-3D/MutPPI/MutPPI+ silently vanished -- and would equally
        allow an array from a superseded mapping to be plotted.
        """
        from analysis.roc_plots import _baseline_array
        with pytest.raises(ValueError, match="mapped"):
            _baseline_array("not_a_known_dataset", "_preds.npy")

    def test_known_short_key_resolves_to_a_stamped_path(self):
        from analysis.roc_plots import _baseline_array
        from utils.legacy_guard import DATASET_SUFFIX
        p = _baseline_array("sahni_fragoza", "_SAAMBE-3D_preds.npy")
        assert DATASET_SUFFIX in p

    def test_an_alias_still_writes_under_the_stamped_name(self):
        """`--dataset sahni_only` is legitimate and must resolve to the stamp.

        The importer writes arrays under `cfg.name`, not under whatever the user
        typed, so the convenient short form cannot produce an unstamped file
        that a later run would fail to find.
        """
        from utils.gcv_common import dataset_config
        from utils.legacy_guard import DATASET_SUFFIX
        for typed in ("sahni_only", "sahni_only" + DATASET_SUFFIX):
            assert dataset_config(typed).name.endswith(DATASET_SUFFIX)

    def test_every_configured_dataset_is_stamped(self):
        """Nothing unstamped can be reached through the config table at all."""
        from utils.gcv_common import DATASET_CONFIGS
        from utils.legacy_guard import DATASET_SUFFIX
        assert DATASET_CONFIGS
        for name, cfg in DATASET_CONFIGS.items():
            assert name.endswith(DATASET_SUFFIX), name
            assert cfg.name == name


class TestDerivedConsumers:
    """Modules that used to hardcode the stamp must now follow it.

    Each of these was a literal until the single-point-of-truth pass; if one
    regresses, swapping in a remapped dataset half-migrates the repo and the
    failure shows up as a missing file deep in a run.
    """

    @pytest.mark.parametrize("key", [
        "prepare_gcv_tables.DATASETS",
        "prepare_gcv_tables.sources",
        "export_cv_reference.NAMING",
        "generate_training_table",
        "blind_test",
        "run_benchmarks",
    ])
    def test_consumer_follows_the_suffix(self, remapped, key):
        values = remapped["derived"][key]
        assert values, f"{key} produced nothing"
        for v in values:
            assert FAKE in v, f"{key}: {v!r} did not follow the stamp"
            assert DATASET_SUFFIX not in v, f"{key}: {v!r} kept the real stamp"
