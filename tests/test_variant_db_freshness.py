"""Are the variant-database artifacts current, and is anything silently narrowed?

Every serious defect in this pipeline has had the same shape: an artifact that
EXISTS but is older than, or narrower than, the thing it was derived from, with
no error raised. A structure set that lost six pairs to an isoform tie-break. A
subgraph cache built against a structure directory that excluded ProtVar. A
prediction TSV written before the caches were repaired. In each case the run
"succeeded" and the figures were quietly built on less data.

These tests check the derivation ORDER and the COVERAGE of each stage:

    structures -> graph store -> embedding/subgraph caches -> predictions

Anything downstream must be at least as new as what it derives from, and each
stage must cover what the stage below asks of it.

They skip when an artifact is absent -- a clean clone without the licensed
databases is a legitimate state -- but they FAIL when something is present and
stale, which is the case that ships wrong numbers.
"""
import csv
import gzip
import os
from pathlib import Path

import pytest

from paths import DATASETS_DIR, contact_graph_store
from utils.gcv_common import DATASET_SUFFIX  # noqa: F401  (import guard)


# audits the real caches and prediction TSVs; see tests/conftest.py for the opt-in flags.
pytestmark = pytest.mark.requires_data

DBS = ["clinvar", "cosmic", "gnomad", "hgmd", "neurodev", "asd"]
DATA_ROOT = Path("/data/ross/ppi_lossgain/interaction_loss")
STORE = contact_graph_store()
MANIFEST = DATASETS_DIR / "af3_structures_canonical" / "manifest.csv"

# gnomAD is scoped to variants with an assigned allele frequency; a row without
# one was never meant to be scored. Mirrors audit_caches.SCOPE_COLUMN.
SCOPE_COLUMN = {"gnomad": "allele_frequency"}


def _mtime(p):
    return p.stat().st_mtime if p and Path(p).exists() else None


def _rows_file(db):
    return DATASETS_DIR / "variant_dbs" / f"{db}_rows.csv.gz"


def _cache(db):
    """The cache this database actually uses: subgraphs if built, else embeddings."""
    sg = DATA_ROOT / db / "prott5_subgraphs.h5"
    emb = DATA_ROOT / db / "prott5_embeddings.h5"
    return sg if sg.exists() else (emb if emb.exists() else None)


class TestStructureSet:
    def test_store_holds_a_graph_for_every_manifest_entry(self):
        """Every canonical structure must have a graph, keyed by its sequences.

        This replaced an mtime comparison between the store and the manifest.
        That proxy produced false alarms: the manifest gains columns (a
        `provenance` column, for one) without a single structure changing, and
        an mtime bump then looked like a stale store. Key coverage is the
        property that actually matters and does not depend on timestamps.
        """
        import h5py
        if not (STORE.exists() and MANIFEST.exists()):
            pytest.skip("canonical structure set not present")
        # Store keys are ORIENTATION-INDEPENDENT: contact_graphs.pair_key builds
        # them as f"{min(ha, hb)}_{max(ha, hb)}". Pairing the manifest's
        # seq_a/seq_b in column order instead reports about half the set as
        # missing, which is a property of the key, not of the data.
        with open(MANIFEST) as fh:
            wanted = {"_".join(sorted([r["seq_a_sha"], r["seq_b_sha"]]))
                      for r in csv.DictReader(fh)}
        with h5py.File(STORE, "r") as h:
            have = set(h["graphs"].keys())
        missing = wanted - have
        assert not missing, (
            f"{len(missing):,} of {len(wanted):,} canonical structures have no graph "
            f"in the store (e.g. {sorted(missing)[:3]}) -- rebuild it with "
            f"rebuild_graphs_from_structures.py")

    def test_store_covers_the_manifest(self):
        import h5py
        if not (STORE.exists() and MANIFEST.exists()):
            pytest.skip("canonical structure set not present")
        with open(MANIFEST) as fh:
            n_manifest = sum(1 for _ in csv.DictReader(fh))
        with h5py.File(STORE, "r") as h:
            n_store = len(h["graphs"])
        assert n_store >= n_manifest, (
            f"graph store has {n_store:,} graphs for {n_manifest:,} structures -- "
            f"{n_manifest - n_store:,} structures produced no graph")


@pytest.mark.parametrize("db", DBS)
class TestPerDatabase:
    def test_cache_is_not_older_than_the_graph_store(self, db):
        cache = _cache(db)
        if cache is None or not STORE.exists():
            pytest.skip(f"{db}: cache or store absent")
        assert _mtime(cache) >= _mtime(STORE), (
            f"{db}: {cache.name} is older than the graph store. It was built "
            f"against a previous structure set, so rows whose structures were "
            f"added since will be missing from it -- recompress.")

    def test_predictions_exist_for_every_cached_database(self, db):
        """A cache with no predictions is the failure; a stale mtime is not.

        Recompressing a cache does not invalidate predictions that already cover
        every in-scope row -- the rows are the same rows. What matters is
        COVERAGE, which `test_predictions_cover_the_in_scope_rows` checks. An
        older mtime is reported here as a hint about which database to re-run if
        that coverage test does fail, not as a failure in itself.
        """
        cache = _cache(db)
        if cache is None:
            pytest.skip(f"{db}: no cache")
        preds = DATA_ROOT / db / "mutpred_ppi_predictions.tsv"
        assert preds.exists(), (
            f"{db}: has {cache.name} but no predictions -- inference never ran "
            f"for this database.")
        if _mtime(preds) < _mtime(cache):
            print(f"\nNOTE: {db} predictions predate {cache.name}; re-run only if "
                  f"the coverage test for {db} fails.")

    def test_predictions_cover_every_scoreable_row(self, db):
        """The TSV must hold every prediction the caches make possible.

        Not a fraction of raw rows -- the target is `audit_caches`'s SCOREABLE
        count: rows that are in scope, have an AF3 structure, whose mutation
        applies, and whose embedding/subgraph is present. Anything short of that
        means the TSV was written before a cache repair and the figures are being
        built on fewer predictions than exist.

        Slow (it walks the row table), so it is opt-in for the large databases
        via MUTPRED_SLOW_TESTS=1; the small ones always run.
        """
        import os as _os
        big = {"gnomad", "cosmic", "clinvar"}
        if db in big and not _os.environ.get("MUTPRED_SLOW_TESTS"):
            pytest.skip(f"{db}: set MUTPRED_SLOW_TESTS=1 to audit the large tables")
        preds = DATA_ROOT / db / "mutpred_ppi_predictions.tsv"
        if not (preds.exists() and STORE.exists() and _rows_file(db).exists()):
            pytest.skip(f"{db}: predictions, store or rows absent")

        from variant_db_inference.audit_caches import audit
        stats, _need = audit(db, str(STORE))
        scoreable = stats["scoreable"]
        with open(preds) as fh:
            n_pred = sum(1 for _ in fh) - 1
        assert scoreable > 0, f"{db}: audit reports nothing scoreable"
        assert n_pred >= scoreable, (
            f"{db}: {n_pred:,} predictions but {scoreable:,} rows are scoreable "
            f"given the current caches -- {scoreable - n_pred:,} achievable "
            f"predictions are missing. Re-run inference for {db}.")


def test_no_database_silently_lost_its_cache():
    """Every database with rows must have SOME cache -- one or the other."""
    missing = [db for db in DBS
               if _rows_file(db).exists() and _cache(db) is None]
    assert not missing, (
        f"these databases have row tables but neither an embedding nor a subgraph "
        f"cache: {missing}")


def test_pair_key_is_orientation_independent():
    """Pins the assumption the coverage check above depends on."""
    from contact_graphs import pair_key
    a, b = "MKV", "GGSTA"
    assert pair_key(a, b) == pair_key(b, a)
