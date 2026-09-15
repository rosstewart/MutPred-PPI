"""Variant-repository prediction coverage must match what the row tables record.

A `{db}_rows.csv.gz` keeps the FULL enumeration of (interactor, partner,
mutation) triples and flags each one `in_embedding_store`, because the ProtT5
caches cover a subset: gnomAD enumerates 10.5M triples and about 1.5M are
embedded. That is a deliberate design -- the enumeration is the denominator for
every enrichment, and the flag says which rows could be scored.

The failure mode this guards is a SILENT one. If a cache is rebuilt, or a row
table is regenerated against a different cache, the prediction count drifts away
from the flag and nothing complains: the figures simply draw on fewer variants
than the tables claim. Nothing else in the suite checks it -- `in_embedding_store`
appeared in exactly one source file and no test before this.

Two independent statements are asserted:

  1. the completion sentinel accounts for every row it was given, and
  2. the number of predictions tracks `in_embedding_store`.

(2) is a tolerance, not an equality. The flag is computed when the table is
built and the caches are topped up afterwards, so a few rows flagged absent do
score; conversely a flagged-present row can still fail on an empty 2-hop
subgraph. Both are bounded and enumerated by the sentinel, so the tolerance is
small and a real regression (a cache pointing at the wrong build) blows past it.
"""
from pathlib import Path

import pandas as pd
import pytest

from paths import DATA_ROOT, DATASETS_DIR, VARIANT_DBS_DIR

DBS = ("clinvar", "gnomad", "cosmic", "hgmd", "neurodev", "asd")

#: Allowed drift between predictions and `in_embedding_store == 1`, as a
#: fraction of the flagged rows. Observed across all six repositories at the
#: 090826 rebaseline: worst case is neurodev at +5.5% (996 rows that scored
#: despite being flagged absent, from a post-build ProtT5 top-up).
TOLERANCE = 0.10


def _rows_table(db: str) -> Path:
    return DATASETS_DIR / "variant_dbs" / f"{db}_rows.csv.gz"


def _predictions(db: str) -> Path:
    """The predictions, in either supported layout (collected or per-database)."""
    collected = VARIANT_DBS_DIR / f"{db}_mutpred_ppi_predictions.tsv"
    if collected.exists():
        return collected
    return DATA_ROOT / db / "mutpred_ppi_predictions.tsv"


def _available() -> list[str]:
    return [db for db in DBS
            if _rows_table(db).exists() and _predictions(db).exists()]


needs_data = pytest.mark.skipif(
    not _available(),
    reason="no variant-repository row tables with predictions on this machine")


@needs_data
@pytest.mark.requires_data
@pytest.mark.parametrize("db", DBS)
def test_sentinel_accounts_for_every_row(db):
    """The `.complete` sentinel's buckets must sum to the rows it was given.

    The sentinel is the only record of why a row went unscored. If its buckets
    do not add up, rows were dropped somewhere that does not report them.
    """
    if db not in _available():
        pytest.skip(f"{db}: no row table or no predictions")
    sentinel = Path(f"{_predictions(db)}.complete")
    if not sentinel.exists():
        pytest.skip(f"{db}: no completion sentinel (run was interrupted)")

    stats = {}
    for line in sentinel.read_text().splitlines():
        if not line.strip():
            continue
        k, _, v = line.rpartition("\t")
        stats[k] = int(v)

    n_rows = len(pd.read_csv(_rows_table(db), usecols=["interactor"]))
    total = sum(stats.values())
    assert total == n_rows, (
        f"{db}: sentinel buckets sum to {total:,} but the row table has "
        f"{n_rows:,}. {n_rows - total:+,} rows are unaccounted for.\n"
        f"  buckets: {stats}")


@needs_data
@pytest.mark.requires_data
@pytest.mark.parametrize("db", DBS)
def test_predictions_track_the_embedding_store_flag(db):
    """Prediction count must stay close to `in_embedding_store == 1`.

    Drift here means the predictions and the row table were built against
    different caches, which silently changes the denominator of every
    enrichment computed from them.
    """
    if db not in _available():
        pytest.skip(f"{db}: no row table or no predictions")

    flagged = int(
        (pd.read_csv(_rows_table(db), usecols=["in_embedding_store"])
         ["in_embedding_store"] == 1).sum())
    with open(_predictions(db)) as fh:
        n_pred = sum(1 for _ in fh) - 1          # minus the header

    assert flagged > 0, f"{db}: no rows flagged in_embedding_store"
    drift = abs(n_pred - flagged) / flagged
    assert drift <= TOLERANCE, (
        f"{db}: {n_pred:,} predictions against {flagged:,} rows flagged "
        f"in_embedding_store ({drift:.1%} drift, tolerance {TOLERANCE:.0%}). "
        f"The predictions and the row table were probably built against "
        f"different ProtT5 caches.")
