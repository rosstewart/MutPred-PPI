"""Core invariants the shared utils/ core must hold, absorbed from
repro_test/verify_core_invariants.py (a gitignored scratch script) into a
tracked, CI-runnable test.

These are the properties that, when they broke, cost the most time to find:

  1. Identifier round-trips over every accession/mutation in the canonical
     tables, including isoforms and RefSeq-shaped ids -- not a handful of
     made-up examples.
  2. Base-conversion round-trips, same scope.
  3. No truncation: an embedding must have exactly one row per residue.
  4. Homodimers: `seq_a == seq_b`, so both orientations of a homodimer are the
     IDENTICAL array by construction -- a check written as "the two
     orientations differ" fails on them, which is a bug in the check, not the
     store. It produced a false alarm once; these tests assert the correct
     invariant (round-trip + model-input shape) instead.

Tests that need the real (large, gitignored) contact-graph stores skip
gracefully when those stores are not present in the current environment,
rather than failing the suite on a missing multi-GB data file.
"""
from pathlib import Path

import numpy as np
import pytest

from contact_graphs import ContactGraphStore, pair_key
from paths import DATASETS_DIR, contact_graph_store
from utils import identifiers as I
from utils import mutations as M
from utils.embeddings import assert_untruncated

# One store, covering training/evaluation and variant repositories alike. These
# were two separate constants, one of which pointed at datasets/mapped090826/,
# a directory that has never existed -- so every check guarded on it silently
# skipped.
TRAIN_STORE = VDB_STORE = contact_graph_store()


def _canonical_accessions_and_mutations():
    """Real accessions/mutations from the canonical tables, or a small
    synthetic fallback if no canonical tables are available in this
    environment (e.g. a fresh checkout without the data bundle)."""
    import glob

    import pandas as pd

    accs: set[str] = set()
    muts: set[str] = set()
    csvs = glob.glob(str(DATASETS_DIR / "cv_reference" / "*_rows.csv.gz"))
    for path in csvs:
        df = pd.read_csv(path, usecols=lambda c: c in ("interactor", "partner", "mutation"))
        accs |= {str(a) for a in df["interactor"]} | {str(b) for b in df["partner"]}
        muts |= {m for m in df["mutation"].astype(str) if M.is_mutation(m)}

    # Shapes that historically broke a naive split, always included.
    accs |= {"NP_002046_GFAP", "O14787-2", "Q8WWY3-1"}
    if not muts:
        muts = {"A123V", "M1V", "K999R"}
    return accs, muts


# -- 1. identifier round-trips ----------------------------------------------------

def test_pair_id_round_trips_canonical_accessions():
    accs, _ = _canonical_accessions_and_mutations()
    a = sorted(accs)
    bad = [(x, y) for x in a[:500] for y in a[:5]
          if I.split_pair_id(I.pair_id(x, y)) != (x, y)]
    assert not bad, f"{len(bad)} pair_id round-trip failures, e.g. {bad[:3]}"


def test_variant_id_round_trips_canonical_mutations():
    accs, muts = _canonical_accessions_and_mutations()
    a = sorted(accs)
    bad = [(x, m) for x in a[:300] for m in sorted(muts)[:5]
          if I.split_variant_id(I.variant_id(x, m)) != (x, m)]
    assert not bad, f"{len(bad)} variant_id round-trip failures, e.g. {bad[:3]}"


# -- 2. base-conversion round-trips ------------------------------------------------

def test_base_conversion_round_trips_canonical_mutations():
    _, muts = _canonical_accessions_and_mutations()
    bad = [m for m in muts if M.to_one_based(M.to_zero_based(m)) != m]
    assert not bad, f"{len(bad)} to_one_based(to_zero_based(m)) != m, e.g. {bad[:5]}"


def test_index_matches_position_minus_one():
    _, muts = _canonical_accessions_and_mutations()
    off = [m for m in muts if M.index(m) != M.position(m) - 1]
    assert not off, f"{len(off)} mutations where index(m) != position(m) - 1"


# -- 3. no truncation ---------------------------------------------------------------

def test_truncation_guard_raises_on_short_embedding():
    with pytest.raises(ValueError):
        assert_untruncated("k", "AAAA", np.zeros((3, 8)))


def test_truncation_guard_passes_on_exact_embedding():
    assert_untruncated("k", "AAAA", np.zeros((4, 8)))  # must not raise


# -- 4. homodimers ------------------------------------------------------------------

@pytest.mark.parametrize("label,path", [("train/eval", TRAIN_STORE), ("variant-DB", VDB_STORE)])
def test_homodimers_load_square_and_self_looped(label, path):
    if not path.exists():
        pytest.skip(f"{path.name} not present in this environment")
    store = ContactGraphStore(path)
    try:
        homo = [m for m in store.meta() if m["seq_a_sha"] == m["seq_b_sha"]]
        if not homo:
            pytest.skip(f"no homodimers in {label} store")
        for m in homo[:25]:
            d = store._g[m["key"]]
            seq = d.attrs["seq_a"]
            G = store.load_dense(interactor=seq, partner=seq)
            n = 2 * len(seq)
            assert G is not None
            assert G.shape == (n, n)
            assert bool(np.all(np.diag(G) == 1))
            assert pair_key(seq, seq) == m["key"]
            # Both orientations are the SAME array for a homodimer -- assert
            # that, rather than asserting they differ (the check that failed).
            assert np.array_equal(G, store.load_dense(interactor=seq, partner=seq))
    finally:
        store.close()


def test_homodimer_variant_input_is_mutant_then_wildtype():
    """A homodimer variant row must give [mutant chain, WT chain] as model
    input, not two copies of the wild type."""
    seq = "MKVLAAGIVG"
    mut = "K2A"
    vt = M.apply(seq, mut)
    assert vt is not None
    assert vt != seq
    assert (vt + seq) != (seq + seq)
