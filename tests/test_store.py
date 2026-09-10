"""Tests for ContactGraphStore -- content-addressed read/write, orientation,
and the homodimer case.

The homodimer test matters specifically because an earlier check in this repo
concluded the store was "broken" on homodimers, when in fact the check itself
was wrong: for a homodimer (seq_a == seq_b), the two orientations ARE the same
array by construction, so asserting they differ is the wrong test. These tests
assert the correct thing instead -- round-trip and model-input shape.
"""
import numpy as np
import pytest

from contact_graphs import ContactGraphStore, dense_from_edges, edges_from_dense


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "test_store.h5"


def _edges_for(seq_a: str, seq_b: str) -> np.ndarray:
    """A trivial deterministic edge set: residue i of chain A contacts residue
    i of chain B (when both exist), plus one within-chain edge, so put/get has
    something non-trivial to round-trip."""
    n = len(seq_a) + len(seq_b)
    G = np.zeros((n, n), dtype=np.uint8)
    if len(seq_a) > 1:
        G[0, 1] = G[1, 0] = 1
    for i in range(min(len(seq_a), len(seq_b))):
        G[i, len(seq_a) + i] = G[len(seq_a) + i, i] = 1
    return edges_from_dense(G)


# -- basic put/get round-trip -----------------------------------------------------

def test_put_then_get_returns_same_edges(store_path):
    store = ContactGraphStore(store_path, "w")
    seq_a, seq_b = "MKTAYIAK", "MSGLGRS"
    edges = _edges_for(seq_a, seq_b)
    key = store.put(seq_a, seq_b, edges)
    store.close()

    store = ContactGraphStore(store_path, "r")
    hit = store.get(seq_a, seq_b)
    assert hit is not None
    ei, n, is_first = hit
    assert n == len(seq_a) + len(seq_b)
    assert is_first
    assert np.array_equal(dense_from_edges(ei, n), dense_from_edges(edges, n))
    store.close()


def test_put_is_idempotent_without_overwrite(store_path):
    store = ContactGraphStore(store_path, "w")
    seq_a, seq_b = "MKTAYIAK", "MSGLGRS"
    edges = _edges_for(seq_a, seq_b)
    key1 = store.put(seq_a, seq_b, edges, source="first")
    key2 = store.put(seq_a, seq_b, edges, source="second")  # no overwrite=True
    assert key1 == key2
    assert len(store) == 1
    store.close()


def test_pair_key_is_order_independent_in_store(store_path):
    store = ContactGraphStore(store_path, "w")
    seq_a, seq_b = "MKTAYIAK", "MSGLGRS"
    key_ab = store.put(seq_a, seq_b, _edges_for(seq_a, seq_b))
    assert key_ab in store
    store.close()

    # A fresh get() for the REVERSED pair must hit the same record.
    store = ContactGraphStore(store_path, "r")
    hit = store.get(seq_b, seq_a)
    assert hit is not None
    store.close()


# -- orientation: stored reversed, requested forward -----------------------------

def test_get_reorients_when_stored_reversed(store_path):
    seq_a, seq_b = "MKTAYIAK", "MSGLGRS"
    edges = _edges_for(seq_a, seq_b)

    # Store with seq_b first (reversed from how we will request it).
    store = ContactGraphStore(store_path, "w")
    store.put(seq_b, seq_a, _edges_for(seq_b, seq_a))
    store.close()

    # Request with seq_a first: get() must rotate indices so seq_a leads.
    store = ContactGraphStore(store_path, "r")
    hit = store.get(seq_a, seq_b)
    assert hit is not None
    ei, n, is_first = hit
    assert is_first
    dense = dense_from_edges(ei, n)
    # residue 0 of seq_a (now at array position 0) must still see its
    # cross-chain partner at position len(seq_a) + 0, matching _edges_for's
    # construction regardless of stored order.
    assert dense[0, len(seq_a)] == 1
    store.close()


def test_get_returns_none_on_miss(store_path):
    store = ContactGraphStore(store_path, "w")
    store.put("AAAA", "BBBB", _edges_for("AAAA", "BBBB"))
    store.close()

    store = ContactGraphStore(store_path, "r")
    assert store.get("CCCC", "DDDD") is None
    store.close()


# -- monomer records ---------------------------------------------------------------

def test_monomer_round_trip(store_path):
    store = ContactGraphStore(store_path, "w")
    seq = "MKTAYIAKQRQ"
    G = np.zeros((len(seq), len(seq)), dtype=np.uint8)
    G[0, 1] = G[1, 0] = 1
    edges = edges_from_dense(G)
    key = store.put(seq, "", edges)
    assert key.startswith("mono_")
    store.close()

    store = ContactGraphStore(store_path, "r")
    hit = store.get(seq, "")
    assert hit is not None
    ei, n, _ = hit
    assert n == len(seq)
    store.close()


# -- homodimer: the case an earlier (wrong) check flagged as "broken" -----------

def test_homodimer_round_trip(store_path):
    """seq_a == seq_b (a true homodimer). The two orientations are the SAME
    array by construction -- asserting they differ is the wrong test; asserting
    round-trip correctness is the right one."""
    seq = "MKTAYIAKQRQ"
    edges = _edges_for(seq, seq)

    store = ContactGraphStore(store_path, "w")
    store.put(seq, seq, edges)
    store.close()

    store = ContactGraphStore(store_path, "r")
    hit = store.get(seq, seq)
    assert hit is not None
    ei, n, is_first = hit
    assert n == 2 * len(seq)
    assert is_first
    assert np.array_equal(dense_from_edges(ei, n), dense_from_edges(edges, n))
    store.close()


def test_homodimer_model_input_shape(store_path):
    """The actual invariant that matters for training: concatenating the two
    per-chain embeddings gives a (2L, D) tensor regardless of homodimer
    status, and the store's edge_index addresses exactly those 2L nodes."""
    seq = "MKTAYIAKQRQ"
    edges = _edges_for(seq, seq)

    store = ContactGraphStore(store_path, "w")
    store.put(seq, seq, edges)
    store.close()

    store = ContactGraphStore(store_path, "r")
    edge_index = store.load_edge_index(interactor=seq, partner=seq)
    assert edge_index is not None
    n = 2 * len(seq)
    assert edge_index.max() < n
    assert edge_index.min() >= 0
    # Self-loops are always added on read (see module docstring).
    diag_present = any(edge_index[0, k] == edge_index[1, k] for k in range(edge_index.shape[1]))
    assert diag_present
    store.close()


# -- context manager ----------------------------------------------------------------

def test_store_as_context_manager(store_path):
    with ContactGraphStore(store_path, "w") as store:
        store.put("AAAA", "BBBB", _edges_for("AAAA", "BBBB"))
        assert len(store) == 1
