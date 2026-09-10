"""Tests for utils/embeddings.py -- the shared ProtT5 driver.

The `_FakeModel`/`_FakeVocab` pair below stand in for a real T5 encoder so
these tests run in milliseconds with no GPU, no network, and no 3 GB model
download -- they exercise exactly the tensor-shape logic in `embed_sequences`,
not the model's numerics.

`test_embed_sequences_single_residue_not_flattened` is the regression test for
the `.squeeze()` bug: a one-residue sequence used to come back with shape
`(hidden,)` instead of `(1, hidden)`, which made `assert_untruncated` read the
hidden size as the row count and raise a false truncation error.
"""
import numpy as np
import pytest
import torch

from utils.embeddings import (
    assert_untruncated,
    batched_by_residues,
    clean_sequence,
    embed_sequences,
)

HIDDEN = 8


class _FakeVocab:
    """Pads with zeros to the longest sequence in the batch, ESM/T5-tokenizer-style."""

    def batch_encode_plus(self, spaced, add_special_tokens=True, padding="longest"):
        lengths = [len(s.split()) for s in spaced]
        max_len = max(lengths)
        input_ids = [[1] * n + [0] * (max_len - n) for n in lengths]
        attention_mask = [[1] * n + [0] * (max_len - n) for n in lengths]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeModel:
    """Returns a deterministic per-position embedding; ignores attention_mask
    (real T5 does not either -- padding rows are masked out, never zeroed)."""

    def __call__(self, ids, attention_mask):
        batch, max_len = ids.shape
        vals = torch.arange(batch * max_len * HIDDEN, dtype=torch.float32)
        last_hidden_state = vals.reshape(batch, max_len, HIDDEN)
        return type("Output", (), {"last_hidden_state": last_hidden_state})()


# -- clean_sequence ------------------------------------------------------------

def test_clean_sequence_maps_uzo_to_x():
    assert clean_sequence("MUZOK") == "MXXXK"


def test_clean_sequence_passthrough_when_disabled():
    assert clean_sequence("MUZOK", map_nonstandard=False) == "MUZOK"


def test_clean_sequence_leaves_standard_residues_alone():
    assert clean_sequence("MKTAYIAKQRQ") == "MKTAYIAKQRQ"


def test_clean_sequence_maps_all_ambiguity_codes():
    # The per-caller `map_b` flag was removed 2026-09-10: B does not occur in
    # any canonical sequence, so mapping it unconditionally is identical on
    # every input this repo embeds, and B is an ambiguity code (Asx), not a
    # residue ProtT5 models.
    assert clean_sequence("MBK") == "MXK"
    assert clean_sequence("MBUZOK") == "MXXXXK"


def test_clean_sequence_passthrough_when_disabled():
    assert clean_sequence("MBUZOK", map_nonstandard=False) == "MBUZOK"


# -- assert_untruncated ---------------------------------------------------------

def test_assert_untruncated_passes_on_matching_length():
    assert_untruncated("k", "MKT", np.zeros((3, HIDDEN)))  # no raise


def test_assert_untruncated_raises_on_mismatch():
    with pytest.raises(ValueError, match="truncation is never permitted"):
        assert_untruncated("k", "MKTAYIAKQRQ", np.zeros((5, HIDDEN)))


def test_assert_untruncated_one_residue():
    # The exact shape the .squeeze() bug used to destroy.
    assert_untruncated("k", "M", np.zeros((1, HIDDEN)))


# -- batched_by_residues ---------------------------------------------------------

def test_batched_by_residues_sorts_longest_first():
    items = [("a", "M" * 10), ("b", "M" * 100), ("c", "M" * 50)]
    batches = list(batched_by_residues(items, batch_residue_budget=1000))
    flat = [k for batch in batches for k, _ in batch]
    assert flat == ["b", "c", "a"]


def test_batched_by_residues_respects_budget():
    items = [(str(i), "M" * 60) for i in range(5)]
    batches = list(batched_by_residues(items, batch_residue_budget=100))
    assert all(sum(len(s) for _, s in b) <= 100 for b in batches)
    assert sum(len(b) for b in batches) == 5  # nothing dropped


def test_batched_by_residues_oversized_sequence_still_embedded_whole():
    # No length threshold exists any more (removed 2026-09-10). A sequence
    # longer than the WHOLE budget must still be emitted -- alone, in full,
    # never split and never dropped.
    items = [("long", "M" * 9000), ("short", "M" * 10)]
    batches = list(batched_by_residues(items, batch_residue_budget=4000))
    assert [k for k, _ in batches[0]] == ["long"]
    assert len(batches[0][0][1]) == 9000
    assert [k for k, _ in batches[1]] == ["short"]


def test_batched_by_residues_packs_by_budget_only():
    # Under the old threshold=1000 rule "long" would have been isolated.
    # Now the only question is whether the residue budget fits.
    items = [("long", "M" * 3000), ("short", "M" * 900)]
    batches = list(batched_by_residues(items, batch_residue_budget=4000))
    assert len(batches) == 1
    assert [k for k, _ in batches[0]] == ["long", "short"]


def test_batched_by_residues_drops_nothing():
    items = [(str(i), "M" * (i + 1)) for i in range(37)]
    batches = list(batched_by_residues(items, batch_residue_budget=50))
    seen = {k for batch in batches for k, _ in batch}
    assert seen == {str(i) for i in range(37)}


# -- embed_sequences: the .squeeze() regression --------------------------------

def test_embed_sequences_single_residue_not_flattened():
    out = embed_sequences({"single": "M"}, _FakeModel(), _FakeVocab(), "cpu")
    assert out["single"].shape == (1, HIDDEN)


def test_embed_sequences_normal_length_shape():
    out = embed_sequences({"normal": "MKTAYIAKQRQ"}, _FakeModel(), _FakeVocab(), "cpu")
    assert out["normal"].shape == (11, HIDDEN)


def test_embed_sequences_mixed_batch_shapes():
    seqs = {"single": "M", "normal": "MKTAYIAKQRQ"}
    out = embed_sequences(seqs, _FakeModel(), _FakeVocab(), "cpu")
    assert out["single"].shape == (1, HIDDEN)
    assert out["normal"].shape == (11, HIDDEN)


def test_embed_sequences_per_protein_mode_averages():
    out = embed_sequences({"normal": "MKTAYIAKQRQ"}, _FakeModel(), _FakeVocab(), "cpu",
                          per_protein=True)
    assert out["normal"].shape == (HIDDEN,)


# -- embed_sequences: the streaming sink (for datasets too large for RAM) ------

def test_embed_sequences_sink_receives_all_keys_and_returns_empty():
    seqs = {"a": "MKTAYIAKQRQ", "b": "M", "c": "MKT"}
    received: dict = {}

    def sink(batch):
        received.update(batch)

    out = embed_sequences(seqs, _FakeModel(), _FakeVocab(), "cpu", sink=sink)
    assert out == {}
    assert set(received) == {"a", "b", "c"}
    assert received["b"].shape == (1, HIDDEN)          # squeeze bug applies here too
    assert received["a"].shape == (11, HIDDEN)


def test_embed_sequences_sink_matches_non_sink_values():
    seqs = {"a": "MKTAYIAKQRQ", "b": "MKT"}
    no_sink = embed_sequences(seqs, _FakeModel(), _FakeVocab(), "cpu")

    received: dict = {}
    embed_sequences(seqs, _FakeModel(), _FakeVocab(), "cpu",
                    sink=lambda batch: received.update(batch))

    assert set(no_sink) == set(received)
    for k in no_sink:
        assert np.array_equal(no_sink[k], received[k])


def test_embed_sequences_progress_reports_running_total_with_sink():
    seqs = {str(i): "M" * (i + 1) for i in range(5)}
    counts = []
    embed_sequences(seqs, _FakeModel(), _FakeVocab(), "cpu",
                    sink=lambda batch: None, progress=counts.append,
                    batch_residue_budget=1000, max_batch=2)
    assert counts[-1] == 5
    assert counts == sorted(counts)  # monotonically non-decreasing


# -- OOM policy: one policy, no silent batch loss -------------------------------
#
# Before 2026-09-10 `on_oom` was a per-caller parameter and three of the four
# call sites passed "skip", which discarded EVERY sequence in an OOMing batch
# (up to max_batch=100) without naming any of them. These tests pin the single
# replacement policy: retry each sequence alone, skip only what still fails,
# and report it.

class _OOMOnBatchModel(_FakeModel):
    """Raises on any multi-sequence batch; succeeds on a single sequence."""

    def __call__(self, ids, attention_mask=None):
        if ids.shape[0] > 1:
            raise RuntimeError("CUDA out of memory (simulated)")
        return super().__call__(ids, attention_mask=attention_mask)


class _AlwaysOOMForKeyModel(_FakeModel):
    """Raises for one specific sequence length, even alone."""

    def __init__(self, bad_len):
        self.bad_len = bad_len

    def __call__(self, ids, attention_mask=None):
        if any(int(m.sum()) == self.bad_len for m in attention_mask):
            raise RuntimeError("CUDA out of memory (simulated)")
        return super().__call__(ids, attention_mask=attention_mask)


def test_batch_oom_retries_individually_and_loses_nothing():
    seqs = {str(i): "M" * (i + 1) for i in range(6)}
    out = embed_sequences(seqs, _OOMOnBatchModel(), _FakeVocab(), "cpu",
                          batch_residue_budget=1000, max_batch=6)
    assert set(out) == set(seqs)          # every sequence recovered
    assert out.skipped == []
    for k, v in out.items():
        assert v.shape[0] == len(seqs[k])


def test_only_the_unembeddable_sequence_is_skipped():
    seqs = {"good1": "M" * 5, "bad": "M" * 7, "good2": "M" * 3}
    out = embed_sequences(seqs, _AlwaysOOMForKeyModel(bad_len=7), _FakeVocab(),
                          "cpu", batch_residue_budget=1000, max_batch=3)
    # The whole batch OOMs because of "bad", but its neighbours survive.
    assert set(out) == {"good1", "good2"}
    assert out.skipped == ["bad"]


def test_skipped_keys_are_reported_via_callback():
    seqs = {"good": "M" * 5, "bad": "M" * 7}
    seen = []
    embed_sequences(seqs, _AlwaysOOMForKeyModel(bad_len=7), _FakeVocab(), "cpu",
                    batch_residue_budget=1000, max_batch=2, on_skip=seen.append)
    assert seen == ["bad"]


def test_skipped_is_reported_even_when_a_sink_is_used():
    # With a sink the result dict is empty by design -- .skipped must still
    # survive, or a streaming caller has no way to learn about the hole.
    seqs = {"good": "M" * 5, "bad": "M" * 7}
    received = {}
    out = embed_sequences(seqs, _AlwaysOOMForKeyModel(bad_len=7), _FakeVocab(),
                          "cpu", batch_residue_budget=1000, max_batch=2,
                          sink=received.update)
    assert out == {}
    assert out.skipped == ["bad"]
    assert set(received) == {"good"}
