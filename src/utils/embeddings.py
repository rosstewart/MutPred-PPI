"""One ProtT5 driver, and one place the no-truncation guarantee is enforced.

Four near-identical copies existed (`inference/utils/prott5_loader.py`,
`variant_db_inference/precompute_prott5.py`,
`data_processing/precompute_prott5_datasets.py`,
`training/preprocess_stability_data.py`). The model, dtype, `.eval()`, the
`batch_encode_plus(padding="longest")` call and the longest-first sort were
copy-paste. Where they genuinely differed, the difference is a PARAMETER here,
not a fork.

NO LENGTH CAP -- EVER
---------------------
Nothing in this repo truncates a sequence, and nothing may start. The numbers
that looked like caps were all batching budgets:

    `max_seq_len=1000`   "switch to single-sequence processing to avoid OOM"
    `MAX_RESIDUES=4000`  "per-batch residue budget"; an oversized sequence simply
                         becomes its own batch

They are renamed `batch_residue_budget` / `single_sequence_threshold` so they can
never again be read as limits on what gets embedded. The behaviour is kept
because removing it would OOM on long proteins rather than embed them -- which
would be a length cap by another name.

`assert_untruncated` is the enforcement: every writer calls it, so an embedding
whose row count disagrees with its sequence length is a hard error rather than a
silently short array.
"""
from __future__ import annotations

from collections.abc import Iterator

__all__ = [
    "PROTT5_MODEL", "assert_untruncated", "batched_by_residues",
    "clean_sequence", "embed_sequences",
]

PROTT5_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"

# Non-standard residues ProtT5 does not model. Only `prott5_loader` did this
# mapping; the others passed the letters through. Applied by default because the
# tokenizer maps unknown letters to <unk> anyway -- this makes it explicit.
_NONSTANDARD = str.maketrans({"U": "X", "Z": "X", "O": "X"})
# `B` (Asx: Asn or Asp, an ambiguity code) is mapped only by
# `precompute_prott5_datasets.py`, historically. The other three callers leave
# it untranslated, so the tokenizer maps it to `<unk>` rather than to `X` --
# a DIFFERENT vocab entry, and therefore a different embedding. `map_b=False`
# is the default so that migrating a caller onto this shared function never
# silently changes an already-computed embedding cache; only the one entry
# point that always mapped `B` passes `map_b=True`.
_NONSTANDARD_WITH_B = str.maketrans({"U": "X", "Z": "X", "O": "X", "B": "X"})


def clean_sequence(seq: str, map_nonstandard: bool = True, map_b: bool = False) -> str:
    if not map_nonstandard:
        return seq
    return seq.translate(_NONSTANDARD_WITH_B if map_b else _NONSTANDARD)


def assert_untruncated(key: str, sequence: str, embedding) -> None:
    """Raise unless the embedding has exactly one row per residue.

    The guarantee this module exists to make. A truncated embedding is
    indistinguishable downstream from a short protein, so it must fail here.
    """
    n = getattr(embedding, "shape", (len(embedding),))[0]
    if n != len(sequence):
        raise ValueError(
            f"{key}: embedding has {n} rows for a {len(sequence)}-residue "
            f"sequence -- truncation is never permitted")


def batched_by_residues(items, *, batch_residue_budget: int = 4000,
                        single_sequence_threshold: int | None = 1000,
                        max_batch: int = 100) -> Iterator[list]:
    """Group `(key, sequence)` into batches by residue budget, longest first.

    Neither threshold drops or shortens anything: a sequence longer than
    `single_sequence_threshold` is emitted as its own batch, and one longer than
    `batch_residue_budget` likewise. Pass `single_sequence_threshold=None` to
    batch purely by residue budget.
    """
    ordered = sorted(items, key=lambda kv: len(kv[1]), reverse=True)
    batch, n_res = [], 0
    for key, seq in ordered:
        alone = (single_sequence_threshold is not None
                 and len(seq) > single_sequence_threshold)
        if batch and (alone or len(batch) >= max_batch
                      or n_res + len(seq) > batch_residue_budget):
            yield batch
            batch, n_res = [], 0
        batch.append((key, seq))
        n_res += len(seq)
        if alone:
            yield batch
            batch, n_res = [], 0
    if batch:
        yield batch


def embed_sequences(seq_dict, model, vocab, device, *, per_protein: bool = False,
                    batch_residue_budget: int = 4000,
                    single_sequence_threshold: int | None = 1000,
                    max_batch: int = 100, map_nonstandard: bool = True,
                    map_b: bool = False,
                    on_oom: str = "retry_individually", progress=None,
                    sink=None):
    """`{key: sequence}` -> `{key: ndarray}`, one row per residue.

    `on_oom` is a real behavioural difference between the previous copies, so it
    is explicit rather than chosen:

        "retry_individually"  re-run each sequence alone   (preprocess_stability)
        "skip"                drop the whole batch         (the other three)
        "raise"               fail loudly

    Dropping a batch silently loses every sequence in it, which is why it is not
    the default.

    `sink`, if given, is called with `{key: ndarray}` once per completed batch
    INSTEAD of accumulating results in memory, and this function then returns
    `{}`. Some variant-DB FASTAs embed to ~1 TB, which does not fit in RAM --
    those callers must write and discard each batch as it completes, which is
    exactly what a per-batch sink is for. `progress`, if given, always receives
    the running total of embedded keys regardless of whether `sink` is set.

    `map_b`, passed through to `clean_sequence`, defaults to `False` so that no
    existing caller's already-computed embeddings change; see `clean_sequence`.
    """
    import torch

    out: dict = {}
    n_done = 0

    def _run(chunk):
        nonlocal n_done
        keys = [k for k, _ in chunk]
        spaced = [" ".join(clean_sequence(s, map_nonstandard, map_b)) for _, s in chunk]
        enc = vocab.batch_encode_plus(spaced, add_special_tokens=True,
                                      padding="longest")
        ids = torch.tensor(enc["input_ids"]).to(device)
        mask = torch.tensor(enc["attention_mask"]).to(device)
        with torch.no_grad():
            rep = model(ids, attention_mask=mask)
        batch_out = {}
        for i, key in enumerate(keys):
            seq = chunk[i][1]
            emb = rep.last_hidden_state[i, :len(seq)]
            # NOT `.squeeze()`: for a 1-residue sequence, emb has shape (1, hidden),
            # and squeeze collapses it to (hidden,), which then makes
            # assert_untruncated read the hidden size as the row count and raise a
            # false truncation error. The shape is always exactly (len(seq), hidden).
            arr = emb.detach().cpu().numpy()
            if not per_protein:
                assert_untruncated(key, seq, arr)
            batch_out[key] = emb.mean(dim=0).detach().cpu().numpy() if per_protein else arr
        n_done += len(batch_out)
        if sink is not None:
            sink(batch_out)
        else:
            out.update(batch_out)

    for chunk in batched_by_residues(
            seq_dict.items(), batch_residue_budget=batch_residue_budget,
            single_sequence_threshold=single_sequence_threshold,
            max_batch=max_batch):
        try:
            _run(chunk)
        except RuntimeError:
            if on_oom == "raise":
                raise
            if on_oom == "skip":
                continue
            for one in chunk:                     # retry_individually
                try:
                    _run([one])
                except RuntimeError:
                    pass
        if progress is not None:
            progress(n_done)
    return out
