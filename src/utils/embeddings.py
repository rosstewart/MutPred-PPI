"""One ProtT5 driver, and one place the no-truncation guarantee is enforced.

Four near-identical copies existed (`inference/pipeline/prott5_loader.py`,
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

`max_seq_len` is GONE (2026-09-10). A number that reads as a length threshold has
no place in this module even when it only steers batching -- and it was never the
thing keeping long proteins alive. Batching is now purely by residue budget, and
OOM is handled by falling back to single-sequence execution (below), which is what
actually rescues a long protein. The one thing the threshold bought was avoiding
`padding="longest"` waste when a very long sequence shares a batch with short
ones; the OOM fallback covers that case correctly instead of pre-empting it.

`batch_residue_budget` is the only knob, and it is a budget, not a cap: a sequence
longer than the whole budget still becomes its own batch and is embedded in full.

`assert_untruncated` is the enforcement: every writer calls it, so an embedding
whose row count disagrees with its sequence length is a hard error rather than a
silently short array.

ONE OOM POLICY
--------------
There used to be three (`skip` / `raise` / `retry_individually`), chosen per
caller. `skip` dropped an entire batch -- up to 100 sequences -- on one OOM,
silently. That is now impossible. The single policy is:

    batch OOMs  ->  retry every sequence in it alone
    still OOMs  ->  skip that ONE sequence, and report it

and the skipped keys are both warned about on stderr and returned via
`EmbeddingResult.skipped`, so a caller can fail on them rather than discover a
hole in an H5 later.
"""
from __future__ import annotations

import sys as _sys
from collections.abc import Iterator

__all__ = [
    "PROTT5_MODEL", "EmbeddingResult", "assert_untruncated",
    "batched_by_residues", "clean_sequence", "embed_sequences",
    "load_embeddings_h5", "load_prott5",
]

PROTT5_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"

# Non-standard residues ProtT5 does not model. Only `prott5_loader` did this
# mapping; the others passed the letters through. Applied by default because the
# tokenizer maps unknown letters to <unk> anyway -- this makes it explicit.
# Ambiguity/rare codes ProtT5 does not model, all folded to `X`.
#
# `B` (Asx: Asn or Asp) used to be a per-caller flag (`map_b`): one entry point
# mapped it to `X`, the other three left it for the tokenizer to turn into
# `<unk>` -- a different vocab entry, hence a different embedding. The flag
# existed only to avoid invalidating already-computed caches. It is gone: `B`
# does not occur in ANY canonical sequence (0 occurrences across all 2,806 rows
# of `datasets/training_eval/sequences.csv.gz`; the only non-standard residue
# present anywhere is a single `U`), so the two behaviours were identical on
# every input this repo has ever embedded. Folding `B` here is the defensible
# resolution: it is an ambiguity code, not a residue.
_NONSTANDARD = str.maketrans({"U": "X", "Z": "X", "O": "X", "B": "X"})


class EmbeddingResult(dict):
    """`{key: ndarray}` that also carries the keys an OOM forced us to skip.

    A plain dict makes a skipped sequence indistinguishable from one that was
    never requested. Callers writing an H5 should check `.skipped`.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.skipped: list = []


def clean_sequence(seq: str, map_nonstandard: bool = True) -> str:
    if not map_nonstandard:
        return seq
    return seq.translate(_NONSTANDARD)


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
                        max_batch: int = 100) -> Iterator[list]:
    """Group `(key, sequence)` into batches by residue budget, longest first.

    Nothing is dropped or shortened: a sequence longer than
    `batch_residue_budget` is emitted as its own batch and embedded in full.
    Longest-first ordering keeps `padding="longest"` waste down within a batch.
    """
    ordered = sorted(items, key=lambda kv: len(kv[1]), reverse=True)
    batch, n_res = [], 0
    for key, seq in ordered:
        if batch and (len(batch) >= max_batch
                      or n_res + len(seq) > batch_residue_budget):
            yield batch
            batch, n_res = [], 0
        batch.append((key, seq))
        n_res += len(seq)
    if batch:
        yield batch


def embed_sequences(seq_dict, model, vocab, device, *, per_protein: bool = False,
                    batch_residue_budget: int = 4000,
                    max_batch: int = 100, map_nonstandard: bool = True,
                    progress=None, sink=None, on_skip=None):
    """`{key: sequence}` -> `{key: ndarray}`, one row per residue.

    OOM handling is not a parameter. A batch that OOMs is retried one sequence
    at a time; a sequence that still OOMs alone is skipped, warned about on
    stderr, and recorded. There is no mode in which an OOM silently discards a
    whole batch of up to `max_batch` sequences, which is what `on_oom="skip"`
    used to do at three of the four call sites.

    Returns a `dict` subclass carrying a `.skipped` list of keys that could not
    be embedded (empty in the normal case), so a caller can hard-fail on a
    non-empty result rather than find the hole later. `on_skip`, if given, is
    called with each skipped key as it happens.

    `sink`, if given, is called with `{key: ndarray}` once per completed batch
    INSTEAD of accumulating results in memory, and this function then returns an
    empty result (still carrying `.skipped`). Some variant-DB FASTAs embed to
    ~1 TB, which does not fit in RAM -- those callers must write and discard each
    batch as it completes, which is exactly what a per-batch sink is for.
    `progress`, if given, always receives the running total of embedded keys
    regardless of whether `sink` is set.
    """
    import torch

    out = EmbeddingResult()
    n_done = 0

    def _run(chunk):
        nonlocal n_done
        keys = [k for k, _ in chunk]
        spaced = [" ".join(clean_sequence(s, map_nonstandard)) for _, s in chunk]
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
            max_batch=max_batch):
        try:
            _run(chunk)
        except RuntimeError:
            # Batch-level OOM: nothing is abandoned here, every sequence in the
            # batch gets its own attempt before anything can be skipped.
            if getattr(device, "type", str(device)) == "cuda":
                torch.cuda.empty_cache()
            for one in chunk:
                try:
                    _run([one])
                except RuntimeError as exc:
                    key, seq = one
                    out.skipped.append(key)
                    print(f"  [OOM] {key} ({len(seq)} residues) could not be "
                          f"embedded even alone -- skipped: {exc}",
                          file=_sys.stderr, flush=True)
                    if on_skip is not None:
                        on_skip(key)
        if progress is not None:
            progress(n_done)
    if out.skipped:
        print(f"  [OOM] {len(out.skipped)} sequence(s) skipped; "
              f"embeddings for these keys are ABSENT", file=_sys.stderr, flush=True)
    return out


def load_prott5(device, cache_dir=None):
    """Load the ProtT5 encoder + tokenizer. The only copy.

    Four identical copies existed (`inference/pipeline/prott5_loader.get_T5_model`,
    `variant_db_inference/precompute_prott5._get_t5_model`,
    `data_processing/precompute_prott5_datasets.load_model`,
    `training/preprocess_stability_data`). All four agreed -- fp32 on CPU,
    `.eval()`, `do_lower_case=False` -- and all four hardcoded the model string
    even though `PROTT5_MODEL` was right here.

    fp16 is used on CUDA only: `half()` on CPU is both unsupported for many ops
    and slower, which is why the dtype is device-dependent rather than fixed.
    """
    import torch
    from transformers import T5EncoderModel, T5Tokenizer

    kw = {"cache_dir": str(cache_dir)} if cache_dir else {}
    model = T5EncoderModel.from_pretrained(PROTT5_MODEL, **kw).to(device)
    model = model.half() if getattr(device, "type", str(device)) != "cpu" else model.float()
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(PROTT5_MODEL, do_lower_case=False, **kw)
    return model, vocab


def load_embeddings_h5(h5_path, progress: bool = False):
    """`{key: ndarray}` from a flat per-key embedding H5. The only reader.

    Two identical copies existed (`inference/pipeline/inference_utils.read_h5` and
    `variant_db_inference/run_variant_db_inference._load_embeddings_h5`): both
    iterated `f.keys()` and sliced each dataset into a dict. Consolidated
    2026-09-10.

    This loads the WHOLE file into memory, which is correct for an inference
    working directory and wrong for the multi-hundred-GB variant-DB stores --
    those use the subgraph path instead. `progress` prints the count, matching
    the variant-DB copy's logging.
    """
    import h5py

    if progress:
        print(f"Loading embeddings from {h5_path} ...", flush=True)
    out = {}
    with h5py.File(str(h5_path), "r") as f:
        for key in f.keys():
            out[key] = f[key][:]
    if progress:
        print(f"  {len(out)} sequences loaded", flush=True)
    return out
