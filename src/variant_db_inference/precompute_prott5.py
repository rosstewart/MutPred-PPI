#!/usr/bin/env python
"""Precompute ProtT5 embeddings for a variant database dataset and save to H5.

Each key in the output H5 is a sequence ID from the FASTA; each value is an
L×1024 float32 array of per-residue embeddings.  Supports resume: sequences
already present in the output H5 are skipped.

Run for each dataset (nohup recommended — takes hours for large databases):

    nohup conda run -n ppi python precompute_prott5.py \\
        --fasta $MUTPRED_DATA_ROOT/clinvar/clinvar_interaction_loss_wt_and_vt.fasta \\
        --out $MUTPRED_DATA_ROOT/clinvar/prott5_embeddings.h5 \\
        --device cuda:0 > precompute_clinvar.log 2>&1 &

Storage estimates (float32, L×1024 per residue):
    clinvar  WT  ~18 GB  VT  ~740 GB
    gnomad   WT  ~3 GB   VT  ~212 GB
    cosmic   WT  ~17 GB  VT  ~1,060 GB
    hgmd     WT  ~7 GB   VT  ~24 GB
    neurodev WT  ~14 GB  VT  ~5 GB
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from utils.embeddings import (PROTT5_MODEL, embed_sequences as _embed_sequences_shared,  # noqa: E402
                              load_prott5)
from utils.sequences import h5_safe_key, read_fasta  # noqa: E402


def _read_fasta(fasta_path: str) -> dict[str, str]:
    """{key: sequence} from a WT_VT-format FASTA. `key` is the whole header,
    H5-safe-mangled, so a wild type and its variants key to different H5
    dataset names -- see `utils.sequences.h5_safe_key`.
    """
    return read_fasta(fasta_path, lambda h: h5_safe_key(h, whole=True),
                      on_duplicate="last")


def _load_done(h5_path: str) -> set[str]:
    if not Path(h5_path).exists():
        return set()
    try:
        with h5py.File(h5_path, "r") as f:
            return set(f.keys())
    except Exception:
        return set()


def _get_t5_model(device: torch.device):
    """Alias for the one shared loader; see `utils.embeddings.load_prott5`."""
    print(f"Loading ProtT5: {PROTT5_MODEL}", flush=True)
    return load_prott5(device)


def embed_sequences(
    sequences: dict[str, str],
    model,
    vocab,
    device: torch.device,
    h5_path: str,
    batch_residue_budget: int = 4000,
    max_batch: int = 100,
) -> None:
    """Embed sequences and append results directly to h5_path (resume-safe).

    A thin wrapper around `utils.embeddings.embed_sequences`, using its `sink`
    parameter to write and discard each batch as it completes rather than
    holding the whole embedding set in memory: some variant-DB FASTAs embed to
    ~1 TB (see module docstring), which does not fit in RAM.

    OOM behaviour changed on 2026-09-10 and is now strictly safer: this entry
    point used to pass `on_oom="skip"`, dropping every sequence in an OOMing
    batch (up to `max_batch`=100) without naming any of them. The shared
    function now retries each sequence alone first and skips only the
    individual sequences that still cannot fit, printing each one. Fewer
    sequences go missing, and the ones that do are named on stderr and
    available as `.skipped`.

    The `[n/total]` progress count is "successfully embedded so far" rather
    than "attempted so far". The `{written}` count stays exact: it is
    incremented here, once per NEW H5 dataset actually created.

    Storage dtype is forced to float32 explicitly, matching this entry point's
    previous behaviour: the model may run in float16 on GPU (the checkpoint is
    "half"-precision by name), and casting only on the CPU path -- as
    `_get_t5_model` does -- would otherwise leave GPU-computed embeddings
    written as float16.
    """
    total = len(sequences)
    written = 0
    start = time.time()

    def sink(batch: dict) -> None:
        nonlocal written
        with h5py.File(h5_path, "a") as hf:
            for seq_id, emb in batch.items():
                if seq_id in hf:
                    continue
                hf.create_dataset(seq_id, data=emb.astype(np.float32))
                written += 1

    def progress(n_done: int) -> None:
        elapsed = time.time() - start
        print(f"  [{n_done}/{total}] {written} written, {elapsed:.0f}s elapsed", flush=True)

    _embed_sequences_shared(
        sequences, model, vocab, device,
        batch_residue_budget=batch_residue_budget,
        max_batch=max_batch,
        sink=sink,
        progress=progress,
    )


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}", flush=True)

    print(f"Reading FASTA: {args.fasta}", flush=True)
    all_seqs = _read_fasta(args.fasta)
    print(f"  {len(all_seqs)} sequences in FASTA", flush=True)

    done_keys = _load_done(args.out)
    if done_keys:
        print(f"  {len(done_keys)} already in {args.out} — skipping", flush=True)
    remaining = {k: v for k, v in all_seqs.items() if k not in done_keys}
    print(f"  {len(remaining)} sequences to embed", flush=True)

    if not remaining:
        print("Nothing to do.", flush=True)
        return

    os.makedirs(Path(args.out).parent, exist_ok=True)

    model, vocab = _get_t5_model(device)

    print(f"\nEmbedding {len(remaining)} sequences → {args.out}", flush=True)
    embed_sequences(remaining, model, vocab, device, args.out)

    total_done = len(_load_done(args.out))
    print(f"\nDone. {total_done}/{len(all_seqs)} sequences in {args.out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Precompute ProtT5 per-residue embeddings to H5")
    p.add_argument("--fasta", required=True,
                   help="FASTA file with all sequences (WT, VT, partners) for the dataset")
    p.add_argument("--out", required=True,
                   help="Output H5 file path (will be created/appended to for resume)")
    p.add_argument("--device", default="",
                   help="PyTorch device string (e.g. 'cuda:0'). Defaults to auto-detect.")
    main(p.parse_args())
