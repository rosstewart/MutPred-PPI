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
    autism   WT  ~14 GB  VT  ~5 GB
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer


def _read_fasta(fasta_path: str) -> dict[str, str]:
    sequences: dict[str, str] = {}
    with open(fasta_path) as f:
        key = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                key = line[1:].strip().replace("/", "_").replace(".", "_")
                sequences[key] = ""
            elif key is not None:
                sequences[key] += line.upper().replace("-", "")
    return sequences


def _load_done(h5_path: str) -> set[str]:
    if not Path(h5_path).exists():
        return set()
    try:
        with h5py.File(h5_path, "r") as f:
            return set(f.keys())
    except Exception:
        return set()


def _get_t5_model(device: torch.device):
    link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print(f"Loading ProtT5: {link}", flush=True)
    model = T5EncoderModel.from_pretrained(link)
    if device.type == "cpu":
        model = model.to(torch.float32)
    model = model.to(device).eval()
    vocab = T5Tokenizer.from_pretrained(link, do_lower_case=False)
    return model, vocab


def embed_sequences(
    sequences: dict[str, str],
    model,
    vocab,
    device: torch.device,
    h5_path: str,
    max_residues: int = 4000,
    max_seq_len: int = 1000,
    max_batch: int = 100,
) -> None:
    """Embed sequences and append results directly to h5_path (resume-safe)."""

    # Sort longest-first for efficient batching
    sorted_seqs = sorted(sequences.items(), key=lambda kv: len(kv[1]), reverse=True)
    total = len(sorted_seqs)
    done = 0

    batch: list = []
    start = time.time()

    for seq_idx, (seq_id, seq) in enumerate(sorted_seqs, 1):
        seq_clean = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
        seq_len = len(seq_clean)
        seq_spaced = " ".join(list(seq_clean))
        batch.append((seq_id, seq_spaced, seq_len))

        n_res_batch = sum(s_len for _, _, s_len in batch) + seq_len
        flush = (
            len(batch) >= max_batch
            or n_res_batch >= max_residues
            or seq_idx == total
            or seq_len > max_seq_len
        )

        if not flush:
            continue

        pdb_ids, seqs, seq_lens = zip(*batch)
        batch = []

        token_encoding = vocab.batch_encode_plus(
            list(seqs), add_special_tokens=True, padding="longest"
        )
        input_ids = torch.tensor(token_encoding["input_ids"]).to(device)
        attention_mask = torch.tensor(token_encoding["attention_mask"]).to(device)

        try:
            with torch.no_grad():
                emb_repr = model(input_ids, attention_mask=attention_mask)
        except RuntimeError as e:
            print(f"[WARN] RuntimeError embedding batch containing {pdb_ids[0]}: {e}", flush=True)
            continue

        with h5py.File(h5_path, "a") as hf:
            for batch_idx, seq_id in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                emb = emb_repr.last_hidden_state[batch_idx, :s_len].detach().cpu().numpy().astype(np.float32)
                if seq_id in hf:
                    continue
                hf.create_dataset(seq_id, data=emb)
                done += 1

        elapsed = time.time() - start
        print(
            f"  [{seq_idx}/{total}] {done} written, {elapsed:.0f}s elapsed",
            flush=True,
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
