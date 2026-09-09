#!/usr/bin/env python
"""ESM-2 per-residue embeddings for a canonical dataset (eSIG-Net input).

Vendored from `/home/rcstewart/ppi_lossgain/esignet_scripts/precompute_esm_embeddings.py`,
which lived outside the repo and imported `esignet_gcv_iter` (since archived) for
its FASTA parsing. The embedding code is unchanged; only the input path is --
sequences come from the canonical tables instead of a FASTA.

Emits, matching what `predictors/esignet.py::_esm_diff` looks up:
    {accession}              -> (L, 1280) float32   wild type
    {accession}_{mutation}   -> (L, 1280) float32   mutant, 1-based mutation

Keys are 1-based. The upstream script's FASTA parser converted stored 0-based
variant headers to 1-based; there is nothing to convert here because the tables
are 1-based already.

Usage:
    conda run -n ppi python src/data_processing/precompute_esm2_datasets.py \\
        --dataset sahni_fragoza_varchamp_all_mapped090826 --device cuda:0
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.gcv_common import DATASET_CONFIGS, add_mutated_sequence, load_data  # noqa: E402
from paths import DATASETS_DIR  # noqa: E402

SAVE_EVERY = 500


def load_esm(model_name: str, device: torch.device):
    from transformers import AutoTokenizer, EsmModel
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_name = model_name if "/" in model_name else f"facebook/{model_name}"
    print(f"  Loading via transformers: {hf_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = EsmModel.from_pretrained(hf_name).eval().to(device)
    return model, tokenizer


@torch.no_grad()
def embed_sequences(sequences, model, tokenizer, device, batch_size=4):
    """Per-residue embeddings for [(label, seq), ...] -> {label: (L, D)}."""
    results = {}
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        seqs = [s for _, s in batch]
        inputs = tokenizer(seqs, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hidden = model(**inputs).last_hidden_state       # (B, L+2, D), BOS/EOS
        for i, (label, seq) in enumerate(batch):
            results[label] = hidden[i, 1:len(seq) + 1].cpu().float().numpy()
        if (start // batch_size) % 25 == 0:
            print(f"  {start + len(batch)}/{len(sequences)} embedded", flush=True)
    return results


def collect_sequences(df) -> dict:
    """{key: sequence} for every wild-type protein and mutant variant."""
    seqs = {}
    for a, s in zip(df["interactor"], df["interactor_sequence"]):
        seqs.setdefault(str(a), s)
    for b, s in zip(df["partner"], df["partner_sequence"]):
        seqs.setdefault(str(b), s)
    for a, m, s in zip(df["interactor"], df["mutation"], df["mutated_sequence"]):
        seqs.setdefault(f"{a}_{m}", s)
    return seqs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    out = Path(args.output or
               DATASETS_DIR / "mapped090826" / f"{args.dataset}_esm2.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading canonical dataset: {args.dataset}", flush=True)
    df = add_mutated_sequence(load_data(DATASET_CONFIGS[args.dataset]))
    print(f"  {len(df)} rows", flush=True)
    seqs = collect_sequences(df)
    print(f"  {len(seqs)} unique sequences (WT + mutant), "
          f"longest {max(len(s) for s in seqs.values())} aa", flush=True)

    cache = {}
    if out.exists():
        with open(out, "rb") as f:
            cache = pickle.load(f)
        print(f"  resuming: {len(cache)} already cached", flush=True)
    todo = [(k, v) for k, v in seqs.items() if k not in cache]
    print(f"  {len(todo)} to embed", flush=True)
    if not todo:
        return 0

    model, tokenizer = load_esm(args.esm_model, device)
    t0 = time.time()
    for i in range(0, len(todo), SAVE_EVERY):
        chunk = todo[i:i + SAVE_EVERY]
        cache.update(embed_sequences(chunk, model, tokenizer, device, args.batch_size))
        with open(out, "wb") as f:
            pickle.dump(cache, f)
        print(f"  checkpoint: {len(cache)}/{len(seqs)} keys", flush=True)
    print(f"Done. {len(cache)} keys in {(time.time() - t0) / 60:.1f} min -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
