#!/usr/bin/env python3
"""ProtT5 embeddings for a canonical dataset.

Reads the canonical rows table (via gcv_common) and emits:
    {accession}              -> (L, 1024) float32   wild type
    {accession}_{mutation}   -> (L, 1024) float32   mutant, 1-based mutation

Keys are 1-based throughout, matching the tables; nothing converts between
conventions. There is no maximum sequence length: ProtT5 uses relative position
embeddings, MAX_RESIDUES is the per-batch memory budget, and an oversized
sequence simply becomes its own batch.

Usage:
    conda run -n ppi env OPENBLAS_NUM_THREADS=1 python \
        src/data_processing/precompute_prott5.py \
        --dataset sahni_fragoza_varchamp_all_mapped090826 --device cuda:0
"""
from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import DATASETS_DIR, DATA_CACHES_DIR  # noqa: E402


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pooled_t5")

TRANSFORMER_LINK = "Rostlab/prot_t5_xl_half_uniref50-enc"
TRAINING_CSV = str(DATA_CACHES_DIR / "training_data_internal.csv")
MAX_RESIDUES = 4000   # per-batch residue budget -- the actual memory control
MAX_BATCH = 100
# No MAX_SEQ_LEN. ProtT5 uses relative position embeddings and has no architectural
# length limit; the old `MAX_SEQ_LEN = 1000` silently *skipped* (not cropped) every
# protein above it -- 125 proteins touching 1,361 rows (5.8%) of the 090826 master.
# It bought nothing: a sequence longer than MAX_RESIDUES simply becomes its own
# batch, and the longest sequence in the master is 3,685 aa, under that budget.
# If a future dataset ever does need clipping, crop rather than skip: choose the
# window so it covers every mutation in the protein and maximises the flanking
# margin on both sides.
SAVE_EVERY = 500


def load_model(device: torch.device):
    logger.info("Loading ProtT5 from %s", TRANSFORMER_LINK)
    model = T5EncoderModel.from_pretrained(TRANSFORMER_LINK)
    if device.type == "cpu":
        model = model.to(torch.float32)
    model = model.to(device).eval()
    vocab = T5Tokenizer.from_pretrained(TRANSFORMER_LINK, do_lower_case=False)
    return model, vocab


def apply_mutation(seq: str, mutation: str) -> str:
    """Apply single amino acid substitution. mutation like 'E80K' (1-based)."""
    wt_aa = mutation[0]
    mut_aa = mutation[-1]
    pos_1b = int("".join(c for c in mutation[1:-1] if c.isdigit()))
    pos_0b = pos_1b - 1
    if pos_0b >= len(seq) or seq[pos_0b] != wt_aa:
        return seq  # sequence mismatch — return unchanged
    lst = list(seq)
    lst[pos_0b] = mut_aa
    return "".join(lst)


def collect_sequences(df: pd.DataFrame) -> Dict[str, str]:
    """Collect {key: sequence} for all WT proteins and mutant variants.

    Keys are `{accession}` for wild type and `{accession}_{mutation}` for the
    mutant, with the mutation string exactly as the canonical tables store it --
    1-based. Nothing converts between conventions anywhere in the pipeline.
    """
    seqs: Dict[str, str] = {}

    # WT sequences for all unique proteins
    for _, row in df.iterrows():
        inter_id = str(row["interactor"])
        partner_id = str(row["partner"])
        inter_seq = str(row["interactor_sequence"]) if pd.notna(row["interactor_sequence"]) else ""
        partner_seq = str(row["partner_sequence"]) if pd.notna(row["partner_sequence"]) else ""

        if inter_seq and inter_id not in seqs:
            seqs[inter_id] = inter_seq
        if partner_seq and partner_id not in seqs:
            seqs[partner_id] = partner_seq

    # Mutant sequences for each (interactor, mutation) pair
    for _, row in df.iterrows():
        inter_id = str(row["interactor"])
        mutation = str(row["mutation"])
        mut_key = f"{inter_id}_{mutation}"
        if mut_key not in seqs and inter_id in seqs:
            mut_seq = apply_mutation(seqs[inter_id], mutation)
            seqs[mut_key] = mut_seq

    return seqs


def embed_batch(
    ids: List[str],
    seqs: List[str],
    model,
    vocab,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    results: Dict[str, np.ndarray] = {}
    # Add spaces between amino acids for ProtT5 tokenizer
    spaced = [" ".join(list(s.upper().replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")))
              for s in seqs]
    ids_batch = vocab.batch_encode_plus(spaced, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids_batch["input_ids"]).to(device)
    attention_mask = torch.tensor(ids_batch["attention_mask"]).to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = out.last_hidden_state.cpu().float().numpy()
    for i, (key, seq) in enumerate(zip(ids, seqs)):
        L = len(seq)
        results[key] = embeddings[i, :L]  # strip padding and EOS token
    return results


def embed_all(
    sequences: Dict[str, str],
    model,
    vocab,
    device: torch.device,
    existing: Dict[str, np.ndarray],
    out_path: str,
) -> Dict[str, np.ndarray]:
    cache = dict(existing)
    todo = [(k, v) for k, v in sequences.items() if k not in cache and len(v) > 0]
    longest = max((len(v) for _, v in todo), default=0)
    logger.info("%d sequences to embed (%d already cached, longest %d aa)",
                len(todo), len(existing), longest)
    if longest > MAX_RESIDUES:
        logger.warning("longest sequence %d aa exceeds MAX_RESIDUES=%d; it will be "
                       "embedded alone in its own batch", longest, MAX_RESIDUES)

    batch_ids: List[str] = []
    batch_seqs: List[str] = []
    n_residues = 0

    for i, (key, seq) in enumerate(todo):
        L = len(seq)
        if (n_residues + L > MAX_RESIDUES or len(batch_ids) >= MAX_BATCH) and batch_ids:
            cache.update(embed_batch(batch_ids, batch_seqs, model, vocab, device))
            batch_ids, batch_seqs, n_residues = [], [], 0

        batch_ids.append(key)
        batch_seqs.append(seq)
        n_residues += L

        if (i + 1) % SAVE_EVERY == 0:
            if batch_ids:
                cache.update(embed_batch(batch_ids, batch_seqs, model, vocab, device))
                batch_ids, batch_seqs, n_residues = [], [], 0
            logger.info("Saving checkpoint (%d/%d keys)...", len(cache), len(todo) + len(existing))
            with open(out_path, "wb") as f:
                pickle.dump(cache, f)

    if batch_ids:
        cache.update(embed_batch(batch_ids, batch_seqs, model, vocab, device))

    return cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dataset", default="sahni_fragoza_varchamp_all_mapped090826",
                   help="canonical dataset to embed (see gcv_common.DATASET_CONFIGS)")
    p.add_argument("--out", default=None,
                   help="output pkl (default: datasets/mapped090826/<dataset>_prott5.pkl)")
    p.add_argument("--csv", default=None,
                   help="embed an arbitrary rows CSV instead of a named dataset")
    args = p.parse_args()

    device = torch.device(args.device)

    if args.csv:
        logger.info("Reading rows CSV: %s", args.csv)
        df = pd.read_csv(args.csv)
        stem = Path(args.csv).stem
    else:
        import sys as _s
        _s.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
        from evaluation.gcv_common import DATASET_CONFIGS, load_data  # noqa: E402
        logger.info("Loading canonical dataset: %s", args.dataset)
        df = load_data(DATASET_CONFIGS[args.dataset])
        stem = args.dataset
    logger.info("rows: %d", len(df))

    out_path = args.out or str(DATASETS_DIR / "mapped090826" / f"{stem}_prott5.pkl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting sequences...")
    sequences = collect_sequences(df)
    logger.info("Total unique sequences (WT + mutant): %d", len(sequences))

    # Load existing cache if resuming
    existing: Dict[str, np.ndarray] = {}
    if Path(out_path).exists():
        logger.info("Loading existing cache from %s", out_path)
        with open(out_path, "rb") as f:
            existing = pickle.load(f)
        logger.info("  %d keys already cached", len(existing))

    model, vocab = load_model(device)

    t0 = time.time()
    cache = embed_all(sequences, model, vocab, device, existing, out_path)
    elapsed = time.time() - t0
    logger.info("Done. %d keys total in %.1f min", len(cache), elapsed / 60)

    with open(out_path, "wb") as f:
        pickle.dump(cache, f)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
