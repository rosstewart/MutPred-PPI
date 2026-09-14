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

# --- repo-relative path resolution (see src/paths.py) ---
from paths import DATASETS_DIR, TRAINING_EVAL_DIR  # noqa: E402
from utils import mutations  # noqa: E402
from utils.embeddings import (PROTT5_MODEL,  # noqa: E402
                              embed_sequences as _embed_sequences_shared, load_prott5)
from utils.gcv_common import (dataset_arg, dataset_config,  # noqa: E402
                              dataset_name, load_data)
from utils.runtime import resolve_device  # noqa: E402


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pooled_t5")

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
    """Alias for the one shared loader; see `utils.embeddings.load_prott5`."""
    logger.info("Loading ProtT5 from %s", PROTT5_MODEL)
    return load_prott5(device)




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

    # Mutant sequences for each (interactor, mutation) pair.
    #
    # A mutation that does not fit its sequence is SKIPPED and counted. The
    # previous local `apply_mutation` returned the sequence UNCHANGED on a
    # mismatch, which stored a wild-type sequence under a variant key -- the
    # resulting mutation-site diff is exactly zero, so the model sees no signal
    # and nothing anywhere reports it. Measured on the canonical tables: 0 of
    # 53,239 rows mismatch today, so this guards the future rather than changing
    # the present.
    n_skipped = 0
    for _, row in df.iterrows():
        inter_id = str(row["interactor"])
        mutation = str(row["mutation"])
        mut_key = f"{inter_id}_{mutation}"
        if mut_key not in seqs and inter_id in seqs:
            mut_seq = mutations.apply(seqs[inter_id], mutation)
            if mut_seq is None:
                n_skipped += 1
                continue
            seqs[mut_key] = mut_seq
    if n_skipped:
        logger.warning("%d mutations did not match their sequence and were skipped",
                       n_skipped)

    return seqs


def embed_all(
    sequences: Dict[str, str],
    model,
    vocab,
    device: torch.device,
    existing: Dict[str, np.ndarray],
    out_path: str,
) -> Dict[str, np.ndarray]:
    """Embed every not-yet-cached sequence, checkpointing to `out_path` every
    SAVE_EVERY newly-embedded keys.

    A thin wrapper around `utils.embeddings.embed_sequences`. This dataset's
    full embedding set fits in memory (unlike the ~1 TB variant-DB FASTAs
    `precompute_prott5.py` handles), so the whole cache stays resident and is
    periodically re-dumped in full -- matching this entry point's previous
    checkpointing granularity (a full pickle overwrite every SAVE_EVERY keys,
    not a per-key H5 append).

    Two behaviours were removed on 2026-09-10, neither of which changes this
    entry point's output. `map_b=True` was this script's private mapping of the
    ambiguity code B (Asx) -> X; B does not occur in any canonical sequence, so
    folding it unconditionally in `clean_sequence` is identical here. And
    `on_oom="raise"` (kill the run on any OOM) is replaced by the single shared
    policy -- retry each sequence alone, skip and report only what still cannot
    fit -- so a transient OOM no longer discards a partially built cache that
    took hours to fill. Anything genuinely unembeddable is named on stderr and
    is simply absent from the pickle, which `assert_untruncated` and the
    downstream key lookups already treat as a hard error.
    """
    cache = dict(existing)
    todo = {k: v for k, v in sequences.items() if k not in cache and len(v) > 0}
    longest = max((len(v) for v in todo.values()), default=0)
    logger.info("%d sequences to embed (%d already cached, longest %d aa)",
                len(todo), len(existing), longest)
    if longest > MAX_RESIDUES:
        logger.warning("longest sequence %d aa exceeds MAX_RESIDUES=%d; it will be "
                       "embedded alone in its own batch", longest, MAX_RESIDUES)

    total = len(todo) + len(existing)
    since_checkpoint = 0

    def sink(batch: Dict[str, np.ndarray]) -> None:
        nonlocal since_checkpoint
        cache.update(batch)
        since_checkpoint += len(batch)
        if since_checkpoint >= SAVE_EVERY:
            logger.info("Saving checkpoint (%d/%d keys)...", len(cache), total)
            with open(out_path, "wb") as f:
                pickle.dump(cache, f)
            since_checkpoint = 0

    _embed_sequences_shared(
        todo, model, vocab, device,
        batch_residue_budget=MAX_RESIDUES,
        max_batch=MAX_BATCH,
        map_nonstandard=True,
        sink=sink,
    )
    return cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dataset", type=dataset_arg,
                   default=dataset_name("sahni_fragoza_varchamp_all"),
                   help="canonical dataset to embed; short aliases accepted "
                        "(sahni_fragoza, varchamp_all, ...)")
    p.add_argument("--out", default=None,
                   help="output pkl (default: datasets/training_eval/<dataset>_prott5.pkl)")
    p.add_argument("--csv", default=None,
                   help="embed an arbitrary rows CSV instead of a named dataset")
    args = p.parse_args()

    device = resolve_device(args.device)

    if args.csv:
        logger.info("Reading rows CSV: %s", args.csv)
        df = pd.read_csv(args.csv)
        stem = Path(args.csv).stem
    else:
        logger.info("Loading canonical dataset: %s", args.dataset)
        df = load_data(dataset_config(args.dataset))
        stem = args.dataset
    logger.info("rows: %d", len(df))

    out_path = args.out or str(TRAINING_EVAL_DIR / f"{stem}_prott5.pkl")
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
