#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:33:22 2020

@author: mheinzinger --> rstewart
"""

import time
import torch
import h5py

from utils.embeddings import embed_sequences, load_prott5  # noqa: E402
from utils.sequences import h5_safe_key, read_fasta as _read_fasta_shared  # noqa: E402


def get_T5_model(model_dir, device):
    """Thin alias for `utils.embeddings.load_prott5` -- the only loader.

    Kept as a name because the 3-step inference pipeline's docs refer to it.
    """
    return load_prott5(device, cache_dir=model_dir)


def read_fasta(fasta_path):
    """{key: sequence} from a WT_VT-format FASTA (`>P25054` / `>P25054 S305R`).

    `key` is the WHOLE header, H5-safe-mangled (`whole=True`): the wild type and
    each of its variants must key to DIFFERENT H5 dataset names, so first-token
    keying (which collapses `P25054 S305R` onto `P25054`) is not an option here.
    `on_duplicate="last"` matches this function's previous behaviour, which
    reset a repeated header's accumulator to empty and rebuilt it from the
    later occurrence only.
    """
    return _read_fasta_shared(fasta_path, lambda h: h5_safe_key(h, whole=True),
                              on_duplicate="last")


def get_embeddings(seq_path,
                   emb_path,
                   model,
                   vocab,
                   per_protein,          # mean-pool to one vector per protein
                   device,
                   batch_residue_budget=4000,
                   max_batch=100):
    """Embed every sequence in `seq_path` and write them to `emb_path`.

    Batching is shared with the other ProtT5 entry points via
    `utils.embeddings`. The budget does not limit what gets embedded -- an
    oversized sequence becomes its own batch and is embedded in full.
    `embed_sequences` asserts one row per residue, so a truncation cannot
    pass silently, and an OOM can no longer drop a whole batch: it falls back
    to single-sequence execution and reports anything it still cannot embed.
    """
    seq_dict = read_fasta(seq_path)
    emb_dict = embed_sequences(
        seq_dict, model, vocab, device,
        per_protein=per_protein,
        batch_residue_budget=batch_residue_budget,
        max_batch=max_batch,
    )

    with h5py.File(str(emb_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)

    # print('\n############# STATS #############')
    # print('Total number of embeddings: {}'.format(len(emb_dict)))
    # print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format( 
    #         end-start, (end-start)/len(emb_dict), avg_length))
    return True


def run_T5_from_model(seq_path, emb_path, model, vocab, device):
    get_embeddings( seq_path, emb_path, model, vocab, per_protein=False, device=device)
