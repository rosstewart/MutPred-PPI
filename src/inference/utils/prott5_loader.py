#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:33:22 2020

@author: mheinzinger --> rstewart
"""

import time
import torch
import h5py
from transformers import T5EncoderModel, T5Tokenizer
from utils.embeddings import embed_sequences  # noqa: E402
from utils.sequences import h5_safe_key, read_fasta as _read_fasta_shared  # noqa: E402


def get_T5_model(model_dir, device, transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"):
    print("Loading: {}".format(transformer_link))
    if model_dir is not None:
        print("##########################")
        print("Loading cached model from: {}".format(model_dir))
        print("##########################")
    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)
    # only cast to full-precision if no GPU is available
    if device==torch.device("cpu"):
        print("Casting model to full precision for running on CPU ...")
        model.to(torch.float32)

    model = model.to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False )
    return model, vocab


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
                   single_sequence_threshold=1000,   # above this, a sequence is batched ALONE
                   max_batch=100):
    """Embed every sequence in `seq_path` and write them to `emb_path`.

    Batching is shared with the other ProtT5 entry points via
    `utils.embeddings`. Neither budget limits what gets embedded -- an
    oversized sequence becomes its own batch and is embedded in full.
    `embed_sequences` asserts one row per residue, so a truncation cannot
    pass silently.
    """
    seq_dict = read_fasta(seq_path)
    emb_dict = embed_sequences(
        seq_dict, model, vocab, device,
        per_protein=per_protein,
        batch_residue_budget=batch_residue_budget,
        single_sequence_threshold=single_sequence_threshold,
        max_batch=max_batch,
        on_oom="skip",     # historical behaviour of this entry point
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
