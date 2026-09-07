#!/usr/bin/env python
"""Precompute ProtT5 embeddings for gnomAD and write directly to 2-hop subgraph H5.

Unlike the general precompute_prott5.py + compress_to_subgraphs.py two-pass pipeline,
this script never writes full L×1024 embeddings to disk — it runs ProtT5 and immediately
extracts and stores only the 2-hop neighborhood needed for inference.

This keeps storage requirements at ~257 GB instead of ~35 TB for gnomAD.

Restrictions applied vs the full gnomAD FASTA:
  - Only variants present in gnomad_allele_frequencies.tsv (AF-annotated)
  - Only variants whose interactor has an AF3 contact graph (.mat file)
  - Partners without an AF3 graph or FASTA entry are skipped

Position convention:
  - gnomad_allele_frequencies.tsv uses 1-based positions (e.g. "A154T")
  - gnomad_interaction_loss_wt_and_vt.fasta uses 0-based positions (e.g. "A153T")
  - This script converts AF keys to 0-based for FASTA lookup

Output HDF5 structure (same as compress_to_subgraphs.py output):
  /{complex_id}/{variant}/
      node_emb    (k, 1024) float32
      edge_index  (2, e)    int32
      mut_diff    (1024,)   float32
      attrs: mut_local_idx int

Usage:
    # Delete old full-embedding H5 first to free ~5 TB, then:
    nohup /home/rcstewart/miniconda3/envs/ppi/bin/python \\
        /data/ross/ppi_lossgain/interaction_loss/publication/src/variant_db_inference/precompute_gnomad_subgraphs.py \\
        --device cuda:0 \\
        >> /data/ross/ppi_lossgain/interaction_loss/gnomad/precompute_subgraphs.log 2>&1 &
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch
from transformers import T5EncoderModel, T5Tokenizer

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA_ROOT  # noqa: E402


_BASE = DATA_ROOT / "gnomad"
_FASTA     = _BASE / "gnomad_interaction_loss_wt_and_vt.fasta"
_AF_FILE   = _BASE / "gnomad_allele_frequencies.tsv"
_GRAPH_DIR = _BASE / "af3_graphs"
_OUT_H5    = _BASE / "prott5_subgraphs.h5"


# ── subgraph helpers (same logic as compress_to_subgraphs.py) ─────────────────

def _get_2hop(G_csr, mut_idx: int):
    hop1 = set(G_csr.indices[G_csr.indptr[mut_idx]:G_csr.indptr[mut_idx + 1]])
    hop1.add(mut_idx)
    hop2 = set(hop1)
    for nb in list(hop1):
        hop2.update(G_csr.indices[G_csr.indptr[nb]:G_csr.indptr[nb + 1]])
    return sorted(hop2)


def _extract_subgraph(G_csr, hop2_nodes: list, n_total: int):
    hop2_arr = np.array(hop2_nodes, dtype=np.int32)
    in_hop2  = np.zeros(n_total, dtype=bool)
    in_hop2[hop2_arr] = True
    G_coo = G_csr.tocoo()
    mask  = in_hop2[G_coo.row] & in_hop2[G_coo.col]
    local_map = np.full(n_total, -1, dtype=np.int32)
    local_map[hop2_arr] = np.arange(len(hop2_arr), dtype=np.int32)
    edge_index = np.stack(
        [local_map[G_coo.row[mask]], local_map[G_coo.col[mask]]], axis=0
    ).astype(np.int32)
    return edge_index, hop2_arr, local_map


def _is_complete(vgrp) -> bool:
    try:
        return "node_emb" in vgrp and "mut_diff" in vgrp
    except Exception:
        return False


# ── ProtT5 helpers ─────────────────────────────────────────────────────────────

def _load_prott5(device: torch.device):
    link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print(f"Loading ProtT5: {link}", flush=True)
    model = T5EncoderModel.from_pretrained(link)
    if device.type == "cpu":
        model = model.to(torch.float32)
    model = model.to(device).eval()
    vocab = T5Tokenizer.from_pretrained(link, do_lower_case=False)
    return model, vocab


def _embed_batch(seqs: list[str], model, vocab, device,
                 max_residues: int = 3000) -> list[np.ndarray]:
    """Embed a list of sequences (assumed same or similar length). Returns list of (L,1024)."""
    results: list[np.ndarray | None] = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)

    batch_indices: list[int] = []
    batch_seqs_sp: list[str] = []
    batch_lens:    list[int] = []
    n_res = 0

    def _flush():
        if not batch_indices:
            return
        enc = vocab.batch_encode_plus(batch_seqs_sp, add_special_tokens=True, padding="longest")
        ids  = torch.tensor(enc["input_ids"]).to(device)
        mask = torch.tensor(enc["attention_mask"]).to(device)
        with torch.no_grad():
            out = model(ids, attention_mask=mask)
        for k, (orig_i, s_len) in enumerate(zip(batch_indices, batch_lens)):
            results[orig_i] = (
                out.last_hidden_state[k, :s_len].detach().cpu().numpy().astype(np.float32)
            )
        batch_indices.clear(); batch_seqs_sp.clear(); batch_lens.clear()

    for i in order:
        seq = seqs[i].replace("U", "X").replace("Z", "X").replace("O", "X").upper()
        s_len = len(seq)
        if n_res + s_len > max_residues and batch_indices:
            _flush()
            n_res = 0
        batch_indices.append(i)
        batch_seqs_sp.append(" ".join(list(seq)))
        batch_lens.append(s_len)
        n_res += s_len

    _flush()
    return results


# ── data loading ──────────────────────────────────────────────────────────────

def _load_af_variants_0based(af_path: str, mat_inters: set[str]) -> dict[str, list[str]]:
    """Load AF file, convert 1-based positions to 0-based, filter to mat interactors.

    Returns: inter_id → [variant_0based, ...]
    """
    inter_to_variants: dict[str, list[str]] = {}
    with open(af_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            key = parts[0]  # "UNIPROT VARIANT_1BASED"
            if " " not in key:
                continue
            inter, var_1b = key.split(" ", 1)
            if inter not in mat_inters:
                continue
            # Convert 1-based → 0-based
            try:
                aa_ref = var_1b[0]
                aa_alt = var_1b[-1]
                pos_0b = int(var_1b[1:-1]) - 1
                var_0b = f"{aa_ref}{pos_0b}{aa_alt}"
            except (ValueError, IndexError):
                continue
            inter_to_variants.setdefault(inter, []).append(var_0b)
    # Deduplicate
    return {k: list(dict.fromkeys(v)) for k, v in inter_to_variants.items()}


def _load_fasta_filtered(fasta_path: str, needed_keys: set[str]) -> dict[str, str]:
    """Load only FASTA entries whose key is in needed_keys.

    The gnomAD FASTA repeats the same WT entry once per complex it appears in.
    Using assignment (seqs[key] = "") rather than setdefault ensures the second
    occurrence overwrites rather than appends, keeping the correct single-length seq.
    """
    seqs: dict[str, str] = {}
    key = None
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                raw = line[1:].strip()
                key = raw.replace("|", " ")
                if key not in needed_keys:
                    key = None
                else:
                    seqs[key] = ""  # reset on each occurrence; last one wins
            elif key is not None:
                seqs[key] += line.upper().replace("-", "")
    return seqs


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    t0 = time.time()
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}", flush=True)
    print(f"Output: {args.out}", flush=True)

    # ── mat file index ────────────────────────────────────────────────────────
    mat_files = sorted(glob.glob(str(Path(args.graph_dir) / "*.mat")))
    inter_to_mats: dict[str, list[str]] = {}
    for m in mat_files:
        inter = Path(m).stem.split("_")[0]
        inter_to_mats.setdefault(inter, []).append(m)
    mat_inters = set(inter_to_mats)
    print(f"  {len(mat_files)} mat files, {len(mat_inters)} unique interactors", flush=True)

    # ── AF variants (0-based) for interactors that have mat files ─────────────
    print(f"Loading AF variants from {args.af_file} ...", flush=True)
    inter_to_af = _load_af_variants_0based(args.af_file, mat_inters)
    total_vt = sum(len(v) for v in inter_to_af.values())
    print(f"  {len(inter_to_af)} interactors, {total_vt:,} AF variants", flush=True)

    # ── build set of FASTA keys we need ──────────────────────────────────────
    print("Building needed FASTA key set ...", flush=True)
    needed_fasta: set[str] = set()
    partner_ids: set[str] = set()
    for inter_id, mats in inter_to_mats.items():
        if inter_id not in inter_to_af:
            continue
        needed_fasta.add(inter_id)  # WT interactor
        for var_0b in inter_to_af[inter_id]:
            needed_fasta.add(f"{inter_id} {var_0b}")  # VT interactor
        for m in mats:
            partner = "_".join(Path(m).stem.split("_")[1:])
            needed_fasta.add(partner)
            partner_ids.add(partner)
    print(f"  Need {len(needed_fasta):,} FASTA entries", flush=True)

    # ── load filtered FASTA ────────────────────────────────────────────────────
    print(f"Loading filtered FASTA from {args.fasta} ...", flush=True)
    fasta_seqs = _load_fasta_filtered(args.fasta, needed_fasta)
    wt_count = sum(1 for k in fasta_seqs if " " not in k)
    vt_count = sum(1 for k in fasta_seqs if " " in k)
    print(f"  {wt_count:,} WT, {vt_count:,} VT sequences loaded", flush=True)

    # ── load ProtT5 ────────────────────────────────────────────────────────────
    model, vocab = _load_prott5(device)

    # ── open output H5 ─────────────────────────────────────────────────────────
    f_out = h5py.File(args.out, "a", libver="latest")

    # Resume: count complete entries
    n_existing = 0
    for cid in f_out.keys():
        cgrp = f_out[cid]
        for var in list(cgrp.keys()):
            if _is_complete(cgrp[var]):
                n_existing += 1
            else:
                del cgrp[var]
    print(f"  {n_existing} subgraph entries already complete — resuming", flush=True)

    n_written = n_skipped = n_missing_seq = n_missing_emb = n_bound = n_dim_mismatch = 0
    n_inter_done = 0

    interactors = sorted(inter_to_af)

    for inter_id in interactors:
        if inter_id not in fasta_seqs:
            n_missing_seq += len(inter_to_af[inter_id]) * len(inter_to_mats.get(inter_id, []))
            continue

        variants = inter_to_af[inter_id]
        mats     = inter_to_mats.get(inter_id, [])

        # ── embed WT interactor ──────────────────────────────────────────────
        wt_seq = fasta_seqs[inter_id]
        wt_embs = _embed_batch([wt_seq], model, vocab, device)
        wt_inter = wt_embs[0]
        if wt_inter is None:
            n_missing_emb += len(variants) * len(mats)
            continue

        # ── embed all VT sequences for this interactor (same length = efficient batch) ──
        vt_keys = [f"{inter_id} {v}" for v in variants]
        vt_seqs = [fasta_seqs.get(k) for k in vt_keys]
        present  = [(i, s) for i, s in enumerate(vt_seqs) if s is not None]

        if not present:
            del wt_inter
            continue

        vt_indices, vt_seq_list = zip(*present)
        vt_emb_list = _embed_batch(list(vt_seq_list), model, vocab, device)

        # Map back: variant index → embedding
        vt_emb_map: dict[int, np.ndarray] = {}
        for local_i, (orig_i, _) in enumerate(present):
            if vt_emb_list[local_i] is not None:
                vt_emb_map[orig_i] = vt_emb_list[local_i]

        # ── process each partner complex ─────────────────────────────────────
        for mat_path in mats:
            complex_id = Path(mat_path).stem
            partner_id = "_".join(complex_id.split("_")[1:])

            if partner_id not in fasta_seqs:
                n_missing_seq += len(variants)
                continue

            # Embed WT partner
            partner_emb_list = _embed_batch([fasta_seqs[partner_id]], model, vocab, device)
            partner_emb = partner_emb_list[0]
            if partner_emb is None:
                n_missing_emb += len(variants)
                continue

            # Load contact graph
            mat     = sio.loadmat(mat_path)
            n_inter = int(mat["NRR"].flat[0])
            G       = mat["G"].tocsr()
            n_total = G.shape[0]

            if wt_inter.shape[0] != n_inter or partner_emb.shape[0] != (n_total - n_inter):
                n_dim_mismatch += len(variants)
                del partner_emb
                continue

            if complex_id not in f_out:
                f_out.create_group(complex_id)
            cgrp = f_out[complex_id]

            for var_i, variant in enumerate(variants):
                # Resume
                if variant in cgrp and _is_complete(cgrp[variant]):
                    n_skipped += 1
                    continue

                mut_idx = int(variant[1:-1])
                if mut_idx >= n_inter:
                    n_bound += 1
                    continue

                vt_inter = vt_emb_map.get(var_i)
                if vt_inter is None:
                    n_missing_emb += 1
                    continue
                if vt_inter.shape[0] != n_inter:
                    n_dim_mismatch += 1
                    continue

                # 2-hop subgraph
                hop2_nodes = _get_2hop(G, mut_idx)
                hop2_arr   = np.array(hop2_nodes, dtype=np.int32)
                edge_index, _, local_map = _extract_subgraph(G, hop2_nodes, n_total)

                # Node embeddings: VT for interactor, WT for partner
                node_emb = np.empty((len(hop2_arr), 1024), dtype=np.float32)
                inter_mask   = hop2_arr < n_inter
                partner_mask = ~inter_mask
                if inter_mask.any():
                    node_emb[inter_mask]  = vt_inter[hop2_arr[inter_mask]]
                if partner_mask.any():
                    node_emb[partner_mask] = partner_emb[hop2_arr[partner_mask] - n_inter]

                mut_diff      = (vt_inter[mut_idx] - wt_inter[mut_idx]).astype(np.float32)
                mut_local_idx = int(local_map[mut_idx])

                if variant in cgrp:
                    del cgrp[variant]
                vgrp = cgrp.create_group(variant)
                vgrp.create_dataset("node_emb",  data=node_emb,  dtype=np.float32)
                vgrp.create_dataset("edge_index", data=edge_index, dtype=np.int32)
                vgrp.create_dataset("mut_diff",   data=mut_diff,  dtype=np.float32)
                vgrp.attrs["mut_local_idx"] = mut_local_idx
                n_written += 1

            del partner_emb

        del wt_inter, vt_emb_map, vt_emb_list
        gc.collect()
        torch.cuda.empty_cache()

        n_inter_done += 1
        if n_inter_done % 50 == 0:
            elapsed = time.time() - t0
            total_done = n_existing + n_written
            print(
                f"  [{n_inter_done}/{len(interactors)}] "
                f"written={n_written}  skipped={n_skipped}  "
                f"missing_seq={n_missing_seq}  missing_emb={n_missing_emb}  "
                f"dim_mismatch={n_dim_mismatch}  bound={n_bound}  "
                f"elapsed={elapsed/60:.1f}m",
                flush=True,
            )
            f_out.flush()

    f_out.flush()
    f_out.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.", flush=True)
    print(f"  Written:      {n_written:,}", flush=True)
    print(f"  Skipped:      {n_skipped:,}", flush=True)
    print(f"  Missing seq:  {n_missing_seq:,}", flush=True)
    print(f"  Missing emb:  {n_missing_emb:,}", flush=True)
    print(f"  Bound errors: {n_bound:,}", flush=True)
    print(f"  Dim mismatch: {n_dim_mismatch:,}", flush=True)
    print(f"  Total entries: {n_existing + n_written:,}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Precompute ProtT5 gnomAD AF-only variants directly to 2-hop subgraph H5"
    )
    p.add_argument("--fasta",      default=str(_FASTA))
    p.add_argument("--af-file",    default=str(_AF_FILE))
    p.add_argument("--graph-dir",  default=str(_GRAPH_DIR))
    p.add_argument("--out",        default=str(_OUT_H5))
    p.add_argument("--device",     default="",
                   help="PyTorch device (e.g. 'cuda:0'). Defaults to auto-detect.")
    run(p.parse_args())


if __name__ == "__main__":
    main()
