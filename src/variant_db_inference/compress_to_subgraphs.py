#!/usr/bin/env python
"""Convert full-protein ProtT5 H5 embeddings to compact per-variant 2-hop subgraph H5.

With 2 GAT layers, prediction at the mutation site is a pure function of its
2-hop neighborhood.  Storing only those nodes is lossless and reduces ClinVar
from ~1.1 TB to ~110 GB, COSMIC from ~1.7 TB to ~200 GB.

Output HDF5 structure:
  /{complex_id}/{variant}/
      node_emb    (k, 1024) float32  — subgraph node features
                                       (VT interactor emb for interactor nodes,
                                        WT partner emb for partner nodes)
      edge_index  (2, e)   int32     — COO edges in local subgraph coords
      mut_diff    (1024,)  float32   — vt_inter[mut_idx] - wt_inter[mut_idx]
      attrs:
        mut_local_idx  int  — mutation site index within local node list

Safety: write to a temp sub-key first, then rename — corrupt partial writes
won't block resume.  On resume, any entry missing mut_diff is cleaned up.

Usage:
    # Process ClinVar first (fits in available disk); verify, delete old, then COSMIC.
    nohup /home/rcstewart/miniconda3/envs/ppi/bin/python compress_to_subgraphs.py \\
        --dataset clinvar >> /data/ross/ppi_lossgain/interaction_loss/clinvar/compress.log 2>&1 &

    # After verifying and deleting old clinvar prott5_embeddings.h5:
    nohup /home/rcstewart/miniconda3/envs/ppi/bin/python compress_to_subgraphs.py \\
        --dataset cosmic >> /data/ross/ppi_lossgain/interaction_loss/cosmic/compress.log 2>&1 &
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA_ROOT  # noqa: E402


_BASE = DATA_ROOT
DATASET_CONFIGS = {
    "clinvar": {
        "h5_in":    _BASE / "clinvar" / "prott5_embeddings.h5",
        "graph_dir": _BASE / "clinvar" / "af3_graphs",
        "h5_out":   _BASE / "clinvar" / "prott5_subgraphs.h5",
    },
    "cosmic": {
        "h5_in":    _BASE / "cosmic" / "prott5_embeddings.h5",
        "graph_dir": _BASE / "cosmic" / "af3_graphs",
        "h5_out":   _BASE / "cosmic" / "prott5_subgraphs.h5",
    },
}


def _is_entry_complete(cgrp) -> bool:
    """Check a variant sub-group has all required datasets."""
    try:
        return "node_emb" in cgrp and "edge_index" in cgrp and "mut_diff" in cgrp
    except Exception:
        return False


def _get_2hop(G_csr, mut_idx: int, n_total: int):
    """Return sorted list of 2-hop neighbors (including mut_idx) using CSR slices."""
    hop1 = set(G_csr.indices[G_csr.indptr[mut_idx]:G_csr.indptr[mut_idx + 1]])
    hop1.add(mut_idx)
    hop2 = set(hop1)
    for nb in list(hop1):
        hop2.update(G_csr.indices[G_csr.indptr[nb]:G_csr.indptr[nb + 1]])
    return sorted(hop2)


def _extract_subgraph(G_csr, hop2_nodes: list, n_total: int):
    """Return edge_index (2×e int32) remapped to local node indices."""
    hop2_arr = np.array(hop2_nodes, dtype=np.int32)
    in_hop2 = np.zeros(n_total, dtype=bool)
    in_hop2[hop2_arr] = True

    # COO edges of full graph
    G_coo = G_csr.tocoo()
    mask = in_hop2[G_coo.row] & in_hop2[G_coo.col]
    sub_rows = G_coo.row[mask]
    sub_cols = G_coo.col[mask]

    local_map = np.full(n_total, -1, dtype=np.int32)
    local_map[hop2_arr] = np.arange(len(hop2_arr), dtype=np.int32)

    edge_index = np.stack([local_map[sub_rows], local_map[sub_cols]], axis=0).astype(np.int32)
    return edge_index


def compress(h5_in_path: str, graph_dir: str, h5_out_path: str) -> None:
    t0 = time.time()
    print(f"Input H5:  {h5_in_path}", flush=True)
    print(f"Graph dir: {graph_dir}", flush=True)
    print(f"Output H5: {h5_out_path}", flush=True)

    # ── build interactor → VT variant list from H5 keys ──────────────────────
    print("Scanning H5 keys ...", flush=True)
    with h5py.File(h5_in_path, "r") as f_in:
        all_keys = list(f_in.keys())

    inter_to_variants: dict[str, list[str]] = {}
    wt_key_set: set[str] = set()
    for k in all_keys:
        if " " in k:
            inter, variant = k.split(" ", 1)
            inter_to_variants.setdefault(inter, []).append(variant)
        else:
            wt_key_set.add(k)

    print(f"  {len(wt_key_set)} WT keys, "
          f"{sum(len(v) for v in inter_to_variants.values())} VT keys "
          f"across {len(inter_to_variants)} interactors", flush=True)

    # ── group mat files by interactor ─────────────────────────────────────────
    mat_files = sorted(glob.glob(str(Path(graph_dir) / "*.mat")))
    print(f"  {len(mat_files)} .mat files", flush=True)

    inter_to_mats: dict[str, list[str]] = {}
    for m in mat_files:
        stem = Path(m).stem
        inter = stem.split("_")[0]
        inter_to_mats.setdefault(inter, []).append(m)

    # ── open both H5 files ────────────────────────────────────────────────────
    f_in  = h5py.File(h5_in_path,  "r")
    f_out = h5py.File(h5_out_path, "a", libver="latest")

    # ── resume: count already-complete entries ────────────────────────────────
    n_existing = 0
    for cid in f_out.keys():
        cgrp = f_out[cid]
        for var in list(cgrp.keys()):
            vgrp = cgrp[var]
            if _is_entry_complete(vgrp):
                n_existing += 1
            else:
                # Clean up incomplete entry from a prior interrupted run
                del cgrp[var]
    print(f"  {n_existing} entries already complete — resuming", flush=True)

    n_written = 0
    n_skipped = 0
    n_missing_emb = 0
    n_bound_err = 0
    n_complexes = 0

    interactors_with_mats = sorted(set(inter_to_mats) & set(inter_to_variants))
    print(f"  {len(interactors_with_mats)} interactors with both mats and VT variants",
          flush=True)

    for inter_id in interactors_with_mats:
        # Load WT interactor embedding once per interactor
        if inter_id not in f_in:
            n_missing_emb += len(inter_to_variants[inter_id]) * len(inter_to_mats[inter_id])
            continue
        wt_inter = f_in[inter_id][:]  # (n_inter, 1024)

        variants = inter_to_variants[inter_id]

        # Batch-load all VT embeddings for this interactor once.
        # Each VT embedding is shared across all partners, so loading per-partner
        # would read each VT avg_partners times unnecessarily.
        vt_cache: dict[str, np.ndarray] = {}
        for variant in variants:
            vt_key = f"{inter_id} {variant}"
            if vt_key in f_in:
                vt_cache[variant] = f_in[vt_key][:]

        for mat_path in inter_to_mats[inter_id]:
            complex_id = Path(mat_path).stem
            parts = complex_id.split("_")
            partner_id = "_".join(parts[1:])

            if partner_id not in f_in:
                n_missing_emb += len(variants)
                continue

            wt_partner = f_in[partner_id][:]  # (n_partner, 1024)

            # Load contact graph
            mat = sio.loadmat(mat_path)
            n_inter = int(mat["NRR"].flat[0])
            G = mat["G"].tocsr()  # sparse COO → CSR for fast row slicing
            n_total = G.shape[0]

            # Validate dimensions
            if wt_inter.shape[0] != n_inter:
                continue
            if wt_partner.shape[0] != (n_total - n_inter):
                continue

            # Ensure output group exists
            if complex_id not in f_out:
                f_out.create_group(complex_id)
            cgrp = f_out[complex_id]

            for variant in variants:
                # Resume: skip if already complete
                if variant in cgrp and _is_entry_complete(cgrp[variant]):
                    n_skipped += 1
                    continue

                mut_idx = int(variant[1:-1])  # 0-based position in interactor

                if mut_idx >= n_inter:
                    n_bound_err += 1
                    continue

                vt_inter = vt_cache.get(variant)
                if vt_inter is None:
                    n_missing_emb += 1
                    continue

                if vt_inter.shape[0] != n_inter:
                    n_missing_emb += 1
                    continue

                # ── 2-hop BFS ─────────────────────────────────────────────────
                hop2_nodes = _get_2hop(G, mut_idx, n_total)
                hop2_arr   = np.array(hop2_nodes, dtype=np.int32)

                # ── node embeddings: VT for interactor, WT for partner ─────────
                node_emb = np.empty((len(hop2_arr), 1024), dtype=np.float32)
                inter_mask  = hop2_arr < n_inter
                partner_mask = ~inter_mask

                if inter_mask.any():
                    node_emb[inter_mask]  = vt_inter[hop2_arr[inter_mask]]
                if partner_mask.any():
                    node_emb[partner_mask] = wt_partner[hop2_arr[partner_mask] - n_inter]

                # ── subgraph edge_index ────────────────────────────────────────
                edge_index = _extract_subgraph(G, hop2_nodes, n_total)

                # ── mutation diff ─────────────────────────────────────────────
                mut_diff = (vt_inter[mut_idx] - wt_inter[mut_idx]).astype(np.float32)

                # ── local mutation index ───────────────────────────────────────
                local_map_small = {g: l for l, g in enumerate(hop2_nodes)}
                mut_local_idx = local_map_small[mut_idx]

                # ── write (overwrite any partial entry) ───────────────────────
                if variant in cgrp:
                    del cgrp[variant]
                vgrp = cgrp.create_group(variant)
                vgrp.create_dataset("node_emb",   data=node_emb,   dtype=np.float32)
                vgrp.create_dataset("edge_index",  data=edge_index, dtype=np.int32)
                vgrp.create_dataset("mut_diff",    data=mut_diff,   dtype=np.float32)
                vgrp.attrs["mut_local_idx"] = mut_local_idx

                n_written += 1

            n_complexes += 1
            if n_complexes % 500 == 0:
                elapsed = time.time() - t0
                total_done = n_existing + n_written
                print(f"  [{n_complexes}/{len(mat_files)}] "
                      f"written={n_written}  skipped={n_skipped}  "
                      f"missing={n_missing_emb}  "
                      f"elapsed={elapsed/60:.1f}m", flush=True)
                f_out.flush()

        # Free per-interactor memory before moving on
        del wt_inter, vt_cache

    f_in.close()
    f_out.flush()
    f_out.close()

    elapsed = time.time() - t0
    total = n_existing + n_written
    print(f"\nDone in {elapsed/60:.1f} min.", flush=True)
    print(f"  Written:        {n_written}", flush=True)
    print(f"  Already done:   {n_skipped}", flush=True)
    print(f"  Missing emb:    {n_missing_emb}", flush=True)
    print(f"  Bound errors:   {n_bound_err}", flush=True)
    print(f"  Total entries:  {total}", flush=True)
    print(f"  Output:         {h5_out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Compress ProtT5 H5 to 2-hop subgraph H5")
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS),
                   help="Named dataset (sets all paths)")
    p.add_argument("--h5-in",    help="Input full-embedding H5")
    p.add_argument("--graph-dir", help="Directory containing .mat files")
    p.add_argument("--h5-out",   help="Output subgraph H5 path")
    args = p.parse_args()

    if args.dataset:
        cfg = DATASET_CONFIGS[args.dataset]
        h5_in     = str(args.h5_in    or cfg["h5_in"])
        graph_dir = str(args.graph_dir or cfg["graph_dir"])
        h5_out    = str(args.h5_out   or cfg["h5_out"])
    else:
        if not (args.h5_in and args.graph_dir and args.h5_out):
            p.error("Provide --dataset or all three of --h5-in, --graph-dir, --h5-out")
        h5_in     = args.h5_in
        graph_dir = args.graph_dir
        h5_out    = args.h5_out

    compress(h5_in, graph_dir, h5_out)


if __name__ == "__main__":
    main()
