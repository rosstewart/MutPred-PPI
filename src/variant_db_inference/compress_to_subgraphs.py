#!/usr/bin/env python
"""Convert full-protein ProtT5 H5 embeddings to compact per-variant 2-hop subgraph H5.

With 2 GAT layers, prediction at the mutation site is a pure function of its
2-hop neighborhood.  Storing only those nodes is lossless and reduces ClinVar
from ~1.1 TB to ~110 GB, COSMIC from ~1.7 TB to ~200 GB.

Rows come from `datasets/variant_dbs/{db}_rows.csv.gz` and graphs from the
content-addressed store, keyed on the two chain SEQUENCES.  The previous version
globbed `{db}/af3_graphs/*.mat`, took the first `_`-separated token of the stem
as the interactor and sliced the interactor block at the stored `NRR`; for 121
clinvar and 121 cosmic files the stored chain order contradicts the filename, so
those complexes were compressed around the wrong chain.  The store returns the
graph already oriented to the requested interactor, so there is no NRR and no
reversal handling here.

Output HDF5 structure (unchanged):
  /{interactor}_{partner}/{variant}/     — `variant` is 0-BASED, as before
      node_emb    (k, 1024) float32  — subgraph node features
                                       (VT interactor emb for interactor nodes,
                                        WT partner emb for partner nodes)
      edge_index  (2, e)   int32     — COO edges in local subgraph coords
      mut_diff    (1024,)  float32   — vt_inter[mut_idx] - wt_inter[mut_idx]
      attrs:
        mut_local_idx  int  — mutation site index within local node list

Safety: incomplete entries from an interrupted run are deleted on resume, so a
partial write never blocks re-processing.

Usage:
    # Process ClinVar first (fits in available disk); verify, delete old, then COSMIC.
    nohup python compress_to_subgraphs.py \\
        --dataset clinvar >> $MUTPRED_DATA_ROOT/clinvar/compress.log 2>&1 &

    # After verifying and deleting old clinvar prott5_embeddings.h5:
    nohup python compress_to_subgraphs.py \\
        --dataset cosmic >> $MUTPRED_DATA_ROOT/cosmic/compress.log 2>&1 &
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from contact_graphs import ContactGraphStore, check_embedding_lengths  # noqa: E402
from paths import DATA_ROOT, DATASETS_DIR  # noqa: E402
from variant_db_inference import variant_rows as vr  # noqa: E402
from utils import mutations  # noqa: E402


_BASE = DATA_ROOT
STORE = DATASETS_DIR / "variant_dbs" / "contact_graphs.h5"
# All six databases, derived from the inference runner so the paths cannot drift
# apart. Only ClinVar and COSMIC were listed here before -- gnomAD, HGMD, NDD and
# ASD were compressed by hand or not at all, which is why three of them still
# hold full-length embeddings instead of subgraphs.
from variant_db_inference.run_variant_db_inference import (  # noqa: E402
    DATASET_CONFIGS as _VDB_CONFIGS)

DATASET_CONFIGS = {
    db: {"h5_in": cfg["default_emb_h5"], "h5_out": cfg["default_subgraph_h5"]}
    for db, cfg in _VDB_CONFIGS.items()
}


def _is_entry_complete(cgrp) -> bool:
    """Check a variant sub-group has all required datasets."""
    try:
        return "node_emb" in cgrp and "edge_index" in cgrp and "mut_diff" in cgrp
    except Exception:
        return False


def _count_reason(stats: Counter, examples: dict, reason: str, n: int = 1) -> None:
    """Bucket a `check_embedding_lengths` reason, keeping one full example.

    The reason carries the offending lengths, which would give the Counter a key
    per complex; the text before them is the class of failure.
    """
    bucket = reason.split(" has ", 1)[0]
    stats[f"{bucket} length disagrees with its sequence"] += n
    examples.setdefault(bucket, reason)


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


def _group_rows(db: str, rows_path, stats: Counter):
    """{interactor: {partner: sorted mutations}}, plus {(i, p): (iseq, pseq)}."""
    grouped: dict[str, dict[str, set[str]]] = {}
    pair_seqs: dict[tuple[str, str], tuple[str, str]] = {}
    n = 0
    for r in vr.iter_table_rows(db, rows_path, stats=stats):
        i, p = r["interactor"], r["partner"]
        grouped.setdefault(i, {}).setdefault(p, set()).add(r["mutation"])
        pair_seqs.setdefault((i, p), (r["interactor_sequence"], r["partner_sequence"]))
        n += 1
    stats["rows read"] = n
    return grouped, pair_seqs


def compress(db: str, h5_in_path: str, h5_out_path: str, rows_path: str,
             store_path: str) -> None:
    t0 = time.time()
    print(f"Input H5:  {h5_in_path}", flush=True)
    print(f"Rows:      {rows_path}", flush=True)
    print(f"Graphs:    {store_path}", flush=True)
    print(f"Output H5: {h5_out_path}", flush=True)

    stats: Counter = Counter()
    examples: dict[str, str] = {}

    print("Reading rows ...", flush=True)
    grouped, pair_seqs = _group_rows(db, rows_path, stats)
    n_pairs = sum(len(v) for v in grouped.values())
    print(f"  {stats['rows read']:,} rows over {n_pairs:,} pairs "
          f"and {len(grouped):,} interactors", flush=True)

    store = ContactGraphStore(store_path)
    f_in = h5py.File(h5_in_path, "r")
    f_out = h5py.File(h5_out_path, "a", libver="latest")

    # ── resume: count already-complete entries ────────────────────────────────
    n_existing = 0
    for cid in f_out.keys():
        cgrp = f_out[cid]
        for var in list(cgrp.keys()):
            if _is_entry_complete(cgrp[var]):
                n_existing += 1
            else:
                # Clean up incomplete entry from a prior interrupted run
                del cgrp[var]
    print(f"  {n_existing} entries already complete — resuming", flush=True)

    n_written = 0
    n_skipped = 0
    n_pairs_done = 0

    for inter_id in sorted(grouped):
        partners = grouped[inter_id]
        n_rows = sum(len(m) for m in partners.values())

        if inter_id not in f_in:
            stats["interactor WT embedding absent from the input H5"] += n_rows
            continue
        wt_inter = f_in[inter_id][:]  # (n_inter, 1024)

        # Batch-load all VT embeddings for this interactor once.
        # Each VT embedding is shared across all partners, so loading per-partner
        # would read each VT avg_partners times unnecessarily.
        vt_cache: dict[str, np.ndarray | None] = {}
        for muts in partners.values():
            for mut in muts:
                variant = vr.to_zero_based(mut)   # H5 keys are 0-based
                if variant not in vt_cache:
                    key = f"{inter_id} {variant}"
                    vt_cache[variant] = f_in[key][:] if key in f_in else None

        for partner_id, muts in sorted(partners.items()):
            complex_id = f"{inter_id}_{partner_id}"
            iseq, pseq = pair_seqs[(inter_id, partner_id)]
            n_inter, n_total = len(iseq), len(iseq) + len(pseq)
            n_pairs_done += 1

            if partner_id not in f_in:
                stats["partner WT embedding absent from the input H5"] += len(muts)
                continue
            wt_partner = f_in[partner_id][:]  # (n_partner, 1024)

            reason = check_embedding_lengths(
                interactor_seq=iseq, partner_seq=pseq,
                interactor_emb=wt_inter, partner_emb=wt_partner)
            if reason:
                _count_reason(stats, examples, reason, len(muts))
                continue

            ei = store.load_edge_index(interactor=iseq, partner=pseq)
            if ei is None:
                stats["pair has no graph in the contact-graph store"] += len(muts)
                continue
            G = sp.csr_matrix(
                (np.ones(ei.shape[1], dtype=np.int8), (ei[0], ei[1])),
                shape=(n_total, n_total))

            # Ensure output group exists
            if complex_id not in f_out:
                f_out.create_group(complex_id)
            cgrp = f_out[complex_id]

            for mut in sorted(muts):
                variant = vr.to_zero_based(mut)
                # Resume: skip if already complete
                if variant in cgrp and _is_entry_complete(cgrp[variant]):
                    n_skipped += 1
                    continue

                mut_idx = mutations.position(variant)   # H5 keys are 0-based
                if mut_idx >= n_inter:
                    stats["mutation position past the end of the interactor"] += 1
                    continue

                vt_inter = vt_cache.get(variant)
                if vt_inter is None:
                    stats["variant embedding absent from the input H5"] += 1
                    continue
                reason = check_embedding_lengths(
                    interactor_seq=iseq, partner_seq=pseq, interactor_emb=vt_inter)
                if reason:
                    _count_reason(stats, examples, reason)
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

            if n_pairs_done % 500 == 0:
                elapsed = time.time() - t0
                print(f"  [{n_pairs_done}/{n_pairs}] "
                      f"written={n_written}  skipped={n_skipped}  "
                      f"dropped={sum(v for k, v in stats.items() if k != 'rows read')}  "
                      f"elapsed={elapsed/60:.1f}m", flush=True)
                f_out.flush()

        # Free per-interactor memory before moving on
        del wt_inter, vt_cache

    f_in.close()
    f_out.flush()
    f_out.close()
    store.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.", flush=True)
    print(f"  Written:        {n_written}", flush=True)
    print(f"  Already done:   {n_skipped}", flush=True)
    print(f"  Total entries:  {n_existing + n_written}", flush=True)
    print("  Rows not compressed, by reason:", flush=True)
    for k, v in stats.most_common():
        if k != "rows read":
            print(f"    {k}: {v:,}", flush=True)
    for ex in examples.values():
        print(f"    e.g. {ex}", flush=True)
    print(f"  Output:         {h5_out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Compress ProtT5 H5 to 2-hop subgraph H5")
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS),
                   help="Named dataset (sets all paths)")
    p.add_argument("--h5-in",    help="Input full-embedding H5")
    p.add_argument("--h5-out",   help="Output subgraph H5 path")
    p.add_argument("--rows",     help="Row table (default: datasets/variant_dbs/{db}_rows.csv.gz)")
    p.add_argument("--store", default=str(STORE),
                   help=f"Contact-graph HDF5 store (default: {STORE})")
    p.add_argument("--graph-dir", help=argparse.SUPPRESS)  # accepted, ignored
    args = p.parse_args()

    if args.graph_dir:
        print("[note] --graph-dir is ignored: graphs now come from --store, "
              "addressed by sequence", flush=True)

    # --dataset is now required: rows and WT sequences are per-database, so the
    # three explicit paths no longer describe a run on their own.
    if not args.dataset:
        p.error("--dataset is required (--h5-in/--h5-out/--rows still override paths)")
    cfg = DATASET_CONFIGS[args.dataset]
    db     = args.dataset
    h5_in  = str(args.h5_in  or cfg["h5_in"])
    h5_out = str(args.h5_out or cfg["h5_out"])

    rows = str(args.rows or vr.table_path(db))
    if not Path(rows).exists():
        p.error(f"{rows} not found — run build_variant_db_tables.py --db {db}")

    compress(db, h5_in, h5_out, rows, args.store)


if __name__ == "__main__":
    main()
