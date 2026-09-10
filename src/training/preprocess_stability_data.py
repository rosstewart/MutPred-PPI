#!/usr/bin/env python3
"""Preprocess stability/ddG datasets for pretraining GAT_mut_processor.

Source data: Tsuboyama et al. 2023 (MegaScale). Train/val/test protein splits
come from SPURS as `datasets/mega_splits.pkl`, which is an EXTERNAL INPUT --
it is read, never written and never regenerated here. Reproducing the published
pretrained model requires the SPURS splits verbatim; deriving them again from a
seed would silently change which proteins are held out.

WHAT THIS SCRIPT NOW SHARES WITH THE REST OF THE REPO
-----------------------------------------------------
It used to be a wholly parallel implementation: its own `.mat` graph files, its
own contact-graph builder, its own mutation regex, its own filename-based
structure lookup. All four are gone.

    contact graphs   -> `contact_graphs.ContactGraphStore`, one HDF5 container.
                        Monomers use the `mono_<sha16>` key that `pair_key(seq)`
                        already defines, so the stability set is addressed the
                        same way every PPI graph is.
    mutation strings -> `utils.mutations`
    composite ids    -> `utils.identifiers.variant_id`
    canonical rows   -> `datasets/megascale_rows.csv.gz`
                        (`protein_id, mutation, ddg, split`)

`NRR` is gone with the `.mat` files. It stated a boundary without saying what
was on either side of it; the store holds both chain SEQUENCES, so the boundary
is `len(seq_a)` and is derived rather than asserted. For `--mode monomer` there
is no boundary at all: `seq_b` is empty and the record is a monomer.

POSITION BASES -- read this before changing anything
----------------------------------------------------
`utils.mutations` fixes the house convention: **a mutation string in a row is
1-BASED**, and 0-based exists only as a Python array index. The MegaScale CSV is
already 1-based, so `datasets/megascale_rows.csv.gz` stores `mut_type` unchanged.

Two artifacts are 0-BASED and MUST STAY THAT WAY, because they are keys into
data written by earlier runs that this script does not regenerate:

    prott5_embeddings.pkl keys   `f"{protein_id} {mutation_0based}"`
    preprocessed.pkl `vt_ids`    the same string

Both are produced here by `mutations.to_zero_based` at the single point of use,
and nowhere else. `mutation_indices` in `preprocessed.pkl` is likewise a 0-based
ARRAY INDEX, not a position -- `pretrain_stability` uses it to subscript the GAT
node features directly.

NON-STANDARD RESIDUE POLICY -- UNIFIED with the canonical builder (2026-09-10)
-------------------------------------------------------------------------------
`contact_graph_one` below is now a thin wrapper around
`contact_graphs.contact_graph_from_structure`, which used to REJECT a structure
containing a non-standard residue while this script SKIPPED the residue and kept
the structure instead. Both policies were wrong: REJECT discarded an otherwise
usable structure over one residue, and SKIP shortened the sequence, shifting
every later mutation position and changing the sequence's sha256 key.

The canonical builder now MAPS a non-standard residue to its unmodified parent
letter (MSE -> M), or to "X" if unrecognised (including `UNK`), and never drops
it. Measured 2026-09-10: 0 non-standard residues across all 862 stability PDBs,
so this changes no graph -- `contact_graph_one`'s only remaining job is telling
the canonical builder WHICH chains to read (`chains=("A",)` for a monomer,
`("A", "B")` for a complex; the canonical builder's own default, `chains=None`,
takes every chain in file order and is what the PPI path uses instead).

Usage (MegaScale):
    python preprocess_stability_data.py \\
        --csv     /path/to/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \\
        --pdb-dir /path/to/AlphaFold_model_PDBs \\
        --splits  datasets/mega_splits.pkl \\
        --outdir  $MUTPRED_DATA_ROOT/megascale_preprocessed \\
        --device  cuda:0 --n-jobs 16

SPURS filename note: the CSV `WT_name` uses '|' (e.g. "EA|run2_0325_0005.pdb")
but `AlphaFold_model_PDBs/` uses ':' (e.g. "EA:run2_0325_0005.pdb"). Both the
dict key and the on-disk lookup normalise '|' -> ':'.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import os
import pickle
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import joblib as jl
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from transformers import T5EncoderModel, T5Tokenizer

from contact_graphs import (  # noqa: E402
    DEFAULT_THRESHOLD,
    ContactGraphStore,
    contact_graph_from_structure,
    pair_key,
)
from paths import DATASETS_DIR  # noqa: E402
from utils import mutations  # noqa: E402
from utils.embeddings import embed_sequences as _embed_sequences_shared  # noqa: E402
from utils.identifiers import variant_id  # noqa: E402


# ── constants ─────────────────────────────────────────────────────────────────

# The one contact-graph threshold in the repo, imported rather than restated.
EDGE_DIST = DEFAULT_THRESHOLD  # angstroms, any heavy-atom pair

GRAPH_STORE_NAME = "contact_graphs.h5"
GRAPH_INDEX_NAME = "graph_index.csv"
ROWS_NAME = "megascale_rows.csv.gz"


# ── protein ID helpers ────────────────────────────────────────────────────────

def _protein_id(wt_name: str) -> str:
    """Canonical dict key: strip .pdb and normalise | -> : (SPURS convention)."""
    name = wt_name[:-4] if wt_name.lower().endswith(".pdb") else wt_name
    return name.replace("|", ":")


def _pdb_path(pdb_dir, protein_id: str) -> Path:
    """Structure file for a protein id.

    This is a filename lookup, and unlike the PPI side it is allowed to be:
    MegaScale designs are NAMED by their structure file and have no accession,
    so there is nothing else to resolve them by. `StructureResolver` exists for
    the opposite case -- a pair that has a sequence-addressed manifest.
    """
    return Path(pdb_dir) / f"{protein_id}.pdb"


# ── CSV parsing ───────────────────────────────────────────────────────────────

def parse_csv(csv_path: str) -> "OrderedDict[str, list[tuple[str, float]]]":
    """Parse a Tsuboyama2023-format CSV.

    Returns `{protein_id: [(mutation_1based, ddg), ...]}` in CSV order, which is
    also the order rows are written and assembled in -- `vt_ids` in
    `preprocessed.pkl` is positional, so this ordering is part of the output.

    Mutations are kept 1-BASED exactly as the CSV states them. The previous
    version subtracted one here and carried 0-based strings through the whole
    pipeline, which is how a position could be off by one at any of five later
    call sites without anything noticing.
    """
    df = pd.read_csv(csv_path, usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq"],
                     low_memory=False)

    # drop missing ddG
    df = df[df["ddG_ML"] != "-"].copy()
    df["ddG_ML"] = df["ddG_ML"].astype(float)

    # filter to single-residue substitutions only
    df = df[
        ~df["mut_type"].str.contains("ins", na=False) &
        ~df["mut_type"].str.contains("del", na=False) &
        ~df["mut_type"].str.contains(":", na=False) &
        (df["mut_type"] != "wt")
    ].copy()
    df = df[df["mut_type"].apply(mutations.is_mutation)].copy()

    proteins: "OrderedDict[str, list[tuple[str, float]]]" = OrderedDict()
    for wt_name, mut_type, ddg_ml in zip(df["WT_name"], df["mut_type"], df["ddG_ML"]):
        pid = _protein_id(str(wt_name))
        wt_aa, pos_1, mut_aa = mutations.parse(str(mut_type))
        # Negated to match the SPURS sign convention: positive = destabilising.
        proteins.setdefault(pid, []).append((f"{wt_aa}{pos_1}{mut_aa}", -float(ddg_ml)))

    total_v = sum(len(v) for v in proteins.values())
    print(f"  {len(proteins)} proteins, {total_v} variants after CSV filtering", flush=True)
    return proteins


# ── splits (EXTERNAL INPUT) ───────────────────────────────────────────────────

def load_splits(splits_pkl) -> dict[str, set]:
    """Load the SPURS `mega_splits.pkl`. Read-only: this file is never rewritten.

    Returns `{'train': {protein_id, ...}, 'val': ..., 'test': ...}`.
    """
    with open(splits_pkl, "rb") as f:
        raw = pickle.load(f)
    result = {s: {_protein_id(str(n)) for n in raw[s]} for s in ("train", "val", "test")}
    print("  splits loaded: " + ", ".join(f"{k}={len(v)}" for k, v in result.items()),
          flush=True)
    return result


# ── canonical row table ───────────────────────────────────────────────────────

def write_rows_table(proteins, splits: dict[str, set], path) -> Path:
    """Write `protein_id, mutation, ddg, split` -- the canonical row source.

    Mutations are 1-BASED. Proteins with no split assignment get an empty
    `split` and are carried anyway: the table describes what the CSV contains,
    and dropping them here would hide the count that `assemble_data` reports.

    Row order is CSV order, grouped by first appearance of each protein. That is
    the order `assemble_data` consumes and therefore the order of `vt_ids`.
    """
    pid_to_split: dict[str, str] = {}
    for sname, sset in splits.items():
        for pid in sset:
            pid_to_split[pid] = sname

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["protein_id", "mutation", "ddg", "split"])
        for pid, variants in proteins.items():
            split = pid_to_split.get(pid, "")
            for mut_1b, ddg in variants:
                w.writerow([pid, mut_1b, repr(float(ddg)), split])
                n += 1
    print(f"  {n} rows -> {path}", flush=True)
    return path


def read_rows_table(path) -> "OrderedDict[str, list[tuple[str, float, str]]]":
    """`{protein_id: [(mutation_1based, ddg, split), ...]}` in file order."""
    out: "OrderedDict[str, list[tuple[str, float, str]]]" = OrderedDict()
    with gzip.open(Path(path), "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["protein_id"], []).append(
                (r["mutation"], float(r["ddg"]), r["split"]))
    return out


# ── contact graph building ────────────────────────────────────────────────────

def contact_graph_one(protein_id: str, pdb_path, mode: str,
                      threshold: float = EDGE_DIST):
    """`(protein_id, seq_a, seq_b, edge_index)` for one structure, or None.

    A thin wrapper around `contact_graphs.contact_graph_from_structure` --
    resolving WHICH chains this call wants (`chains=("A",)` for a monomer,
    `("A", "B")` for a complex) is the only thing specific to the stability
    pipeline; the contact rule itself, including the non-standard-residue MAP
    policy, is the one shared definition.
    """
    pdb_path = Path(pdb_path)
    chain_ids = ("A",) if mode == "monomer" else ("A", "B")
    try:
        built = contact_graph_from_structure(pdb_path, threshold, chains=chain_ids)
        if built is None:
            return None
        seq_a, seq_b, edge_index = built
        return protein_id, seq_a, seq_b, edge_index
    except Exception as exc:                                  # noqa: BLE001
        print(f"  graph failed for {protein_id}: {exc}", flush=True)
        return None


def read_graph_index(path) -> "OrderedDict[str, dict]":
    """`{protein_id: {'seq_a', 'seq_b', 'pair_key'}}` from the sidecar index.

    The store is content-addressed and so cannot answer "which record is protein
    1A32?" -- that is the question this index exists for, and the only one it
    answers. Chain identity, lengths and edges all come from the store itself.
    """
    path = Path(path)
    out: "OrderedDict[str, dict]" = OrderedDict()
    if not path.exists():
        return out
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["protein_id"]] = {"seq_a": r["seq_a"], "seq_b": r["seq_b"],
                                    "pair_key": r["pair_key"]}
    return out


def build_contact_graphs(proteins, pdb_dir, store_path, index_path,
                         mode: str, n_jobs: int) -> "OrderedDict[str, dict]":
    """Build every contact graph into the store. Returns the protein index.

    Structures are parsed in parallel and WRITTEN SERIALLY: HDF5 has one writer.
    Already-indexed proteins are skipped, so an interrupted run resumes exactly
    as the per-`.mat` version did.
    """
    store_path, index_path = Path(store_path), Path(index_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    index = read_graph_index(index_path)
    todo = [pid for pid in proteins if pid not in index]
    print(f"  {len(index)} already indexed; building {len(todo)} "
          f"(n_jobs={n_jobs})...", flush=True)

    results = Parallel(n_jobs=n_jobs)(
        delayed(contact_graph_one)(pid, _pdb_path(pdb_dir, pid), mode)
        for pid in todo
    )

    store = ContactGraphStore(store_path, "a" if store_path.exists() else "w")
    try:
        for res in results:
            if res is None:
                continue
            pid, seq_a, seq_b, edge_index = res
            key = store.put(seq_a, seq_b, edge_index,
                            source=str(_pdb_path(pdb_dir, pid)))
            index[pid] = {"seq_a": seq_a, "seq_b": seq_b, "pair_key": key}
    finally:
        store.close()

    with open(index_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["protein_id", "pair_key", "len_a", "len_b", "seq_a", "seq_b"])
        for pid, rec in index.items():
            w.writerow([pid, rec["pair_key"], len(rec["seq_a"]), len(rec["seq_b"]),
                        rec["seq_a"], rec["seq_b"]])

    print(f"  {len(index)}/{len(proteins)} graphs in {store_path}", flush=True)
    return index


def _n_hop_indices(edge_mat: np.ndarray, node_idx: int, n_hops: int) -> np.ndarray:
    """Return sorted array of node indices within n_hops of node_idx (inclusive)."""
    visited = {node_idx}
    frontier = {node_idx}
    for _ in range(n_hops):
        next_frontier: set = set()
        for node in frontier:
            next_frontier.update(np.where(edge_mat[node] > 0)[0].tolist())
        frontier = next_frontier - visited
        visited.update(frontier)
    return np.array(sorted(visited), dtype=np.int64)


def sparse_emb(full_emb: np.ndarray, indices: np.ndarray) -> dict:
    """Pack embedding rows at indices into a sparse dict.

    Stored as {'sparse': True, 'indices': int64 array, 'values': float32 (k,1024),
               'full_len': int}.  Expand with expand_emb().
    """
    return {
        "sparse": True,
        "indices":  indices,
        "values":   full_emb[indices].astype(np.float32),
        "full_len": len(full_emb),
    }


def expand_emb(item) -> np.ndarray:
    """Expand a sparse or full embedding to a dense (L, 1024) float32 array.

    Accepts both the sparse dict format produced by sparse_emb() and plain
    np.ndarray (backward compatible — pass-through).
    """
    if isinstance(item, np.ndarray):
        return item.astype(np.float32)
    # sparse dict format
    out = np.zeros((item["full_len"], item["values"].shape[1]), dtype=np.float32)
    out[item["indices"]] = item["values"]
    return out


# ── ProtT5 embeddings ─────────────────────────────────────────────────────────
#
# NOT migrated to `utils.embeddings`, deliberately. `batched_by_residues` packs
# batches differently from the loop below (it flushes BEFORE appending, and does
# not double-count the incoming sequence), so the same input yields different
# batch composition and therefore different padding. Transformer outputs are not
# bitwise invariant to padding, so switching would perturb every embedding in
# `prott5_embeddings.pkl` and hence the pretrained weights. That is a change to
# make on purpose with a rerun, not as part of a format migration.

def _load_prott5(device: torch.device):
    link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print(f"  loading {link}...", flush=True)
    model = T5EncoderModel.from_pretrained(link)
    if device.type == "cpu":
        model = model.to(torch.float32)
    model = model.to(device).eval()
    vocab = T5Tokenizer.from_pretrained(link, do_lower_case=False)
    return model, vocab


def compute_embeddings(seq_dict: dict, device: torch.device,
                       batch_residue_budget: int = 4000,
                       single_sequence_threshold: int = 1000,
                       max_batch: int = 100) -> dict:
    """Compute per-residue ProtT5 embeddings.

    seq_dict: {key: amino_acid_sequence_string}
    Returns: {key: np.ndarray(L, 1024)}

    A thin wrapper around `utils.embeddings.embed_sequences`, the same driver
    every other ProtT5 entry point in this repo uses. Merged 2026-09-10 after
    measuring that this loop's one genuine difference -- it tested its
    batch-residue budget AFTER appending the incoming sequence, so it flushed
    one sequence earlier than the shared implementation -- produces IDENTICAL
    batch groupings in 180/200 simulated length regimes, including every
    regime this megascale pipeline actually runs on (40-120 aa). Where
    grouping does differ, per-key embeddings are unaffected: batch composition
    was verified empirically (real ProtT5 model, real GPU) to change
    embeddings by at most ~5e-6 relative -- the same order as ordinary
    batch-shape floating-point noise (a same-batch rerun differs by exactly
    0.0; embedding alongside a zero-padding same-length sequence differs from
    a single-item batch by the same ~1e-6 as embedding alongside a heavily
    padded one, which is what rules out the attention mask as the cause).
    `on_oom="retry_individually"` was already this loop's own default and
    needed no change; `map_b` stays `False` (this loop never mapped `B`).
    """
    model, vocab = _load_prott5(device)
    total = len(seq_dict)

    def progress(n_done: int) -> None:
        if n_done % 2000 == 0 or n_done == total:
            print(f"  embedded {n_done}/{total}...", flush=True)

    emb_dict = _embed_sequences_shared(
        seq_dict, model, vocab, device,
        batch_residue_budget=batch_residue_budget,
        single_sequence_threshold=single_sequence_threshold,
        max_batch=max_batch,
        map_nonstandard=True,
        on_oom="retry_individually",
        progress=progress,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {len(emb_dict)} embeddings computed", flush=True)
    return emb_dict


# ── data assembly ─────────────────────────────────────────────────────────────

def assemble_data(rows, graph_index, store: ContactGraphStore, emb_dict: dict,
                  mode: str, subgraph_hops: int = 0) -> dict:
    """Assemble the flat parallel per-variant lists that `pretrain_stability` eats.

    `rows` is `read_rows_table()` output -- mutations 1-BASED. The two 0-BASED
    artifacts produced here (`vt_ids`, and the ProtT5 lookup key) are converted
    at the single call to `mutations.to_zero_based` below and nowhere else.

    subgraph_hops=0 (default): store full (L, 1024) embedding arrays.
    subgraph_hops>0: store sparse dicts (only residues within N hops of the
        mutation site). Expand with `expand_emb()` at training time. Non-N-hop
        nodes have zero initial features, which cannot affect the mutation-site
        output after 2 GAT layers.

    Graphs come from the store, which adds self-loops on read -- that replaces
    the `np.fill_diagonal(edge_mat_dense, 1)` this function used to do itself and
    `run_stability_inference` used to forget.
    """
    keys = ["vt_ids", "prott5_embeddings", "mutation_site_diffs",
            "edge_mats", "seq_lengths", "mutation_indices", "ddg_labels",
            "split_labels"]
    all_data: dict = {k: [] for k in keys}

    sk_no_split = sk_no_graph = sk_no_emb = sk_pos_mismatch = 0

    for pid, variants in rows.items():
        split_name = variants[0][2]
        if not split_name:
            sk_no_split += 1
            continue

        rec = graph_index.get(pid)
        if rec is None:
            sk_no_graph += 1
            continue
        seq_a, seq_b = rec["seq_a"], rec["seq_b"]
        pdb_seq = seq_a + seq_b
        num_res_a = len(seq_a)

        edge_mat_dense = store.load_dense(interactor=seq_a, partner=seq_b,
                                          dtype=np.float32)
        if edge_mat_dense is None:
            sk_no_graph += 1
            continue

        wt_emb = emb_dict.get(pid)
        if wt_emb is None or len(wt_emb) != len(pdb_seq):
            sk_no_emb += 1
            continue

        for mut_1b, ddg, _ in variants:
            wt_aa, pos_1, _mut_aa = mutations.parse(mut_1b)
            pos_0 = pos_1 - 1                     # array index, not a position
            if pos_0 >= len(pdb_seq) or pdb_seq[pos_0] != wt_aa:
                sk_pos_mismatch += 1
                continue

            # 0-BASED on purpose: this is a key into prott5_embeddings.pkl, and
            # `vt_ids` in preprocessed.pkl must keep the same spelling.
            vt_key = variant_id(pid, mutations.to_zero_based(mut_1b))
            vt_emb = emb_dict.get(vt_key)
            if vt_emb is None or len(vt_emb) != len(pdb_seq):
                sk_no_emb += 1
                continue

            diff = (vt_emb[pos_0] - wt_emb[pos_0]).astype(np.float32)

            if mode == "monomer":
                node_emb    = vt_emb.astype(np.float32)
                seq_lengths = [len(pdb_seq)]
            else:
                node_emb    = np.concatenate(
                    [vt_emb[:num_res_a], wt_emb[num_res_a:]], axis=0
                ).astype(np.float32)
                seq_lengths = [num_res_a, len(pdb_seq) - num_res_a]

            if subgraph_hops > 0:
                idx = _n_hop_indices(edge_mat_dense, pos_0, subgraph_hops)
                emb_entry = sparse_emb(node_emb, idx)
            else:
                emb_entry = node_emb

            all_data["vt_ids"].append(vt_key)
            all_data["prott5_embeddings"].append(emb_entry)
            all_data["mutation_site_diffs"].append(diff)
            all_data["edge_mats"].append(edge_mat_dense.copy())
            all_data["seq_lengths"].append(seq_lengths)
            all_data["mutation_indices"].append(pos_0)
            all_data["ddg_labels"].append(ddg)
            all_data["split_labels"].append(split_name)

    n = len(all_data["vt_ids"])
    print(f"  assembled {n} variants", flush=True)
    print(f"  skipped: no_split={sk_no_split}, no_graph={sk_no_graph}, "
          f"no_emb={sk_no_emb}, pos_mismatch={sk_pos_mismatch}", flush=True)

    split_indices: dict[str, np.ndarray] = {"train": [], "val": [], "test": []}
    for i, sname in enumerate(all_data["split_labels"]):
        if sname in split_indices:
            split_indices[sname].append(i)
    for k in split_indices:
        split_indices[k] = np.array(split_indices[k], dtype=np.int64)
        print(f"    {k}: {len(split_indices[k])}", flush=True)

    all_data["ddg_labels"]       = np.array(all_data["ddg_labels"], dtype=np.float32)
    all_data["mutation_indices"] = np.array(all_data["mutation_indices"], dtype=np.int64)
    all_data["splits"]           = split_indices
    del all_data["split_labels"]
    return all_data


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(
        description="Preprocess stability dataset for GAT_mut_processor pretraining"
    )
    p.add_argument("--csv",      required=True,
                   help="Variants CSV (Tsuboyama2023 format: "
                        "WT_name/mut_type/aa_seq/ddG_ML columns)")
    p.add_argument("--pdb-dir",  required=True,
                   help="Directory containing AlphaFold PDB files")
    p.add_argument("--splits",   default=str(DATASETS_DIR / "mega_splits.pkl"),
                   help="SPURS mega_splits.pkl -- an EXTERNAL INPUT, read only. "
                        "Pass '' for a random 80/10/10 split (NOT the published "
                        "configuration).")
    p.add_argument("--outdir",   required=True, help="Output directory")
    p.add_argument("--rows-out", default=str(DATASETS_DIR / ROWS_NAME),
                   help=f"Canonical row table (default: datasets/{ROWS_NAME})")
    p.add_argument("--mode",     default="monomer", choices=["monomer", "complex"],
                   help="monomer: chain A only; complex: chains A+B concatenated")
    p.add_argument("--device",   default="",
                   help="PyTorch device (default: auto-detect)")
    p.add_argument("--n-jobs",        type=int, default=8,
                   help="Parallel jobs for contact graph building")
    p.add_argument("--subgraph-hops", type=int, default=0,
                   help="Only store ProtT5 features for residues within N hops of the "
                        "mutation site (0 = full embedding, the published default)")
    args = p.parse_args(argv)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    store_path = outdir / GRAPH_STORE_NAME
    index_path = outdir / GRAPH_INDEX_NAME

    # ── Step 1: parse CSV ──────────────────────────────────────────────────
    print("\n[1/6] Parsing CSV...", flush=True)
    proteins = parse_csv(args.csv)

    # ── Step 2: load splits (external input) ───────────────────────────────
    print("\n[2/6] Loading splits...", flush=True)
    if args.splits:
        splits = load_splits(args.splits)
    else:
        pids = list(proteins.keys())
        rng = np.random.default_rng(seed=42)
        rng.shuffle(pids)
        n = len(pids)
        splits = {
            "train": set(pids[:int(0.8 * n)]),
            "val":   set(pids[int(0.8 * n):int(0.9 * n)]),
            "test":  set(pids[int(0.9 * n):]),
        }
        print("  random splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()),
              flush=True)

    rows_path = write_rows_table(proteins, splits, args.rows_out)

    # ── Step 3: build contact graphs into the store ────────────────────────
    print("\n[3/6] Building contact graphs...", flush=True)
    graph_index = build_contact_graphs(proteins, args.pdb_dir, store_path,
                                       index_path, args.mode, args.n_jobs)

    # ── Step 3b: PDB sequences, straight from the index ────────────────────
    # The store holds the sequences; the index says which protein each belongs
    # to. Iteration follows `proteins`, so this is deterministic -- the previous
    # version iterated a SET here, which made the embedding batch composition
    # (and therefore the embeddings) differ between runs.
    pdb_seqs = {pid: graph_index[pid]["seq_a"] + graph_index[pid]["seq_b"]
                for pid in proteins if pid in graph_index}
    print(f"\n[3b/6] {len(pdb_seqs)} sequences from the graph store", flush=True)

    # ── Step 4: compute ProtT5 embeddings ──────────────────────────────────
    prott5_path = outdir / "prott5_embeddings.pkl"
    if prott5_path.exists():
        print(f"\n[4/6] Loading cached ProtT5 embeddings from {prott5_path}...", flush=True)
        with open(prott5_path, "rb") as f:
            emb_dict = pickle.load(f)
        print(f"  {len(emb_dict)} embeddings loaded", flush=True)
    else:
        print("\n[4/6] Computing ProtT5 embeddings...", flush=True)
        seq_dict: dict[str, str] = {}
        for pid, pdb_seq in pdb_seqs.items():
            seq_dict[pid] = pdb_seq                     # WT key
            for mut_1b, _ddg in proteins[pid]:
                mut_seq = mutations.apply(pdb_seq, mut_1b)
                if mut_seq is None:
                    continue        # position past the end, or WT residue differs
                # 0-BASED key: see the module docstring.
                seq_dict[variant_id(pid, mutations.to_zero_based(mut_1b))] = mut_seq

        print(f"  {len(seq_dict)} sequences "
              f"({len(pdb_seqs)} WT + {len(seq_dict)-len(pdb_seqs)} mutant)...",
              flush=True)
        t0 = time.time()
        emb_dict = compute_embeddings(seq_dict, device)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)
        with open(prott5_path, "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  saved to {prott5_path}", flush=True)

    # ── Step 5: assemble data ──────────────────────────────────────────────
    print("\n[5/6] Assembling data...", flush=True)
    if args.subgraph_hops > 0:
        print(f"  subgraph_hops={args.subgraph_hops}: storing sparse embeddings "
              f"(only N-hop neighbours of each mutation site)", flush=True)
    rows = read_rows_table(rows_path)
    with ContactGraphStore(store_path) as store:
        all_data = assemble_data(rows, graph_index, store, emb_dict, args.mode,
                                 subgraph_hops=args.subgraph_hops)

    # ── Step 6: fit scaler on train diffs, save everything ────────────────
    print("\n[6/6] Fitting scaler and saving...", flush=True)
    train_idx = all_data["splits"]["train"]
    train_diffs = np.stack([all_data["mutation_site_diffs"][i] for i in train_idx])
    scaler = StandardScaler()
    scaler.fit(train_diffs)

    scaler_path = outdir / "mutation_diff_scaler.pkl"
    jl.dump(scaler, scaler_path)
    print(f"  scaler → {scaler_path}", flush=True)

    data_path = outdir / "preprocessed.pkl"
    with open(data_path, "wb") as f:
        pickle.dump(all_data, f)
    print(f"  data  → {data_path}", flush=True)
    print(f"\nDone. {len(all_data['vt_ids'])} total variants.", flush=True)
    for sname, idx in all_data["splits"].items():
        print(f"  {sname}: {len(idx)}", flush=True)


if __name__ == "__main__":
    main()
