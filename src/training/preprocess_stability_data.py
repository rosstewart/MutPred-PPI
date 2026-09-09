#!/usr/bin/env python3
"""Preprocess stability/ddG datasets for training GAT_mut_processor.

Handles both monomer (stability) and complex (PPI) datasets.
For MegaScale/SPURS: uses mega_splits.pkl for train/val/test assignment.

SPURS filename note: CSV WT_name uses '|' (e.g. "EA|run2_0325_0005.pdb") but
AlphaFold_model_PDBs/ uses ':' (e.g. "EA:run2_0325_0005.pdb").  The script
normalises by replacing '|' -> ':' in all PDB lookups.

Usage (MegaScale):
    python preprocess_stability_data.py \\
        --csv     /path/to/tools/SPURS/data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \\
        --pdb-dir /path/to/tools/SPURS/data/dataset/megascale/AlphaFold_model_PDBs \\
        --splits  /path/to/tools/SPURS/data/dataset/megascale/mega_splits.pkl \\
        --outdir  $MUTPRED_DATA_ROOT/megascale_preprocessed \\
        --device  cuda:0 --n-jobs 16

Usage (generic, no splits pkl — random 80/10/10 split):
    python preprocess_stability_data.py \\
        --csv variants.tsv --pdb-dir pdbs/ --outdir out/ --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import re
import time
from pathlib import Path
from typing import Optional

import joblib as jl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from Bio import PDB
from joblib import Parallel, delayed
from scipy.io import loadmat, savemat
from sklearn.preprocessing import StandardScaler
from transformers import T5EncoderModel, T5Tokenizer


# ── constants ─────────────────────────────────────────────────────────────────

EDGE_DIST = 4.5  # Angstroms, any atom-pair threshold

THREE_LETTER_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D",
    "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
    "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
    "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Sec": "U", "Pyl": "O", "Asx": "B", "Glx": "Z",
    "Xaa": "X", "Ter": "*",
}

_SINGLE_MUT_RE = re.compile(r'^([A-Z])(\d+)([A-Z])$')


# ── protein ID helpers ────────────────────────────────────────────────────────

def _protein_id(wt_name: str) -> str:
    """Canonical dict key: strip .pdb and normalise | -> : (SPURS convention)."""
    name = wt_name[:-4] if wt_name.lower().endswith(".pdb") else wt_name
    return name.replace("|", ":")


def _pdb_filename(wt_name: str) -> str:
    """Actual filename in AlphaFold_model_PDBs/: normalise | -> :."""
    return wt_name.replace("|", ":")


# ── CSV parsing ───────────────────────────────────────────────────────────────

def parse_csv(csv_path: str) -> dict:
    """Parse a Tsuboyama2023-format CSV.

    Returns:
        {protein_id: {'variants': [(variant_str_0based, ddg), ...]}}
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
    df = df[df["mut_type"].apply(lambda m: bool(_SINGLE_MUT_RE.match(str(m))))].copy()

    proteins: dict = {}
    for _, row in df.iterrows():
        pid = _protein_id(str(row["WT_name"]))
        m = _SINGLE_MUT_RE.match(str(row["mut_type"]))
        wt_aa    = m.group(1)
        pos_0    = int(m.group(2)) - 1       # 1-based in CSV → 0-based
        mut_aa   = m.group(3)
        variant  = f"{wt_aa}{pos_0}{mut_aa}"
        ddg      = -float(row["ddG_ML"])     # negate: match SPURS sign convention

        if pid not in proteins:
            proteins[pid] = {"variants": []}
        proteins[pid]["variants"].append((variant, ddg))

    total_v = sum(len(v["variants"]) for v in proteins.values())
    print(f"  {len(proteins)} proteins, {total_v} variants after CSV filtering", flush=True)
    return proteins


# ── splits ────────────────────────────────────────────────────────────────────

def load_splits(splits_pkl: str) -> dict[str, set]:
    """Load SPURS mega_splits.pkl.

    Returns {'train': set_of_protein_ids, 'val': ..., 'test': ...}
    """
    with open(splits_pkl, "rb") as f:
        raw = pickle.load(f)
    result = {s: {_protein_id(str(n)) for n in raw[s]} for s in ("train", "val", "test")}
    print(f"  splits loaded: " + ", ".join(f"{k}={len(v)}" for k, v in result.items()),
          flush=True)
    return result


# ── contact graph building ────────────────────────────────────────────────────

def _build_graph_one(protein_id: str, pdb_path: str, graph_dir: str,
                     mode: str) -> Optional[str]:
    """Build and save a contact graph .mat file.

    Returns protein_id on success, None on failure.
    mode: 'monomer' (chain A only) | 'complex' (chain A + B).
    """
    out_mat = os.path.join(graph_dir, f"{protein_id}.mat")
    if os.path.exists(out_mat):
        return protein_id

    if not os.path.exists(pdb_path):
        return None

    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(protein_id, pdb_path)
        model0 = structure[0]

        aa_labels: list[str] = []
        all_atom_coords: list[dict] = []
        num_residues_a = 0

        chain_ids = ["A"] if mode == "monomer" else ["A", "B"]
        for ci, chain_id in enumerate(chain_ids):
            try:
                chain = model0[chain_id]
            except KeyError:
                if mode == "complex":
                    return None
                continue
            for residue in chain:
                if not PDB.is_aa(residue, standard=True):
                    continue
                aa = THREE_LETTER_TO_ONE.get(residue.get_resname().capitalize(), "X")
                aa_labels.append(aa)
                all_atom_coords.append({atom.get_name(): atom.coord for atom in residue})
            if ci == 0:
                num_residues_a = len(aa_labels)

        n = len(aa_labels)
        if n == 0:
            return None

        edge_mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                found = False
                for c_i in all_atom_coords[i].values():
                    for c_j in all_atom_coords[j].values():
                        if np.linalg.norm(c_i - c_j) <= EDGE_DIST:
                            edge_mat[i, j] = edge_mat[j, i] = 1.0
                            found = True
                            break
                    if found:
                        break

        savemat(out_mat, {"G": sp.csr_matrix(edge_mat), "L": aa_labels,
                          "NRR": num_residues_a})
        return protein_id

    except Exception as exc:
        print(f"  graph failed for {protein_id}: {exc}", flush=True)
        return None


def build_contact_graphs(proteins: dict, pdb_dir: str, graph_dir: str,
                         mode: str, n_jobs: int) -> set:
    """Build all contact graphs in parallel. Returns set of successful protein_ids."""
    os.makedirs(graph_dir, exist_ok=True)
    tasks = [
        (pid, str(Path(pdb_dir) / _pdb_filename(pid + ".pdb")), graph_dir, mode)
        for pid in proteins
    ]
    print(f"  building {len(tasks)} graphs (n_jobs={n_jobs})...", flush=True)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_build_graph_one)(pid, pdb_path, gdir, m)
        for pid, pdb_path, gdir, m in tasks
    )
    success = {r for r in results if r is not None}
    print(f"  {len(success)}/{len(tasks)} graphs built", flush=True)
    return success


def _read_mat_seq(mat) -> tuple[str, int]:
    """Extract sequence string and num_residues_a from a loaded .mat dict."""
    raw = mat["L"]
    if hasattr(raw, "flatten"):
        flat = raw.flatten()
        seq = "".join(str(c) for c in flat)
    else:
        seq = "".join(str(c) for c in raw)
    num_res_a = int(mat["NRR"].item()) if hasattr(mat["NRR"], "item") else int(mat["NRR"])
    return seq, num_res_a


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

def _load_prott5(device: torch.device):
    link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print(f"  loading {link}...", flush=True)
    model = T5EncoderModel.from_pretrained(link)
    if device.type == "cpu":
        model = model.to(torch.float32)
    model = model.to(device).eval()
    vocab = T5Tokenizer.from_pretrained(link, do_lower_case=False)
    return model, vocab


def _embed_batch(batch: list, model, vocab, device: torch.device,
                 emb_dict: dict) -> None:
    ids, seqs_sp, seq_lens = zip(*batch)
    tok = vocab.batch_encode_plus(list(seqs_sp), add_special_tokens=True,
                                  padding="longest")
    input_ids  = torch.tensor(tok["input_ids"]).to(device)
    attn_mask  = torch.tensor(tok["attention_mask"]).to(device)
    with torch.no_grad():
        out = model(input_ids, attention_mask=attn_mask)
    for bi, identifier in enumerate(ids):
        s_len = seq_lens[bi]
        emb_dict[identifier] = out.last_hidden_state[bi, :s_len].detach().cpu().numpy()


def compute_embeddings(seq_dict: dict, device: torch.device,
                       max_residues: int = 4000, max_seq_len: int = 1000,
                       max_batch: int = 100) -> dict:
    """Compute per-residue ProtT5 embeddings.

    seq_dict: {key: amino_acid_sequence_string}
    Returns: {key: np.ndarray(L, 1024)}
    """
    model, vocab = _load_prott5(device)

    # sort descending by length to pack batches efficiently
    sorted_items = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
    emb_dict: dict = {}
    batch: list = []
    total = len(sorted_items)

    for seq_idx, (key, seq) in enumerate(sorted_items, 1):
        seq_clean = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
        s_len = len(seq_clean)
        batch.append((key, " ".join(list(seq_clean)), s_len))

        n_res_batch = sum(sl for _, _, sl in batch) + s_len
        flush = (
            len(batch) >= max_batch
            or n_res_batch >= max_residues
            or seq_idx == total
            or s_len > max_seq_len
        )
        if flush:
            try:
                _embed_batch(batch, model, vocab, device, emb_dict)
            except RuntimeError as exc:
                print(f"  RuntimeError on batch size {len(batch)}: {exc}", flush=True)
                for item in batch:
                    try:
                        _embed_batch([item], model, vocab, device, emb_dict)
                    except Exception as exc2:
                        print(f"  skipping {item[0]}: {exc2}", flush=True)
            batch = []

        if seq_idx % 2000 == 0:
            print(f"  embedded {seq_idx}/{total}...", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {len(emb_dict)} embeddings computed", flush=True)
    return emb_dict


# ── data assembly ─────────────────────────────────────────────────────────────

def assemble_data(proteins: dict, emb_dict: dict, graph_dir: str,
                  splits: dict[str, set], mode: str,
                  subgraph_hops: int = 0) -> dict:
    """Assemble flat parallel lists of per-variant data with train/val/test split indices.

    subgraph_hops=0 (default): store full (L, 1024) embedding arrays — backward compatible.
    subgraph_hops>0: store sparse dicts (only residues within N hops of the mutation site).
        Expand back to dense with expand_emb() at training time.  Non-2-hop nodes have
        zero initial features, which has no effect on the mutation-site output after 2 GAT
        layers (information cannot propagate further than 2 hops in 2 rounds of message
        passing).

    Returns a dict suitable for saving as preprocessed.pkl.
    """
    pid_to_split: dict[str, str] = {}
    for sname, sset in splits.items():
        for pid in sset:
            pid_to_split[pid] = sname

    keys = ["vt_ids", "prott5_embeddings", "mutation_site_diffs",
            "edge_mats", "seq_lengths", "mutation_indices", "ddg_labels", "split_labels"]
    all_data: dict = {k: [] for k in keys}

    sk_no_split = sk_no_graph = sk_no_emb = sk_pos_mismatch = 0

    for pid, pdata in proteins.items():
        split_name = pid_to_split.get(pid)
        if split_name is None:
            sk_no_split += 1
            continue

        mat_path = os.path.join(graph_dir, f"{pid}.mat")
        if not os.path.exists(mat_path):
            sk_no_graph += 1
            continue

        mat = loadmat(mat_path)
        edge_mat_dense = sp.csr_matrix(mat["G"]).toarray().astype(np.float32)
        pdb_seq, num_res_a = _read_mat_seq(mat)
        np.fill_diagonal(edge_mat_dense, 1)

        wt_emb = emb_dict.get(pid)
        if wt_emb is None or len(wt_emb) != len(pdb_seq):
            sk_no_emb += 1
            continue

        for variant_str, ddg in pdata["variants"]:
            m = _SINGLE_MUT_RE.match(variant_str)
            if m is None:
                continue
            wt_aa  = m.group(1)
            pos_0  = int(m.group(2))
            mut_aa = m.group(3)

            if pos_0 >= len(pdb_seq) or pdb_seq[pos_0] != wt_aa:
                sk_pos_mismatch += 1
                continue

            vt_key = f"{pid} {variant_str}"
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

def main() -> None:
    p = argparse.ArgumentParser(
        description="Preprocess stability dataset for GAT_mut_processor training"
    )
    p.add_argument("--csv",      required=True,
                   help="Variants CSV (Tsuboyama2023 format or custom TSV with "
                        "WT_name/mut_type/aa_seq/ddG_ML columns)")
    p.add_argument("--pdb-dir",  required=True,
                   help="Directory containing AlphaFold PDB files")
    p.add_argument("--splits",   default="",
                   help="SPURS mega_splits.pkl for train/val/test (omit for random 80/10/10)")
    p.add_argument("--outdir",   required=True, help="Output directory")
    p.add_argument("--mode",     default="monomer", choices=["monomer", "complex"],
                   help="monomer: single chain A; complex: chains A+B concatenated")
    p.add_argument("--device",   default="",
                   help="PyTorch device (default: auto-detect)")
    p.add_argument("--n-jobs",        type=int, default=8,
                   help="Parallel jobs for contact graph building")
    p.add_argument("--subgraph-hops", type=int, default=0,
                   help="Only store ProtT5 features for residues within N hops of the "
                        "mutation site (0 = full embedding, backward compatible default). "
                        "Reduces preprocessed.pkl size for large proteins with no loss of "
                        "model accuracy (non-N-hop nodes cannot influence the 2-layer GAT "
                        "output at the mutation site).")
    args = p.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    outdir    = Path(args.outdir)
    graph_dir = str(outdir / "graphs")
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: parse CSV ──────────────────────────────────────────────────
    print("\n[1/6] Parsing CSV...", flush=True)
    proteins = parse_csv(args.csv)

    # ── Step 2: load splits ────────────────────────────────────────────────
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
        print(f"  random splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()),
              flush=True)

    # ── Step 3: build contact graphs ───────────────────────────────────────
    print("\n[3/6] Building contact graphs...", flush=True)
    successful = build_contact_graphs(proteins, args.pdb_dir, graph_dir,
                                      args.mode, args.n_jobs)

    # ── Step 3b: extract PDB sequences for embedding consistency ──────────
    print("\n[3b/6] Reading PDB sequences from .mat files...", flush=True)
    pdb_seqs: dict[str, str] = {}
    for pid in successful:
        mat = loadmat(os.path.join(graph_dir, f"{pid}.mat"))
        seq, _ = _read_mat_seq(mat)
        pdb_seqs[pid] = seq
    print(f"  {len(pdb_seqs)} sequences extracted", flush=True)

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
            for variant_str, _ in proteins[pid]["variants"]:
                m = _SINGLE_MUT_RE.match(variant_str)
                if m is None:
                    continue
                wt_aa, pos_0, mut_aa = m.group(1), int(m.group(2)), m.group(3)
                if pos_0 < len(pdb_seq) and pdb_seq[pos_0] == wt_aa:
                    mut_seq = pdb_seq[:pos_0] + mut_aa + pdb_seq[pos_0 + 1:]
                    seq_dict[f"{pid} {variant_str}"] = mut_seq

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
    all_data = assemble_data(proteins, emb_dict, graph_dir, splits, args.mode,
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
