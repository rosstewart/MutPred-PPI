#!/usr/bin/env python
"""MutPred-PPI group cross-validation training script.

Model: single GAT_mut_processor (hidden_dim=64) identical to
predictors/mutpredppi.py.  Architecture does not vary across datasets.

CV fold assignments are generated inline (deterministic: GroupKFold with the
canonical data shuffle, random_state=gcv_seed) rather than read from pre-computed
split files.

Usage:
    conda run -n ppi python src/evaluation/mutpred_ppi_cv.py --dataset sahni_fragoza --device cuda:0
    conda run -n ppi python src/evaluation/mutpred_ppi_cv.py --dataset sahni_fragoza \\
        --device cuda:0 --n-gcv 30 --outdir /path/to/results/
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import pickle
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (  # noqa: E402
    REPO_ROOT as _REPO_ROOT,
    WEIGHTS_DIR as _MODEL_WEIGHTS_DIR,
    DATA_ROOT as _DATA_ROOT,
    HOME_DIR as _HOME_DIR,
    REVISIONS_DIR as _REVISIONS_DIR,
    GCV_RESULTS_DIR as _GCV_RESULTS_DIR,
    DATA_CACHES_DIR as _DATA_CACHES_DIR,
    cdhit_binary,
    cv_reference_dir,
)

_V1_0_SCALER_PATH          = _MODEL_WEIGHTS_DIR / "v1_0" / "mutation_diff_scaler_v1_0.pkl"
_MEGASCALE_SCALER_PATH     = _MODEL_WEIGHTS_DIR / "mutation_diff_scaler.pkl"
_V1_0_PRETRAINED_PATH      = _MODEL_WEIGHTS_DIR / "v1_0" / "MutPred-PPI_v1_0_stability_pretrain.pt"
_MEGASCALE_PRETRAINED_PATH = _MODEL_WEIGHTS_DIR / "MutPred-PPI_stability_pretrain.pt"
_METHOD                    = "interaction_loss"


# ── model — verbatim from predictors/mutpredppi.py ────────────────────────────

class GAT_mut_processor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1,
                 num_heads: int = 4, mutation_diff_dim: int = 1024):
        super().__init__()
        self.mutation_diff_processor = nn.Sequential(
            nn.Linear(mutation_diff_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
        )
        self.complex_gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.complex_gat2 = GATConv(hidden_dim * num_heads, hidden_dim // 2, heads=1, concat=False)
        self.binding_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, output_dim),
        )

    def forward(self, x, edge_index, mutation_idx, num_mut_res, mutation_site_diff):
        if mutation_site_diff.dim() == 1:
            mutation_site_diff = mutation_site_diff.unsqueeze(0)
        processed_mut_diff = self.mutation_diff_processor(mutation_site_diff)
        h = torch.relu(self.complex_gat1(x, edge_index))
        h = torch.relu(self.complex_gat2(h, edge_index))
        features_at_mutation = h[mutation_idx:mutation_idx + 1]
        combined = torch.cat([features_at_mutation, processed_mut_diff], dim=-1)
        return self.binding_predictor(combined)


# ── ablation model variants ───────────────────────────────────────────────────

class GAT_mut_processor_no_gat(nn.Module):
    """Ablation (2): structural GAT removed — mutation diff processor + predictor only."""
    def __init__(self, output_dim: int = 1, mutation_diff_dim: int = 1024):
        super().__init__()
        self.mutation_diff_processor = nn.Sequential(
            nn.Linear(mutation_diff_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
        )
        self.binding_predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, output_dim),
        )

    def forward(self, x, edge_index, mutation_idx, num_mut_res, mutation_site_diff):
        if mutation_site_diff.dim() == 1:
            mutation_site_diff = mutation_site_diff.unsqueeze(0)
        return self.binding_predictor(self.mutation_diff_processor(mutation_site_diff))


class GAT_mut_processor_no_mut(nn.Module):
    """Ablation (3): mutation diff processor removed — structural GAT + predictor only."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1,
                 num_heads: int = 4):
        super().__init__()
        self.complex_gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.complex_gat2 = GATConv(hidden_dim * num_heads, hidden_dim // 2, heads=1, concat=False)
        self.binding_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, output_dim),
        )

    def forward(self, x, edge_index, mutation_idx, num_mut_res, mutation_site_diff):
        h = torch.relu(self.complex_gat1(x, edge_index))
        h = torch.relu(self.complex_gat2(h, edge_index))
        return self.binding_predictor(h[mutation_idx:mutation_idx + 1])


# ── dataset configuration ─────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    name: str
    vt_ids_file: str            # canonical row set + ordering
    clusters_file: str = ""     # canonical cd-hit clusters (GroupKFold groups)
    fold_splits_pat: str = ""   # canonical fold splits, "{s}" for the GCV seed


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "sahni": DatasetConfig(
        name="sahni", vt_ids_file="all_vt_ids.pkl",
        clusters_file="clusters.pkl",
        fold_splits_pat="fold_splits_{s}.pkl"),
    "sahni_fragoza": DatasetConfig(
        name="sahni_fragoza", vt_ids_file="sahni_fragoza_train_all_vt_ids.pkl",
        clusters_file="sahni_fragoza_train_clusters.pkl",
        fold_splits_pat="sahni_fragoza_train_fold_splits_{s}.pkl"),
    "sahni_fragoza_varchamp1p_cava": DatasetConfig(
        name="sahni_fragoza_varchamp1p_cava",
        vt_ids_file="sahni_fragoza_varchamp1p_cava_train_all_vt_ids.pkl",
        clusters_file="sahni_fragoza_varchamp1p_cava_train_clusters.pkl",
        fold_splits_pat="sahni_fragoza_varchamp1p_cava_train_fold_splits_{s}.pkl"),
    "sahni_varchamp1p_cava": DatasetConfig(
        name="sahni_varchamp1p_cava",
        vt_ids_file="sahni_varchamp1p_cava_train_all_vt_ids.pkl",
        clusters_file="sahni_varchamp1p_cava_train_clusters.pkl",
        fold_splits_pat="sahni_varchamp1p_cava_train_fold_splits_{s}.pkl"),
    "sahni_fragoza_varchamp_full": DatasetConfig(
        name="sahni_fragoza_varchamp_full",
        vt_ids_file="sahni_fragoza_varchamp_full_train_all_vt_ids.pkl",
        fold_splits_pat="sahni_fragoza_varchamp_full_train_fold_splits_{s}.pkl"),
    "sahni_fragoza_varchamp_pooled": DatasetConfig(
        name="sahni_fragoza_varchamp_pooled",
        vt_ids_file="sahni_fragoza_varchamp_pooled_train_all_vt_ids.pkl",
        fold_splits_pat="sahni_fragoza_varchamp_pooled_train_fold_splits_{s}.pkl"),
    "sahni_fragoza_varchamp_full_pooled": DatasetConfig(
        name="sahni_fragoza_varchamp_full_pooled",
        vt_ids_file="sahni_fragoza_varchamp_full_pooled_train_all_vt_ids.pkl",
        fold_splits_pat="sahni_fragoza_varchamp_full_pooled_train_fold_splits_{s}.pkl"),
}


def canonical_vt_ids_path(cfg: DatasetConfig) -> Path:
    """Canonical ordering for a dataset: in-repo copy, else the legacy dir."""
    return Path(cv_reference_dir()) / cfg.vt_ids_file


def canonical_clusters_path(cfg: DatasetConfig):
    """Canonical cd-hit clusters, or None if this dataset has none saved."""
    if not cfg.clusters_file:
        return None
    return Path(cv_reference_dir()) / cfg.clusters_file


def canonical_fold_splits_path(cfg: DatasetConfig, gcv_seed: int):
    """Canonical fold splits for one seed, or None if this dataset has none saved."""
    if not cfg.fold_splits_pat:
        return None
    return Path(cv_reference_dir()) / cfg.fold_splits_pat.format(s=gcv_seed)


# ── ID splitting helpers ───────────────────────────────────────────────────────

def split_complex_id_hyphen(complex_id: str) -> Tuple[str, str]:
    """sahni_fragoza: hyphen-delimited, split at last non-integer part."""
    parts = complex_id.split("-")
    for i in range(len(parts) - 1, 0, -1):
        if not parts[i].isdigit():
            return "-".join(parts[:i]), "-".join(parts[i:])
    raise ValueError(f"Could not split: {complex_id}")


def split_wt_id_underscore(wt_id: str) -> Tuple[str, str]:
    """varchamp1p/cava: underscore-delimited gene+orf IDs."""
    if wt_id.startswith("NP_"):
        return "_".join(wt_id.split("_")[:2]), "_".join(wt_id.split("_")[2:])
    if len(wt_id.split("_")) == 2:
        return tuple(wt_id.split("_"))  # type: ignore[return-value]
    # gene_orf format: find the first numeric part to split
    for part_idx, part in enumerate(wt_id.split("_")):
        if len(part) == 1:
            continue
        try:
            int(part)
            split_at = part_idx + 1
            return "_".join(wt_id.split("_")[:split_at]), "_".join(wt_id.split("_")[split_at:])
        except ValueError:
            continue
    raise ValueError(f"Could not split: {wt_id}")


def get_gene_name(gene_and_orf: str) -> str:
    """Strip numeric ORF suffix: 'BRCA1_1' → 'BRCA1'."""
    if "_" not in gene_and_orf:
        return gene_and_orf
    return "_".join(gene_and_orf.split("_")[:-1])


# ── CD-HIT clustering ─────────────────────────────────────────────────────────

def cluster_sequences(sequences: list, identity: float = 0.5) -> list:
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".fasta") as f:
        fasta_path = f.name
        for i, seq in enumerate(sequences):
            f.write(f">seq{i}\n{seq}\n")

    out_path = fasta_path + "_clustered"
    result = subprocess.run(
        [cdhit_binary(), "-i", fasta_path, "-o", out_path, "-c", str(identity), "-n", "3"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        # Returning [] here would surface much later as an opaque IndexError,
        # since these clusters are the GroupKFold groups.
        raise RuntimeError(
            f"cd-hit failed (exit {result.returncode}):\n{result.stderr.decode()}"
        )

    clstr_path = out_path + ".clstr"
    cluster_map: dict[int, int] = {}
    cluster_id = 0
    with open(clstr_path) as f:
        for line in f:
            if line.startswith(">Cluster"):
                cluster_id = int(line.strip().split()[-1])
            else:
                idx = int(line.split(">seq")[1].split("...")[0])
                cluster_map[idx] = cluster_id

    for path in (fasta_path, out_path, clstr_path):
        if os.path.exists(path):
            os.remove(path)

    return [cluster_map[i] for i in range(len(sequences))]


# ── per-source graph data loading ─────────────────────────────────────────────

def _build_emb_dict(
    graph_dir: str,
    t5_emb_dict: dict,
    id_parts_fn,           # callable(complex_id) → (refseq_id, partner_id)
    use_wt_emb: bool = False,
) -> Tuple[dict, dict, dict]:
    """Build updated_emb_dict, complex_length_dict and complex_pair_dict from .mat directory.

    complex_pair_dict records the (interactor, partner) split produced by this
    dataset's own id_parts_fn, so downstream code (e.g. C1/C2/C3 classification)
    never has to re-parse complex_id strings.
    """
    updated_emb_dict: dict = {}
    complex_length_dict: dict = {}
    complex_pair_dict: dict = {}
    for wt_f_mat in glob.glob(f"{graph_dir}/*.mat"):
        complex_id = os.path.splitext(os.path.basename(wt_f_mat))[0]
        try:
            refseq_id, partner_id = id_parts_fn(complex_id)
        except (ValueError, AssertionError):
            continue
        if refseq_id not in t5_emb_dict or partner_id not in t5_emb_dict:
            continue
        for vt_f in glob.glob(f"{graph_dir}/{complex_id}*{_METHOD}_variant*.labels"):
            variant = vt_f.split("variant_")[-1].split(".")[0]
            mut_idx = int(variant[1:-1])
            vt_id = f"{refseq_id} {variant}"
            if vt_id not in t5_emb_dict:
                continue
            vt_emb = t5_emb_dict[vt_id].copy()
            # ablation (4): use WT embeddings for interactor instead of VT
            node_emb = t5_emb_dict[refseq_id] if use_wt_emb else vt_emb
            updated_emb_dict[f"{complex_id} {variant}"] = (
                np.concatenate([node_emb, t5_emb_dict[partner_id]]),
                vt_emb[mut_idx] - t5_emb_dict[refseq_id][mut_idx],
            )
            complex_length_dict[f"{complex_id} {variant}"] = [
                len(vt_emb), len(t5_emb_dict[partner_id])
            ]
            complex_pair_dict[f"{complex_id} {variant}"] = (refseq_id, partner_id)
    return updated_emb_dict, complex_length_dict, complex_pair_dict


def _load_graphs(
    graph_dir: str,
    updated_emb_dict: dict,
    complex_length_dict: dict,
    complex_pair_dict: dict,
    seq_confirmed_set: Optional[set] = None,
) -> dict:
    data: dict = {k: [] for k in [
        "prott5_embeddings", "mutation_site_diffs", "all_vt_ids", "all_wt_ids",
        "vt_seqs", "wt_seqs", "edge_mats", "seq_lengths", "all_pairs",
    ]}
    for wt_f_mat in glob.glob(f"{graph_dir}/*.mat"):
        complex_id = os.path.splitext(os.path.basename(wt_f_mat))[0]
        mat = loadmat(wt_f_mat)
        wt_edge_mat = sp.csr_matrix(mat["G"]).toarray()
        wt_seq = "".join(mat["L"])

        for key, (emb, diff) in updated_emb_dict.items():
            if key.split(" ")[0] != complex_id:
                continue
            if seq_confirmed_set is not None and key not in seq_confirmed_set:
                continue
            variant = key.split(" ")[1]
            missense_idx = int(variant[1:-1])
            if wt_seq[missense_idx] != variant[0]:
                continue
            vt_seq = list(wt_seq)
            if vt_seq[missense_idx] == variant[-1]:
                continue  # synonymous
            vt_seq[missense_idx] = variant[-1]
            vt_seq_str = "".join(vt_seq)
            if len(vt_seq_str) != emb.shape[0] or wt_edge_mat.shape[0] != emb.shape[0]:
                continue
            data["prott5_embeddings"].append(emb)
            data["mutation_site_diffs"].append(diff)
            data["all_vt_ids"].append(key)
            data["all_wt_ids"].append(complex_id)
            data["vt_seqs"].append(vt_seq_str)
            data["wt_seqs"].append(wt_seq)
            np.fill_diagonal(wt_edge_mat, 1)
            data["edge_mats"].append(wt_edge_mat.copy())
            data["seq_lengths"].append(complex_length_dict[key])
            data["all_pairs"].append(complex_pair_dict[key])
    return data


def _gather_labels_pos_neg(graph_dir: str, all_vt_ids: list) -> Tuple[list, list]:
    """sahni / sahni_fragoza label format: last col of _pos/_neg file = variant."""
    pos_labels, neg_labels = [], []
    for vt_id in all_vt_ids:
        wt_id, variant = vt_id.split(" ", 1)
        mut_idx = int(variant[1:-1])
        pos_f = f"{graph_dir}/{wt_id}.{_METHOD}_pos"
        neg_f = f"{graph_dir}/{wt_id}.{_METHOD}_neg"
        seen = False
        if os.path.exists(pos_f):
            for line in open(pos_f):
                if line.strip().split("\t")[-1] == variant:
                    pos_labels.append([mut_idx]); neg_labels.append([])
                    assert not seen; seen = True
        if not seen and os.path.exists(neg_f):
            for line in open(neg_f):
                if line.strip().split("\t")[-1] == variant:
                    pos_labels.append([]); neg_labels.append([mut_idx])
                    assert not seen; seen = True
        assert seen, f"Variant {variant} not found for {wt_id}"
    return pos_labels, neg_labels


def _gather_labels_scored(graph_dir: str, all_vt_ids: list) -> Tuple[list, list]:
    """varchamp1p / cava label format: second-to-last col = variant, last = score."""
    pos_labels, neg_labels = [], []
    for vt_id in all_vt_ids:
        wt_id, variant = vt_id.split(" ", 1)
        mut_idx = int(variant[1:-1])
        pos_f = f"{graph_dir}/{wt_id}.{_METHOD}_pos"
        seen = False
        for line in open(pos_f):
            cols = line.strip().split("\t")
            if cols[-2] == variant:
                assert not seen; seen = True
                if float(cols[-1]) == 0:
                    pos_labels.append([mut_idx]); neg_labels.append([])
                else:
                    pos_labels.append([]); neg_labels.append([mut_idx])
        assert seen, f"Variant {variant} not found in {pos_f}"
    return pos_labels, neg_labels


def _load_seq_confirmed_set(pkl_path: str) -> set:
    with open(pkl_path, "rb") as f:
        variants = pickle.load(f)
    result = set()
    for mutated, one_based, partner, _ in variants:
        vt_id = (mutated + "_" + partner + " "
                 + one_based[0] + str(int(one_based[1:-1]) - 1) + one_based[-1])
        result.add(vt_id)
    return result


# ── full dataset loaders ───────────────────────────────────────────────────────

def load_sahni(use_wt_emb: bool = False) -> dict:
    graph_dir = str(_HOME_DIR / "sahni" / "af3_graphs")
    with open(str(_DATA_ROOT / "sahni_wt_and_vt_t5.pkl"), "rb") as f:
        t5 = pickle.load(f)
    # sahni: complex_id is NP_XXXXX_N_PARTNER, split at position 2
    emb_dict, len_dict, pair_dict = _build_emb_dict(
        graph_dir, t5,
        lambda cid: ("_".join(cid.split("_")[:2]), "_".join(cid.split("_")[2:])),
        use_wt_emb=use_wt_emb,
    )
    data = _load_graphs(graph_dir, emb_dict, len_dict, pair_dict)
    pos, neg = _gather_labels_pos_neg(graph_dir, data["all_vt_ids"])
    data["pos_labels"] = pos
    data["neg_labels"] = neg
    # The interactor is a RefSeq accession but the partner is a gene symbol.
    # C1/C2/C3 asks whether a protein was seen in training, so both sides must
    # live in one namespace: map the interactor to its symbol.
    with open(str(_HOME_DIR / "sahni" / "refseq_to_symbol.pkl"), "rb") as f:
        refseq_to_symbol = pickle.load(f)
    data["all_pairs"] = [(refseq_to_symbol.get(a, a), b) for a, b in data["all_pairs"]]
    clusters = [str(c) for c in cluster_sequences(data["wt_seqs"])]
    data["clusters"] = clusters
    return data


def load_sahni_fragoza(use_wt_emb: bool = False) -> dict:
    main_dir  = str(_DATA_ROOT / "swing_train")
    graph_dir = f"{main_dir}/af3_graphs"
    with open(f"{main_dir}/swing_train_t5_embs.pkl", "rb") as f:
        t5 = pickle.load(f)
    emb_dict, len_dict, pair_dict = _build_emb_dict(graph_dir, t5, split_complex_id_hyphen,
                                         use_wt_emb=use_wt_emb)
    data = _load_graphs(graph_dir, emb_dict, len_dict, pair_dict)
    pos, neg = _gather_labels_pos_neg(graph_dir, data["all_vt_ids"])
    data["pos_labels"] = pos
    data["neg_labels"] = neg
    clusters = [str(c) for c in cluster_sequences(data["wt_seqs"])]
    data["clusters"] = clusters
    return data


def _load_varchamp1p_raw(use_wt_emb: bool = False) -> dict:
    graph_dir = str(_DATA_ROOT / "varchamp1p" / "af3_graphs")
    with open(str(_DATA_ROOT / "varchamp1p" / "varchamp1p_t5_embs.pkl"), "rb") as f:
        t5 = pickle.load(f)
    seq_conf = _load_seq_confirmed_set(str(_HOME_DIR / "varchamp1p" / "seq_confirmed_variants.pkl"))
    # varchamp1p: complex_id is GENE1_orf_GENE2_orf, split at position 1
    emb_dict, len_dict, pair_dict = _build_emb_dict(
        graph_dir, t5,
        lambda cid: ("_".join(cid.split("_")[:1]), "_".join(cid.split("_")[1:])),
        use_wt_emb=use_wt_emb,
    )
    data = _load_graphs(graph_dir, emb_dict, len_dict, pair_dict, seq_conf)
    pos, neg = _gather_labels_scored(graph_dir, data["all_vt_ids"])
    data["pos_labels"] = pos
    data["neg_labels"] = neg
    return data


def _load_cava_raw(use_wt_emb: bool = False) -> dict:
    graph_dir = str(_DATA_ROOT / "cava" / "af3_graphs")
    with open(str(_DATA_ROOT / "cava" / "cava_t5_embs.pkl"), "rb") as f:
        t5 = pickle.load(f)
    seq_conf = _load_seq_confirmed_set(str(_HOME_DIR / "cava" / "seq_confirmed_variants.pkl"))
    emb_dict, len_dict, pair_dict = _build_emb_dict(
        graph_dir, t5,
        lambda cid: ("_".join(cid.split("_")[:1]), "_".join(cid.split("_")[1:])),
        use_wt_emb=use_wt_emb,
    )
    data = _load_graphs(graph_dir, emb_dict, len_dict, pair_dict, seq_conf)
    pos, neg = _gather_labels_scored(graph_dir, data["all_vt_ids"])
    data["pos_labels"] = pos
    data["neg_labels"] = neg
    return data


def _remap_vt_ids(src: dict, new_wt_ids: list, new_pairs: Optional[list] = None) -> None:
    """Update all_wt_ids, all_vt_ids and all_pairs in-place to use new_wt_ids.

    new_pairs carries the (interactor, partner) IDs in the same namespace as
    new_wt_ids; callers already compute this split to build new_wt_ids.
    """
    src["all_wt_ids"] = new_wt_ids
    src["all_vt_ids"] = [
        f"{wt_id} {vt_id.split(' ', 1)[1]}"
        for wt_id, vt_id in zip(new_wt_ids, src["all_vt_ids"])
    ]
    if new_pairs is not None:
        src["all_pairs"] = new_pairs


_DATA_KEYS = [
    "prott5_embeddings", "mutation_site_diffs", "edge_mats",
    "pos_labels", "neg_labels", "all_wt_ids", "all_vt_ids",
    "wt_seqs", "vt_seqs", "seq_lengths", "all_pairs",
]


def _dedup_and_merge(sources: list, drop_conflicts: bool = False) -> dict:
    """Drop duplicate vt_ids across sources (keep first occurrence); merge.

    When drop_conflicts=True, any vt_id with disagreeing labels across sources
    (one says disrupted, another says maintained) is dropped from all sources.
    """
    if drop_conflicts:
        # Two-pass: find conflicted vt_ids then exclude them
        label_sign: dict = {}  # vt_id -> True=pos, False=neg
        conflict_set: set = set()
        for src in sources:
            for i, vt_id in enumerate(src["all_vt_ids"]):
                if vt_id in conflict_set:
                    continue
                sign = bool(src["pos_labels"][i])  # True if disrupted (pos = disrupted)
                if vt_id in label_sign:
                    if label_sign[vt_id] != sign:
                        conflict_set.add(vt_id)
                else:
                    label_sign[vt_id] = sign
        if conflict_set:
            print(f"  _dedup_and_merge: dropping {len(conflict_set)} conflicting-label vt_ids",
                  flush=True)
    else:
        conflict_set = set()

    seen: set = set()
    merged: dict = {k: [] for k in _DATA_KEYS}
    for src in sources:
        for i, vt_id in enumerate(src["all_vt_ids"]):
            if vt_id in seen or vt_id in conflict_set:
                continue
            seen.add(vt_id)
            for k in _DATA_KEYS:
                merged[k].append(src[k][i])
    return merged


def load_sahni_fragoza_varchamp1p_cava(use_wt_emb: bool = False) -> dict:
    """Normalize all IDs to UniProt_UniProt (matching the canonical vt_ids file)."""
    sf  = load_sahni_fragoza(use_wt_emb=use_wt_emb)
    vc  = _load_varchamp1p_raw(use_wt_emb=use_wt_emb)
    cava = _load_cava_raw(use_wt_emb=use_wt_emb)

    with open(str(_HOME_DIR / "varchamp1p" / "gene_symbol_to_uniprot.pkl"), "rb") as f:
        vc_gs2u = pickle.load(f)
    with open(str(_HOME_DIR / "cava" / "gene_symbol_to_uniprot.pkl"), "rb") as f:
        cava_gs2u = pickle.load(f)

    # sahni_fragoza: "A-B" → "A_B"
    sf_new_wt, sf_new_pairs = [], []
    for wt_id in sf["all_wt_ids"]:
        a, b = split_complex_id_hyphen(wt_id)
        sf_new_wt.append(f"{a}_{b}")
        sf_new_pairs.append((a, b))
    _remap_vt_ids(sf, sf_new_wt, sf_new_pairs)

    # varchamp1p: gene+orf → UniProt_UniProt
    vc_new_wt, vc_new_pairs = [], []
    for wt_id in vc["all_wt_ids"]:
        a, b = split_wt_id_underscore(wt_id)
        ua, ub = vc_gs2u[get_gene_name(a)], vc_gs2u[get_gene_name(b)]
        vc_new_wt.append(f"{ua}_{ub}")
        vc_new_pairs.append((ua, ub))
    _remap_vt_ids(vc, vc_new_wt, vc_new_pairs)

    # cava: gene+orf → UniProt_UniProt
    cava_new_wt, cava_new_pairs = [], []
    for wt_id in cava["all_wt_ids"]:
        a, b = split_wt_id_underscore(wt_id)
        ua, ub = cava_gs2u[get_gene_name(a)], cava_gs2u[get_gene_name(b)]
        cava_new_wt.append(f"{ua}_{ub}")
        cava_new_pairs.append((ua, ub))
    _remap_vt_ids(cava, cava_new_wt, cava_new_pairs)

    data = _dedup_and_merge([sf, vc, cava])
    data["clusters"] = [str(c) for c in cluster_sequences(data["wt_seqs"])]
    return data


def _load_varchamp2026_raw(use_wt_emb: bool = False) -> dict:
    """Load 2026 VarChAMP variants from all_labeled_data.csv + 2026/graphs/.

    Graph files are UNIPROT1-UNIPROT2.mat (some stored in reversed order);
    NRR gives residue count for chain A so we can permute when reversed.
    ProtT5 embeddings use 'UNIPROT' (WT) and 'UNIPROT_MutXXX' (VT, 1-based) keys.
    Labels: perturbed=True → disrupted (pos, y=1), perturbed=False → maintained (neg, y=0).
    """
    _CSV      = str(_REVISIONS_DIR / "sfvc2026_labeled_data.csv")
    _T5_PKL   = str(_REVISIONS_DIR / "all_labeled_prott5_embeddings.pkl")
    _GRAPH_DIR = str(_REVISIONS_DIR / "graphs")

    df = pd.read_csv(_CSV)
    vc_df = df[df["dataset"].str.contains("VarChAMP", na=False)]

    with open(_T5_PKL, "rb") as f:
        t5 = pickle.load(f)

    data: dict = {k: [] for k in _DATA_KEYS}

    for _, row in vc_df.iterrows():
        inter     = str(row["interactor"])
        partner   = str(row["partner"])
        mut_1b    = str(row["mutation"])      # e.g. "E80K" (1-based)
        perturbed = bool(row["perturbed"])
        mut_idx   = int("".join(c for c in mut_1b if c.isdigit())) - 1  # 0-based

        fwd = os.path.join(_GRAPH_DIR, f"{inter}-{partner}.mat")
        rev = os.path.join(_GRAPH_DIR, f"{partner}-{inter}.mat")
        if os.path.exists(fwd):
            mat_path, reversed_graph = fwd, False
        elif os.path.exists(rev):
            mat_path, reversed_graph = rev, True
        else:
            continue

        vt_key = f"{inter}_{mut_1b}"
        if inter not in t5 or vt_key not in t5 or partner not in t5:
            continue

        mat = loadmat(mat_path)
        wt_seq_full = "".join(mat["L"])
        G = sp.csr_matrix(mat["G"]).toarray()
        nrr = int(mat["NRR"].flat[0])  # residues in chain A of the stored .mat

        if reversed_graph:
            # stored as (partner, inter); permute so interactor is chain A
            n_total = G.shape[0]
            perm = list(range(nrr, n_total)) + list(range(nrr))
            G = G[np.ix_(perm, perm)]
            wt_seq_full = wt_seq_full[nrr:] + wt_seq_full[:nrr]

        wt_emb   = t5[inter]    # (L_inter, 1024)
        vt_emb   = t5[vt_key]   # (L_inter, 1024)
        part_emb = t5[partner]  # (L_partner, 1024)

        L_inter = len(wt_emb)
        if len(wt_seq_full) != L_inter + len(part_emb):
            continue
        wt_seq = wt_seq_full[:L_inter]
        if mut_idx >= L_inter or wt_seq[mut_idx] != mut_1b[0]:
            continue
        if wt_seq[mut_idx] == mut_1b[-1]:
            continue  # synonymous

        vt_seq_list = list(wt_seq)
        vt_seq_list[mut_idx] = mut_1b[-1]
        vt_seq_str = "".join(vt_seq_list)
        if len(vt_seq_str) != vt_emb.shape[0]:
            continue

        node_emb = wt_emb if use_wt_emb else vt_emb
        combined = np.concatenate([node_emb, part_emb])
        diff     = vt_emb[mut_idx] - wt_emb[mut_idx]
        em       = G.copy()
        np.fill_diagonal(em, 1)

        # vt_id format: "UNIPROT_UNIPROT Xmut_idxY" (0-based position)
        mut_zero = f"{mut_1b[0]}{mut_idx}{mut_1b[-1]}"
        wt_id = f"{inter}_{partner}"
        vt_id = f"{wt_id} {mut_zero}"

        data["prott5_embeddings"].append(combined)
        data["mutation_site_diffs"].append(diff)
        data["edge_mats"].append(em)
        data["all_wt_ids"].append(wt_id)
        data["all_vt_ids"].append(vt_id)
        data["wt_seqs"].append(wt_seq_full)
        data["vt_seqs"].append(vt_seq_str)
        data["seq_lengths"].append([L_inter, len(part_emb)])
        data["all_pairs"].append((inter, partner))

        if perturbed:  # disrupted → pos → y=1 (consistent with Sahni/Fragoza/VC1p/CAVA)
            data["pos_labels"].append([mut_idx])
            data["neg_labels"].append([])
        else:          # maintained → neg → y=0
            data["pos_labels"].append([])
            data["neg_labels"].append([mut_idx])

    return data


def load_sahni_fragoza_varchamp_full(use_wt_emb: bool = False) -> dict:
    """sahni_fragoza + VarChAMP2026 + VarChAMP1p + CAVA (all available data)."""
    sf   = load_sahni_fragoza(use_wt_emb=use_wt_emb)
    vc26 = _load_varchamp2026_raw(use_wt_emb=use_wt_emb)
    vc1p = _load_varchamp1p_raw(use_wt_emb=use_wt_emb)
    cava = _load_cava_raw(use_wt_emb=use_wt_emb)

    sf_new_wt, sf_new_pairs = [], []
    for wt_id in sf["all_wt_ids"]:
        a, b = split_complex_id_hyphen(wt_id)
        sf_new_wt.append(f"{a}_{b}")
        sf_new_pairs.append((a, b))
    _remap_vt_ids(sf, sf_new_wt, sf_new_pairs)

    data = _dedup_and_merge([sf, vc26, vc1p, cava])
    data["clusters"] = [str(c) for c in cluster_sequences(data["wt_seqs"])]
    return data


def _load_varchamp_pooled_raw(use_wt_emb: bool = False) -> dict:
    """Load VarChAMP_pooled variants from training_data.csv.

    Graph .mat files: /varchamp_pooled/af3_graphs/{INTERACTOR}_{PARTNER}.mat
    T5 pkl keys: {INTERACTOR} (WT), {INTERACTOR}_{mutation} (mutant, e.g. O00189_E80K)
    Labels: perturbed=True → disrupted (pos, y=1), perturbed=False → maintained (neg, y=0).
    """
    _CSV      = str(_DATA_CACHES_DIR / "training_data_internal.csv")
    _T5_PKL   = str(_DATA_ROOT / "varchamp_pooled" / "varchamp_pooled_t5_embs.pkl")
    _GRAPH_DIR = str(_DATA_ROOT / "varchamp_pooled" / "af3_graphs")

    df = pd.read_csv(_CSV)
    pool_df = df[df["dataset"].str.contains("VarChAMP_pooled", na=False)].copy()

    with open(_T5_PKL, "rb") as f:
        t5 = pickle.load(f)

    data: dict = {k: [] for k in _DATA_KEYS}

    for _, row in pool_df.iterrows():
        inter     = str(row["interactor"])
        partner   = str(row["partner"])
        mut_1b    = str(row["mutation"])       # e.g. "E80K" (1-based)
        perturbed = bool(row["perturbed"])
        mut_idx   = int("".join(c for c in mut_1b if c.isdigit())) - 1  # 0-based

        mat_path = os.path.join(_GRAPH_DIR, f"{inter}_{partner}.mat")
        if not os.path.exists(mat_path):
            continue

        vt_key = f"{inter}_{mut_1b}"
        if inter not in t5 or vt_key not in t5 or partner not in t5:
            continue

        mat = loadmat(mat_path)
        wt_seq_full = "".join(mat["L"])
        G = sp.csr_matrix(mat["G"]).toarray()
        nrr = int(mat["NRR"].flat[0])

        wt_emb   = t5[inter]    # (L_inter, 1024)
        vt_emb   = t5[vt_key]   # (L_inter, 1024)
        part_emb = t5[partner]  # (L_partner, 1024)

        L_inter = len(wt_emb)
        if len(wt_seq_full) != L_inter + len(part_emb):
            continue
        wt_seq = wt_seq_full[:L_inter]
        if mut_idx >= L_inter or wt_seq[mut_idx] != mut_1b[0]:
            continue
        if wt_seq[mut_idx] == mut_1b[-1]:
            continue  # synonymous

        vt_seq_list = list(wt_seq)
        vt_seq_list[mut_idx] = mut_1b[-1]
        vt_seq_str = "".join(vt_seq_list)
        if len(vt_seq_str) != vt_emb.shape[0]:
            continue

        node_emb = wt_emb if use_wt_emb else vt_emb
        combined = np.concatenate([node_emb, part_emb])
        diff     = vt_emb[mut_idx] - wt_emb[mut_idx]
        em       = G.copy()
        np.fill_diagonal(em, 1)

        mut_zero = f"{mut_1b[0]}{mut_idx}{mut_1b[-1]}"
        wt_id = f"{inter}-{partner}"  # hyphen: matches generate_pooled_splits.py _normalize_vt_ids
        vt_id = f"{wt_id} {mut_zero}"

        data["prott5_embeddings"].append(combined)
        data["mutation_site_diffs"].append(diff)
        data["edge_mats"].append(em)
        data["all_wt_ids"].append(wt_id)
        data["all_vt_ids"].append(vt_id)
        data["wt_seqs"].append(wt_seq_full)
        data["vt_seqs"].append(vt_seq_str)
        data["seq_lengths"].append([L_inter, len(part_emb)])
        data["all_pairs"].append((inter, partner))

        if perturbed:  # disrupted → pos → y=1 (consistent with Sahni/Fragoza/VC1p/CAVA)
            data["pos_labels"].append([mut_idx])
            data["neg_labels"].append([])
        else:          # maintained → neg → y=0
            data["pos_labels"].append([])
            data["neg_labels"].append([mut_idx])

    print(f"  _load_varchamp_pooled_raw: {len(data['all_vt_ids'])} variants loaded", flush=True)
    return data


def load_sahni_fragoza_varchamp_pooled(use_wt_emb: bool = False) -> dict:
    """sahni_fragoza + VarChAMP_pooled; conflict labels dropped."""
    sf   = load_sahni_fragoza(use_wt_emb=use_wt_emb)
    pool = _load_varchamp_pooled_raw(use_wt_emb=use_wt_emb)

    sf_new_wt, sf_new_pairs = [], []
    for wt_id in sf["all_wt_ids"]:
        a, b = split_complex_id_hyphen(wt_id)
        sf_new_wt.append(f"{a}_{b}")
        sf_new_pairs.append((a, b))
    _remap_vt_ids(sf, sf_new_wt, sf_new_pairs)

    data = _dedup_and_merge([sf, pool], drop_conflicts=True)
    data["clusters"] = [str(c) for c in cluster_sequences(data["wt_seqs"])]
    return data


def load_sahni_fragoza_varchamp_full_pooled(use_wt_emb: bool = False) -> dict:
    """sahni_fragoza + VarChAMP2026 + VarChAMP1p + CAVA + VarChAMP_pooled."""
    sf   = load_sahni_fragoza(use_wt_emb=use_wt_emb)
    vc26 = _load_varchamp2026_raw(use_wt_emb=use_wt_emb)
    vc1p = _load_varchamp1p_raw(use_wt_emb=use_wt_emb)
    cava = _load_cava_raw(use_wt_emb=use_wt_emb)
    pool = _load_varchamp_pooled_raw(use_wt_emb=use_wt_emb)

    sf_new_wt, sf_new_pairs = [], []
    for wt_id in sf["all_wt_ids"]:
        a, b = split_complex_id_hyphen(wt_id)
        sf_new_wt.append(f"{a}_{b}")
        sf_new_pairs.append((a, b))
    _remap_vt_ids(sf, sf_new_wt, sf_new_pairs)

    data = _dedup_and_merge([sf, vc26, vc1p, cava, pool], drop_conflicts=True)
    data["clusters"] = [str(c) for c in cluster_sequences(data["wt_seqs"])]
    return data


def load_sahni_varchamp1p_cava(use_wt_emb: bool = False) -> dict:
    """Canonical vt_ids file stores NP_ for sahni and raw gene+orf for vc/cava."""
    sahni = load_sahni(use_wt_emb=use_wt_emb)
    vc    = _load_varchamp1p_raw(use_wt_emb=use_wt_emb)
    cava  = _load_cava_raw(use_wt_emb=use_wt_emb)

    with open(str(_HOME_DIR / "sahni" / "refseq_to_symbol.pkl"), "rb") as f:
        refseq_to_symbol = pickle.load(f)

    # For dedup comparison: convert sahni NP_ → gene_symbol (NP_ ids never match
    # raw varchamp1p/cava gene+orf ids, so cross-source dedup is effectively a no-op;
    # within-source dedup uses original ids)
    sahni_vt_orig = list(sahni["all_vt_ids"])  # preserve original NP_ ids for merge
    sahni_wt_orig = list(sahni["all_wt_ids"])

    sahni_vt_norm = []
    for wt_id, vt_id in zip(sahni["all_wt_ids"], sahni["all_vt_ids"]):
        if wt_id.startswith("NP_"):
            refseq_id = "_".join(wt_id.split("_")[:2])
            partner = "_".join(wt_id.split("_")[2:])
            new_wt = f"{refseq_to_symbol[refseq_id]}_{partner}"
        else:
            new_wt = wt_id
        sahni_vt_norm.append(f"{new_wt} {vt_id.split(' ', 1)[1]}")

    # Use normalized ids only for dedup comparison
    sahni_norm = dict(sahni)
    sahni_norm["all_vt_ids"] = sahni_vt_norm

    data = _dedup_and_merge([sahni_norm, vc, cava])

    # Restore original NP_ ids for the sahni entries (replace normalized back)
    norm_to_orig = {n: o for n, o in zip(sahni_vt_norm, sahni_vt_orig)}
    data["all_vt_ids"] = [norm_to_orig.get(v, v) for v in data["all_vt_ids"]]
    wt_orig_map = {n: o for n, o in zip(sahni_vt_norm, sahni_wt_orig)}
    data["all_wt_ids"] = [wt_orig_map.get(v, data["all_wt_ids"][i])
                          for i, v in enumerate(data["all_vt_ids"])]

    data["clusters"] = [str(c) for c in cluster_sequences(data["wt_seqs"])]
    return data


def load_dataset(cfg: DatasetConfig, use_wt_emb: bool = False) -> dict:
    loaders = {
        "sahni":                               load_sahni,
        "sahni_fragoza":                       load_sahni_fragoza,
        "sahni_fragoza_varchamp1p_cava":       load_sahni_fragoza_varchamp1p_cava,
        "sahni_varchamp1p_cava":               load_sahni_varchamp1p_cava,
        "sahni_fragoza_varchamp_full":         load_sahni_fragoza_varchamp_full,
        "sahni_fragoza_varchamp_pooled":       load_sahni_fragoza_varchamp_pooled,
        "sahni_fragoza_varchamp_full_pooled":  load_sahni_fragoza_varchamp_full_pooled,
    }
    data = loaders[cfg.name](use_wt_emb=use_wt_emb)
    print(f"  {len(data['all_vt_ids'])} data points loaded", flush=True)
    return data


# ── split generation ──────────────────────────────────────────────────────────

def align_to_vt_ids(data: dict, canonical_path, clusters_path=None) -> dict:
    """Restore the canonical row set and ordering from a reference vt_id list.

    Each loaded row is placed at its position in the canonical list, so the result
    is independent of directory listing order and of any rows the source has
    gained since the reference was written (those are dropped). This is what makes
    a run reproduce published numbers exactly; deriving the order from a fresh
    shuffle cannot, because both the row set and the pre-shuffle order can drift.

    Clusters are the GroupKFold groups and cd-hit is order-dependent (it assigns
    representatives greedily in input order), so they are never recomputed here.
    A saved canonical clusters file is used when available; otherwise the clusters
    computed at load time are carried through, which reproduces the original
    behaviour exactly as long as the source data has not grown.
    """
    with open(canonical_path, "rb") as f:
        canonical = pickle.load(f)
    pos = {vt_id: i for i, vt_id in enumerate(canonical)}
    n = len(canonical)

    valid_loaded: list[int] = []
    dest_positions: list[int] = []
    for i, vt_id in enumerate(data["all_vt_ids"]):
        if vt_id in pos:
            valid_loaded.append(i)
            dest_positions.append(pos[vt_id])

    if len(valid_loaded) != n:
        raise ValueError(
            f"{n - len(valid_loaded)} of {n} rows in {Path(canonical_path).name} were "
            f"not found in the loaded dataset ({len(data['all_vt_ids'])} rows). The "
            f"reference and the source data are inconsistent."
        )
    assert len(set(dest_positions)) == len(dest_positions), "duplicate vt_id mapping"

    dropped = len(data["all_vt_ids"]) - n
    if dropped:
        print(f"  align_to_vt_ids: {len(data['all_vt_ids'])} loaded -> {n} "
              f"(dropped {dropped} not in reference)", flush=True)

    ordered: dict = {k: [None] * n for k in _DATA_KEYS}
    for loaded_i, dest_i in zip(valid_loaded, dest_positions):
        for k in _DATA_KEYS:
            ordered[k][dest_i] = data[k][loaded_i]

    if clusters_path is not None and Path(clusters_path).exists():
        with open(clusters_path, "rb") as f:
            saved = pickle.load(f)
        if len(saved) != n:
            raise ValueError(f"{Path(clusters_path).name} has {len(saved)} clusters, "
                             f"expected {n}")
        ordered["clusters"] = [str(c) for c in saved]
    else:
        ordered["clusters"] = [None] * n
        for loaded_i, dest_i in zip(valid_loaded, dest_positions):
            ordered["clusters"][dest_i] = data["clusters"][loaded_i]
    return ordered


def shuffle_data(data: dict, seed: int = 42) -> dict:
    """Apply a fixed permutation to produce a canonical data ordering.

    Only used when no canonical reference exists for a dataset (i.e. when
    establishing one). Prefer align_to_vt_ids for anything reproducing results.
    """
    missing = [k for k in _DATA_KEYS if k not in data]
    if missing:
        raise KeyError(
            f"Loaded data is missing {missing}. A --data-cache written before "
            "'all_pairs' was added is stale: delete it and let it rebuild. The "
            "interactor/partner pair is recorded at load time and cannot be "
            "reliably recovered from complex_id strings afterwards."
        )
    n = len(data["all_vt_ids"])
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    result: dict = {}
    for k in _DATA_KEYS:
        result[k] = [data[k][i] for i in indices]
    result["clusters"] = [data["clusters"][i] for i in indices]
    return result


def make_fold_splits(data: dict, gcv_seed: int, n_splits: int = 10,
                     cfg: Optional[DatasetConfig] = None) -> list:
    """Fold splits for one GCV seed.

    When the canonical saved splits are present they are returned directly —
    a run is deterministic with respect to the original published run.

    When splits must be generated, the loader's cd-hit clusters (on the full
    complex sequence) are used as the GroupKFold groups.
    """
    if cfg is not None:
        saved = canonical_fold_splits_path(cfg, gcv_seed)
        if saved is not None and saved.exists():
            with open(saved, "rb") as f:
                fold_splits = pickle.load(f)
            total = sum(len(te) for _, _, te in fold_splits)
            if total != len(data["all_vt_ids"]):
                raise ValueError(
                    f"{saved.name} covers {total} rows but the aligned dataset has "
                    f"{len(data['all_vt_ids'])}")
            return fold_splits

    n = len(data["all_vt_ids"])
    kf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=gcv_seed)
    return [
        (fold, train_idx, test_idx)
        for fold, (train_idx, test_idx) in enumerate(
            kf.split(range(n), groups=data["clusters"])
        )
    ]


def compute_pair_test_classes(data: dict, fold_splits: list) -> np.ndarray:
    """Classify each test sample as C1/C2/C3 and return in fold order.

    C1: both proteins seen in training; C2: one seen; C3: neither seen.
    Result is a flat array concatenated across folds (fold 0 first), matching
    the layout of all_preds/all_labels in the GCV loop.

    Uses all_pairs, recorded at load time by each dataset's own id_parts_fn, so
    the interactor/partner split is never re-derived from complex_id strings
    (which is unreliable: separators vary per source, gene names may contain
    hyphens, and combined datasets mix conventions in a single dataset).
    """
    pairs = data["all_pairs"]
    classes: list[int] = []
    for _fold, train_idx, test_idx in fold_splits:
        train_proteins: set = set()
        for i in train_idx:
            a, b = pairs[i]
            train_proteins.add(a)
            train_proteins.add(b)
        for i in test_idx:
            a, b = pairs[i]
            a_seen = a in train_proteins
            b_seen = b in train_proteins
            classes.append(1 if (a_seen and b_seen) else 2 if (a_seen or b_seen) else 3)
    return np.array(classes, dtype=np.int64)


# ── training ──────────────────────────────────────────────────────────────────

def train_fold(
    train_val_idx, test_idx, fold,
    X, edge_mats, pos_labels, neg_labels, clusters,
    mut_diffs_raw, seq_lengths,
    device: torch.device,
    ablation: str = "full",
    seed: int = 0,
    prefit_scaler=None,
    precomputed_diffs=None,   # pre-scaled diffs (non-scratch); None triggers per-fold fit
    X_t: Optional[list] = None,      # pre-built CPU float tensors for node features
    edge_t: Optional[list] = None,   # pre-built CPU COO edge_index tensors
    batch_size: int = 16,
    lr: float = 0.001,
    lr_patience: int = 3,
    es_patience: int = 5,
    n_epochs: int = 100,
) -> Tuple[list, list]:
    # ── reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_amp = device.type == "cuda"
    print(f"Fold {fold}  ablation={ablation}  seed={seed}  amp={use_amp}", flush=True)

    # ── scale mutation diffs ──────────────────────────────────────────────────
    if ablation == "scratch":
        # ablation (1): fit scaler on training fold only — no prefit scaler
        fold_scaler = StandardScaler()
        fold_scaler.fit(np.array([mut_diffs_raw[j] for j in train_val_idx]))
        mutation_site_diffs = fold_scaler.transform(np.array(mut_diffs_raw))
    else:
        # Use precomputed array (identical result every fold for a fixed scaler)
        mutation_site_diffs = precomputed_diffs

    # ── build model ───────────────────────────────────────────────────────────
    def _load_ckpt(path, mdl):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "If this is the MegaScale pretrain checkpoint, either download it "
                "into weights/ (see Zenodo, docs/DATA_SOURCES.md) or generate it "
                "from scratch: see 'Stability Pretraining' in docs/TRAINING.md "
                "(preprocess_stability_data.py + pretrain_stability.py)."
            )
        ckpt = torch.load(path, map_location=device)
        mdict = mdl.state_dict()
        mdl.load_state_dict(
            {k: v for k, v in ckpt.items() if k in mdict and mdict[k].shape == v.shape},
            strict=False,
        )

    if ablation in ("no-gat", "megascale_all_no-gat"):
        model = GAT_mut_processor_no_gat().to(device)
        if ablation == "megascale_all_no-gat":
            # transfers mutation_diff_processor weights; other layers absent or shape-mismatched
            _load_ckpt(_MEGASCALE_PRETRAINED_PATH, model)
    elif ablation in ("no-mut", "megascale_all_no-mut"):
        model = GAT_mut_processor_no_mut(input_dim=X[0].shape[1]).to(device)
        if ablation == "megascale_all_no-mut":
            # transfers complex_gat2 weights; binding_predictor input dim differs so skipped
            _load_ckpt(_MEGASCALE_PRETRAINED_PATH, model)
    else:
        model = GAT_mut_processor(input_dim=X[0].shape[1])

        # ── checkpoint loading ─────────────────────────────────────────────
        # Note: complex_gat1 input_dim differs between monomer (1024) and PPI
        # complex (2048), so its weights are skipped by shape-filtering regardless
        # of the checkpoint source.  Layers that *do* transfer: mutation_diff_processor,
        # complex_gat2, binding_predictor.
        if ablation in ("full", "full_all"):
            _load_ckpt(_V1_0_PRETRAINED_PATH, model)
        elif ablation in ("megascale", "megascale_freeze_diff", "megascale_all",
                          "megascale_head", "megascale_all_wt-emb"):
            _load_ckpt(_MEGASCALE_PRETRAINED_PATH, model)
        # scratch / wt-emb: random init

        model = model.to(device)

        # ── freeze strategy ────────────────────────────────────────────────
        if ablation in ("full", "megascale"):
            # Freeze the large embedding-projection layer; fine-tune everything else.
            for param in model.parameters():
                param.requires_grad = False
            for param in model.mutation_diff_processor[-1].parameters():
                param.requires_grad = True
            for param in model.binding_predictor.parameters():
                param.requires_grad = True
            for param in model.complex_gat1.parameters():
                param.requires_grad = True
            for param in model.complex_gat2.parameters():
                param.requires_grad = True
        elif ablation == "megascale_freeze_diff":
            # Keep the entire learned mutation representation frozen; only adapt
            # the structural (GAT) and prediction layers.
            for param in model.parameters():
                param.requires_grad = False
            for param in model.binding_predictor.parameters():
                param.requires_grad = True
            for param in model.complex_gat1.parameters():
                param.requires_grad = True
            for param in model.complex_gat2.parameters():
                param.requires_grad = True
        elif ablation == "megascale_head":
            # Freeze everything; only retrain the classification head.
            for param in model.parameters():
                param.requires_grad = False
            for param in model.binding_predictor.parameters():
                param.requires_grad = True
        # full_all / megascale_all / scratch / wt-emb: all params trainable

    # AMP GradScaler (no-op when use_amp=False)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    num_mut_residues_list = [lengths[0] for lengths in seq_lengths]

    y_test      = [1 if pos_labels[j] else 0 for j in test_idx]
    test_pos    = [pos_labels[j] for j in test_idx]
    test_neg    = [neg_labels[j] for j in test_idx]
    test_nmut   = [num_mut_residues_list[j] for j in test_idx]

    # Inner train/val split via GroupKFold
    inner_kf = GroupKFold(n_splits=9, shuffle=True)
    inner_clusters = [clusters[j] for j in train_val_idx]
    train_rel, val_rel = next(inner_kf.split(range(len(train_val_idx)), groups=inner_clusters))
    train_idx = train_val_idx[train_rel]
    val_idx   = train_val_idx[val_rel]

    assert set(test_idx).isdisjoint(train_idx)
    assert set(test_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(val_idx)

    def _graphs_edges_diffs(indices):
        # Keep tensors on CPU (pinned); move per-sample to device in the training loop.
        # Pinned memory enables truly async PCIe DMA via non_blocking=True,
        # overlapping transfers with GPU compute (avoids loading all N graphs to GPU at
        # once which causes OOM for large datasets like sfvc).
        if X_t is not None:
            g = [X_t[j] for j in indices]   # already pinned from precompute
            e = [edge_t[j] for j in indices] # already pinned from precompute
        else:
            g = [torch.tensor(X[j], dtype=torch.float).pin_memory() for j in indices]
            e = [dense_to_sparse(torch.tensor(edge_mats[j]))[0].pin_memory() for j in indices]
        d = [torch.tensor(mutation_site_diffs[j], dtype=torch.float).pin_memory() for j in indices]
        return g, e, d

    train_g, train_e, train_d = _graphs_edges_diffs(train_idx)
    val_g,   val_e,   val_d   = _graphs_edges_diffs(val_idx)

    y_train   = [1 if pos_labels[j] else 0 for j in train_idx]
    y_val     = [1 if pos_labels[j] else 0 for j in val_idx]
    train_pos = [pos_labels[j] for j in train_idx]
    val_pos   = [pos_labels[j] for j in val_idx]
    train_neg = [neg_labels[j] for j in train_idx]
    val_neg   = [neg_labels[j] for j in val_idx]
    train_nm  = [num_mut_residues_list[j] for j in train_idx]
    val_nm    = [num_mut_residues_list[j] for j in val_idx]

    # Pre-build label tensors once per fold — eliminates per-sample GPU allocation
    y_train_t = torch.tensor(y_train, dtype=torch.float, device=device)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float, device=device)

    num_pos = sum(y_train)
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos if num_pos > 0 else 1.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=lr_patience, min_lr=1e-7)

    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_loss  = float("inf")
    patience_ctr = 0

    for epoch in range(n_epochs):
        model.train()
        shuffled = list(range(len(train_g)))
        random.shuffle(shuffled)
        total_loss = 0.0
        logits_buf: list = []
        targets_buf: list = []

        for idx, i in enumerate(shuffled):
            mut_idx = train_pos[i][0] if train_pos[i] else train_neg[i][0]
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(train_g[i].to(device, non_blocking=True),
                            train_e[i].to(device, non_blocking=True),
                            mut_idx, train_nm[i],
                            train_d[i].to(device, non_blocking=True))
            logits_buf.append(out.squeeze())
            targets_buf.append(y_train_t[i])

            if (idx + 1) % batch_size == 0 or idx == len(shuffled) - 1:
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = loss_fn(torch.stack(logits_buf), torch.stack(targets_buf))
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                total_loss += loss.item()
                logits_buf, targets_buf = [], []

        print(f"fold {fold} Epoch {epoch + 1}: Train loss: {total_loss:.4f}", flush=True)

        model.eval()
        val_loss = 0.0
        vlogits_buf: list = []
        vtargets_buf: list = []

        with torch.no_grad():
            for vi, i in enumerate(range(len(val_g))):
                mut_idx = val_pos[i][0] if val_pos[i] else val_neg[i][0]
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(val_g[i].to(device, non_blocking=True),
                                val_e[i].to(device, non_blocking=True),
                                mut_idx, val_nm[i],
                                val_d[i].to(device, non_blocking=True))
                vlogits_buf.append(out.squeeze())
                vtargets_buf.append(y_val_t[i])

                if (vi + 1) % batch_size == 0 or vi == len(val_g) - 1:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        val_loss += loss_fn(
                            torch.stack(vlogits_buf), torch.stack(vtargets_buf)
                        ).item()
                    vlogits_buf, vtargets_buf = [], []

        # Early stopping check before scheduler.step — matches mutpredppi.py exactly
        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            print(f"fold {fold} New best model (loss: {val_loss:.4f})", flush=True)
        else:
            patience_ctr += 1
            if patience_ctr >= es_patience:
                print(f"Fold {fold} early stopping at epoch {epoch + 1}", flush=True)
                break

        scheduler.step(val_loss)
        print(f"Fold {fold}, Epoch {epoch + 1}, val loss: {val_loss:.4f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()

    test_g, test_e, test_d = _graphs_edges_diffs(test_idx)
    fold_preds, fold_labels = [], []
    with torch.no_grad():
        for i in range(len(test_g)):
            mut_idx = test_pos[i][0] if test_pos[i] else test_neg[i][0]
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(test_g[i].to(device, non_blocking=True),
                            test_e[i].to(device, non_blocking=True),
                            mut_idx, test_nmut[i],
                            test_d[i].to(device, non_blocking=True))
            fold_preds.append(torch.sigmoid(out).squeeze().cpu().item())
            fold_labels.append(float(y_test[i]))

    del model, optimizer, scheduler, amp_scaler
    gc.collect()
    torch.cuda.empty_cache()

    return fold_preds, fold_labels


# ── AUC reporting ─────────────────────────────────────────────────────────────

def _compute_class_aucs(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    pair_test_classes: np.ndarray,
    fold_n_test: list,
) -> Tuple[list, np.ndarray, dict]:
    micro_auc = []
    for ptc in (1, 2, 3):
        mask = pair_test_classes == ptc
        print(f"  c{ptc}: {mask.sum()} preds", flush=True)
        if mask.sum() > 0 and len(np.unique(all_labels[mask])) > 1:
            micro_auc.append(roc_auc_score(all_labels[mask], all_preds[mask]))
        else:
            micro_auc.append(float("nan"))

    class_auc_avgs = np.zeros(3)
    class_counts   = np.zeros(3, dtype=int)
    fold_results: dict = {}

    curr_idx = 0
    for fold, n_test in enumerate(fold_n_test):
        preds  = all_preds[curr_idx:curr_idx + n_test]
        labels = all_labels[curr_idx:curr_idx + n_test]
        ptcs   = pair_test_classes[curr_idx:curr_idx + n_test]
        curr_idx += n_test

        fold_res = {}
        print(f"\nfold {fold}", flush=True)
        for ptc in (1, 2, 3):
            mask  = ptcs == ptc
            cp, cl = preds[mask], labels[mask]
            n_pos, n_neg = int((cl == 1).sum()), int((cl == 0).sum())
            fold_res[f"class_{ptc}"] = {"preds": cp, "labels": cl, "auc": None}
            if n_pos > 0 and n_neg > 0:
                auc = roc_auc_score(cl, cp)
                fold_res[f"class_{ptc}"]["auc"] = auc
                class_auc_avgs[ptc - 1] += len(cp) * auc
                class_counts[ptc - 1]   += len(cp)
                print(f"  c{ptc} (n={len(cp)}): AUC-ROC={auc:.4f}", flush=True)
            else:
                print(f"  c{ptc} (n={len(cp)}): SKIPPED (pos={n_pos}, neg={n_neg})",
                      flush=True)
        fold_results[fold] = fold_res

    macro_auc = np.where(class_counts > 0, class_auc_avgs / class_counts, np.nan)
    return micro_auc, macro_auc, fold_results


# ── main GCV loop — mirrors esignet_gcv_iter.py ───────────────────────────────

def run(args: argparse.Namespace) -> None:
    # Global reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}  ablation: {args.ablation}  seed: {args.seed}", flush=True)

    cfg    = DATASET_CONFIGS[args.dataset]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── load and align dataset ────────────────────────────────────────────────
    use_wt_emb = args.ablation in ("wt-emb", "megascale_all_wt-emb")
    print(f"Loading dataset: {cfg.name}  use_wt_emb={use_wt_emb}", flush=True)
    if args.data_cache and Path(args.data_cache).exists():
        print(f"Loading cached data from {args.data_cache}", flush=True)
        with open(args.data_cache, "rb") as _f:
            data = pickle.load(_f)
    else:
        data = load_dataset(cfg, use_wt_emb=use_wt_emb)
        if args.data_cache:
            Path(args.data_cache).parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving data cache to {args.data_cache}", flush=True)
            with open(args.data_cache, "wb") as _f:
                pickle.dump(data, _f)
            print(f"Data cache saved.", flush=True)

    canonical = canonical_vt_ids_path(cfg)
    if canonical.exists():
        ordered = align_to_vt_ids(data, canonical, canonical_clusters_path(cfg))
    else:
        print(f"  no canonical ordering at {canonical}; deriving one by shuffle "
              f"(results will not match a published run)", flush=True)
        ordered = shuffle_data(data)
    print(f"  {len(ordered['all_vt_ids'])} rows", flush=True)

    # Prefit scaler — megascale ablations use the MegaScale scaler; all others use
    # the FoldX scaler; scratch fits per fold from training data.
    _MEGASCALE_ABLATIONS = {
        "megascale", "megascale_freeze_diff", "megascale_all", "megascale_head",
        "megascale_all_no-gat", "megascale_all_no-mut", "megascale_all_wt-emb",
    }
    prefit_scaler = None
    if args.ablation in _MEGASCALE_ABLATIONS:
        prefit_scaler = joblib.load(_MEGASCALE_SCALER_PATH)
    elif args.ablation != "scratch":
        prefit_scaler = joblib.load(_V1_0_SCALER_PATH)

    X             = ordered["prott5_embeddings"]
    edge_mats     = ordered["edge_mats"]
    pos_labels    = ordered["pos_labels"]
    neg_labels    = ordered["neg_labels"]
    seq_lengths   = ordered["seq_lengths"]
    clusters      = ordered["clusters"]
    mut_diffs_raw = ordered["mutation_site_diffs"]

    # ── precompute tensors once (avoids 900× repeated conversions per 30-seed run) ──
    # Pin to page-locked memory so PCIe DMA can overlap with GPU compute
    # (non_blocking=True in the training loops then gives true async transfer).
    print("Precomputing graph tensors (CPU, pinned)...", flush=True)
    X_t    = [torch.tensor(x, dtype=torch.float).pin_memory() for x in X]
    edge_t = [dense_to_sparse(torch.tensor(e))[0].pin_memory() for e in edge_mats]
    print(f"  done ({len(X_t)} graphs)", flush=True)

    # Precompute scaled diffs once for non-scratch ablations (result is identical
    # every GCV seed since the scaler and data are fixed)
    precomputed_diffs = None
    if prefit_scaler is not None:
        precomputed_diffs = prefit_scaler.transform(np.array(mut_diffs_raw))

    # ── GCV iterations ────────────────────────────────────────────────────────
    ablation_tag = f"_{args.ablation}" if args.ablation != "full" else ""
    stem = f"MutPredPPI_{cfg.name}{ablation_tag}"

    macro_aucs: list = []
    micro_aucs: list = []
    detailed_results: dict = {"iterations": {}}
    start_gcv = 0

    ckpt_path = outdir / f"{stem}_detailed_results.pkl"
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            detailed_results = pickle.load(f)
        completed = sorted(detailed_results["iterations"].keys())
        if completed:
            start_gcv = max(completed) + 1
            micro_aucs = list(np.load(outdir / f"{stem}_micro_aucs.npy"))
            macro_aucs = list(np.load(outdir / f"{stem}_macro_aucs.npy"))
            print(f"Resuming: {len(completed)} GCV seeds done, starting from seed {start_gcv}", flush=True)

    for gcv_seed in range(start_gcv, args.n_gcv):
        print(f"\n{'='*60}", flush=True)
        print(f"GCV seed {gcv_seed}/{args.n_gcv - 1}", flush=True)

        fold_splits = make_fold_splits(ordered, gcv_seed, cfg=cfg)
        fold_n_test = [len(test_idx) for _, _, test_idx in fold_splits]
        pair_test_classes = compute_pair_test_classes(ordered, fold_splits)

        all_preds:  list = []
        all_labels: list = []

        for fold, train_idx, test_idx in fold_splits:
            # Unique seed per (base_seed, gcv_seed, fold) for full reproducibility
            fold_seed = args.seed * 10000 + gcv_seed * 100 + fold
            print(
                f"\nFold {fold}: {len(train_idx)} train / {len(test_idx)} test — "
                f"training MutPredPPI...",
                flush=True,
            )
            preds, labels = train_fold(
                train_idx, test_idx, fold,
                X, edge_mats, pos_labels, neg_labels, clusters,
                mut_diffs_raw, seq_lengths, device,
                ablation=args.ablation,
                seed=fold_seed,
                prefit_scaler=prefit_scaler,
                precomputed_diffs=precomputed_diffs,
                X_t=X_t,
                edge_t=edge_t,
            )
            all_preds.extend(preds)
            all_labels.extend(labels)
            print(f"  Fold {fold} done", flush=True)
            torch.cuda.empty_cache()
            gc.collect()

        all_preds_arr  = np.array(all_preds)
        all_labels_arr = np.array(all_labels)

        print(f"\nGCV seed {gcv_seed} — per-class AUROCs:", flush=True)
        micro_auc, macro_auc, fold_results = _compute_class_aucs(
            all_preds_arr, all_labels_arr, pair_test_classes, fold_n_test)
        print(f"micro AUC (c1/c2/c3): {micro_auc}", flush=True)
        print(f"macro AUC (c1/c2/c3): {macro_auc}", flush=True)

        micro_aucs.append(micro_auc)
        macro_aucs.append(macro_auc)
        detailed_results["iterations"][gcv_seed] = {
            "folds":     fold_results,
            "micro_auc": micro_auc,
            "macro_auc": macro_auc,
        }

        # ── checkpoint after each GCV iteration ───────────────────────────
        np.save(outdir / f"{stem}_micro_aucs.npy", np.array(micro_aucs))
        np.save(outdir / f"{stem}_macro_aucs.npy", np.array(macro_aucs))
        ckpt_path = outdir / f"{stem}_detailed_results.pkl"
        with open(ckpt_path, "wb") as f:
            pickle.dump(detailed_results, f)
        print(f"  [ckpt] saved after GCV seed {gcv_seed} → {ckpt_path}", flush=True)

    print(f"\nAll {args.n_gcv} GCV seeds complete. Results in {outdir}/", flush=True)
    print(f"  {stem}_micro_aucs.npy  shape={np.array(micro_aucs).shape}", flush=True)
    print(f"  {stem}_macro_aucs.npy  shape={np.array(macro_aucs).shape}", flush=True)
    print(f"  {stem}_detailed_results.pkl", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MutPred-PPI GCV training")
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS))
    p.add_argument("--device", default="",
                   help="PyTorch device string (e.g. 'cuda:0'). Defaults to auto-detect.")
    p.add_argument("--n-gcv", type=int, default=30,
                   help="Number of GCV iterations (default: 30)")
    p.add_argument("--outdir", default=str(_REPO_ROOT / "results_revisions" / "macro_aucs"),
                   help="Output directory for results (default: results_revisions/macro_aucs, "
                        "where the figure scripts look for them)")
    p.add_argument("--ablation", default="megascale_all",
                   choices=[
                       "full", "full_all",
                       "megascale", "megascale_freeze_diff", "megascale_all", "megascale_head",
                       "megascale_all_no-gat", "megascale_all_no-mut", "megascale_all_wt-emb",
                       "scratch", "no-gat", "no-mut", "wt-emb",
                   ],
                   help=(
                       "Ablation mode (default: full). "
                       # ── old pretrained (FoldX stability pretrain) ────────────────────
                       "'full': old pretrained weights; freeze mut_diff[0], fine-tune rest. "
                       "'full_all': old pretrained weights; fine-tune all params. "
                       # ── megascale (ddG regression pretrain) ─────────────────────────
                       "'megascale': MegaScale pretrain; same freeze strategy as 'full'. "
                       "'megascale_freeze_diff': MegaScale pretrain; freeze all mut_diff_processor, "
                       "fine-tune GAT + binding_predictor only. "
                       "'megascale_all': MegaScale pretrain; fine-tune all params. "
                       "'megascale_head': MegaScale pretrain; freeze everything, "
                       "fine-tune binding_predictor only. "
                       # ── megascale_all architectural ablations ────────────────────────
                       "'megascale_all_no-gat': MegaScale pretrain into no-gat arch; all trainable. "
                       "'megascale_all_no-mut': MegaScale pretrain into no-mut arch; all trainable. "
                       "'megascale_all_wt-emb': MegaScale pretrain; all trainable; WT interactor embs. "
                       # ── baselines ────────────────────────────────────────────────────
                       "'scratch': random init, all params trainable, scaler fit per fold. "
                       "'no-gat': mutation diff processor + predictor only (no graph). "
                       "'no-mut': structural GAT + predictor only (no mutation processor). "
                       "'wt-emb': full model but interactor uses WT embeddings on the graph."
                   ))
    p.add_argument("--seed", type=int, default=0,
                   help="Base random seed for reproducibility (default: 0)")
    p.add_argument("--data-cache", default=None,
                   help="Path to cache the loaded dataset dict (pkl). "
                        "Saves ~10h reload time on subsequent runs.")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
