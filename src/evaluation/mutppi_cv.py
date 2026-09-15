#!/usr/bin/env python
"""MutPPI / MutPPI+ inference on the canonical 090826 datasets.

Ported from `~/ppi_lossgain/mutppi_scripts/mutppi_preds.py`, an external,
unversioned script wired entirely to the retired pipeline: a hardcoded
`_CV_DIR` pointing at `~/gnn/ppi_interaction_loss/cv_splits`, per-dataset
`*_all_vt_ids_and_labels.txt` label files (0-based positions, orphan **O1** in
`utils.legacy_guard`), unsuffixed `fold_splits.pkl`, and structures
resolved by trying `fold_{gene}_{gene}_model_0.pdb` filenames across three
separate PDB directories. None of that survives here -- see
`utils.legacy_guard`. Everything below reads exactly what
`saambe3d_cv.py` does: `utils.gcv_common.{DATASET_CONFIGS, load_data,
load_splits}` for rows and splits, and `utils.structures.Structures` for the
PDB (converted on demand from the canonical `.cif.gz` tree, chain id read
from the manifest -- never assumed to be "A").

MutPPI (`--model 0`, `GINGATRegressor`) and MutPPI+ (`--model 1`,
`FusionEnsembleRegressor`, GIN+GAT + ESM-2 650M sequence branch) are both
5-seed ensembles pretrained on SKEMPI2 S4169 -- not retrained here, exactly
like SAAMBE-3D. The model architecture and checkpoints come from the pinned
upstream checkout resolved by `paths.method_dir('mutppi')`; only
the input plumbing was replaced.

Usage (GCV, mirrors saambe3d_cv.py):
    conda run -n ppi python src/evaluation/mutppi_cv.py \\
        --dataset sahni_fragoza_mapped090826 --model 0 --outdir results/gcv/

Blind test: `run_varchamp_blind_test.py --method mutppi` /
`--method mutppiplus` call `score_rows()` directly.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent

from utils import mutations  # noqa: E402
from paths import GCV_RESULTS_DIR, method_dir  # noqa: E402
from utils.gcv_common import dataset_arg, dataset_config, DATASET_CHOICES, DATASET_CONFIGS, load_data, load_splits  # noqa: E402
from utils.legacy_guard import reject_legacy  # noqa: E402
from utils.structures import Structures  # noqa: E402

# ── upstream checkout (source + pretrained checkpoints; not repo data) ──
#
# Resolved lazily. Importing this module must not require the checkout to be
# present, or `--help`, the test suite and any downstream import all fail on a
# machine that has not cloned MutPPI. See paths.method_dir.
def _mutppi_dir():
    return method_dir("mutppi")


def _ckpt_dir():
    return _mutppi_dir() / "output" / "checkpoint"
_ESM2_PATH = "facebook/esm2_t33_650M_UR50D"
_ENSEMBLE_SEEDS = [34, 42, 1998, 2025, 3407]
_EPOCH = {0: 1000, 1: 150}
_AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _import_mutppi_source():
    """Load GINGATRegressor/FusionEnsembleRegressor from the pinned checkout.

    Deferred to call time (not module import time) so merely importing this
    file doesn't require the external checkout or its cwd-relative model
    paths, which only inference actually needs.
    """
    _dir = _mutppi_dir()
    sys.path.insert(0, str(_dir))
    os.chdir(_dir)  # MutPPI's own code resolves some paths relative to cwd
    from models.models import FusionEnsembleRegressor, GINGATRegressor  # noqa: E402
    return GINGATRegressor, FusionEnsembleRegressor


# ── PDB parsing + graph construction (unchanged from the external script) ──────

def parse_pdb_chain(pdb_path: Path, chain_id: str) -> tuple[str, np.ndarray]:
    """CA sequence and Cα coordinates for one chain."""
    residues: dict[int, tuple[str, list]] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA" or line[21].strip() != chain_id:
                continue
            aa1 = _AA3_TO_1.get(line[17:20].strip())
            if aa1 is None:
                continue
            res_seq = int(line[22:26].strip())
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            if res_seq not in residues:
                residues[res_seq] = (aa1, [x, y, z])
    if not residues:
        raise ValueError(f"No CA atoms found for chain {chain_id} in {pdb_path}")
    keys = sorted(residues)
    seq = "".join(residues[k][0] for k in keys)
    coords = np.array([residues[k][1] for k in keys], dtype=np.float32)
    return seq, coords


def _aa_to_onehot(seq: str) -> torch.Tensor:
    oh = torch.zeros(len(seq), len(_AA_LIST))
    for i, aa in enumerate(seq):
        if aa in _AA_LIST:
            oh[i, _AA_LIST.index(aa)] = 1.0
    return oh


def build_graph(seq: str, positions: np.ndarray):
    """One-hot node features + 7 Å Cα-Cα edges."""
    from scipy.spatial.distance import cdist
    from torch_geometric.data import Data
    from torch_geometric.utils import dense_to_sparse

    node_feat = _aa_to_onehot(seq)
    adj = (cdist(positions, positions).astype(np.float32) < 7.0).astype(np.float32)
    edge_index, _ = dense_to_sparse(torch.tensor(adj))
    return Data(x=node_feat, edge_index=edge_index, y=torch.tensor([0.0], dtype=torch.float))


def build_variant_graphs(pdb_path: Path, chain_a: str, chain_b: str, mutation: str):
    """(wt_graph, mt_graph, wt_seq_a, mt_seq_a) for one canonical 1-based mutation.

    `chain_a` is the INTERACTOR's chain (the one the mutation applies to);
    `chain_b` is the partner's, resolved from the same manifest entry --
    never assumed to be "A"/"B" respectively, unlike the external script this
    was ported from.
    """
    wt_seq_a, pos_a = parse_pdb_chain(pdb_path, chain_a)
    tgt_seq_b, pos_b = parse_pdb_chain(pdb_path, chain_b)

    wt_aa, pos_1based, mt_aa = mutations.parse(mutation)
    pos_0b = pos_1based - 1
    if not (0 <= pos_0b < len(wt_seq_a)):
        raise ValueError(f"{mutation}: pos {pos_1based} out of range (chain len={len(wt_seq_a)})")
    if wt_seq_a[pos_0b] != wt_aa:
        raise ValueError(
            f"{mutation}: expected {wt_aa} at 1-based {pos_1based}, got "
            f"{wt_seq_a[pos_0b]} in {pdb_path.name}")

    mt_seq_a = wt_seq_a[:pos_0b] + mt_aa + wt_seq_a[pos_0b + 1:]
    all_pos = np.concatenate([pos_a, pos_b], axis=0)
    wt_graph = build_graph(wt_seq_a + tgt_seq_b, all_pos)
    mt_graph = build_graph(mt_seq_a + tgt_seq_b, all_pos)
    return wt_graph, mt_graph, wt_seq_a, mt_seq_a


# ── model loading (cached; ensembles are 5 checkpoints each) ───────────────────

_MODEL_CACHE: dict[tuple[int, str], list] = {}


def _ckpt_path(model_type: int, seed: int) -> Path:
    if model_type == 0:
        stem = f"GINGATRegressor_Train-S4169_ReductionMode0-20_RandomSeed-{seed}"
    else:
        stem = (f"FusionEnsembleRegressor(finetune-1,ESM2-650M)_Train-S4169_"
                f"ReductionMode0-20_RandomSeed-{seed}")
    return _ckpt_dir() / f"{stem}_epoch{_EPOCH[model_type]}.model"


def load_ensemble(model_type: int, device: torch.device) -> list:
    key = (model_type, str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    GINGATRegressor, FusionEnsembleRegressor = _import_mutppi_source()

    def make_model():
        if model_type == 0:
            return GINGATRegressor(in_channels=20).to(device)
        return FusionEnsembleRegressor(
            in_channels=20, pretrained_model=_ESM2_PATH, hidden_size=1280, finetune=1
        ).to(device)

    models = []
    for seed in _ENSEMBLE_SEEDS:
        ckpt = _ckpt_path(model_type, seed)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                f"Train with (from {_mutppi_dir()}): "
                f"python train_model.py --Model {model_type} --RM 0 --reduction 20")
        reject_legacy(ckpt, check_mtime=False)  # pinned upstream checkpoint, not repo data
        m = make_model()
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        models.append(m)
    _MODEL_CACHE[key] = models
    return models


@torch.no_grad()
def predict_ddg(wt_graph, mt_graph, wt_seq_a: str, mt_seq_a: str,
                models: list, model_type: int, device) -> float:
    from torch_geometric.data import Batch

    wt_b = Batch.from_data_list([wt_graph]).to(device)
    mt_b = Batch.from_data_list([mt_graph]).to(device)
    preds = []
    for m in models:
        if model_type == 0:
            out = m(wt_b, mt_b).squeeze().cpu().item()
        else:
            ddg_ave, ddg_min, ddg_max = m(wt_b, mt_b, [wt_seq_a], [mt_seq_a])
            out = ((ddg_ave + ddg_min + ddg_max) / 3).squeeze().cpu().item()
        preds.append(out)
    return float(np.mean(preds))


# ── row scoring, shared by GCV and the blind test ──────────────────────────────

def score_rows(rows: pd.DataFrame, model_variant: int,
              device: str = "", pdb_cache: Path | None = None,
              max_rows: int | None = None) -> np.ndarray:
    """Score every row of a canonical table. `model_variant`: 0=MutPPI, 1=MutPPI+.

    `max_rows` stops after that many rows and leaves the rest NaN, which is the
    value this array already carries for a row with no structure. The array
    keeps its full length, so it stays aligned with the fold labels. Smoke
    tests only: MutPPI+ over sahni_fragoza is about 100 minutes.
    """
    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    models = load_ensemble(model_variant, dev)
    structures = Structures(pdb_cache=pdb_cache)

    scores = np.full(len(rows), np.nan, dtype=np.float32)
    skipped: Counter = Counter()
    for i, row in enumerate(rows.itertuples()):
        if max_rows is not None and i >= max_rows:
            break
        pdb_path, chain_a = structures.find_pdb(
            interactor=row.interactor_sequence, partner=row.partner_sequence)
        if pdb_path is None:
            skipped["no_structure"] += 1
            continue
        # Partner's chain id, from the same manifest entry (role-swapped lookup).
        _, chain_b = structures.find(
            interactor=row.partner_sequence, partner=row.interactor_sequence)
        try:
            wt_g, mt_g, wt_seq, mt_seq = build_variant_graphs(
                pdb_path, chain_a, chain_b or "B", row.mutation)
            scores[i] = predict_ddg(wt_g, mt_g, wt_seq, mt_seq, models, model_variant, dev)
        except Exception as exc:
            skipped["error"] += 1
            print(f"  ERROR {row.interactor} {row.mutation}: {exc}", flush=True)
    for reason, count in sorted(skipped.items()):
        print(f"  skipped, {reason}: {count}", flush=True)
    return scores


# ── GCV entry point (mirrors saambe3d_cv.py) ───────────────────────────────────

def run(args: argparse.Namespace) -> None:
    cfg = dataset_config(args.dataset)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_stem = "MutPPI" if args.model == 0 else "MutPPIPlus"
    out_npy = outdir / f"{cfg.name}_{model_stem}_preds.npy"
    if out_npy.exists() and not args.overwrite:
        print(f"Output already exists: {out_npy}  (use --overwrite to rerun)", flush=True)
        return

    rows = load_data(cfg)
    fold_splits, _ = load_splits(cfg, seed=args.seed)
    n_test_total = sum(len(test_idx) for _, _, test_idx in fold_splits)
    print(f"Dataset: {cfg.name}  model: {model_stem}  "
          f"rows: {len(rows)}  test: {n_test_total}", flush=True)

    # Every row appears in test exactly once across the fold splits (same
    # guarantee saambe3d_cv.py relies on), so scoring the whole table once
    # and never re-scoring is equivalent to iterating folds.
    scores = score_rows(rows, args.model, device=args.device,
                       pdb_cache=outdir / "_pdb_cache",
                       max_rows=args.max_rows)
    np.save(out_npy, scores)
    print(f"Saved: {out_npy}  shape={scores.shape}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MutPPI/MutPPI+ inference — 090826 canonical datasets")
    p.add_argument("--dataset", required=True, type=dataset_arg, choices=list(DATASET_CONFIGS))
    p.add_argument("--model", type=int, required=True, choices=(0, 1),
                   help="0 = MutPPI (GINGATRegressor), 1 = MutPPI+ (FusionEnsembleRegressor)")
    p.add_argument("--seed", type=int, default=0,
                   help="GCV split seed (default: 0; all rows appear in test exactly once)")
    p.add_argument("--device", default="")
    p.add_argument("--outdir", default=str(GCV_RESULTS_DIR),
                   help="Output directory (default: results/gcv/, matching every "
                        "other CV script). Was '.' until 2026-09-10, which put "
                        "this method's arrays in $CWD while its siblings wrote to "
                        "results/gcv/ -- so a plain invocation produced results the "
                        "figure scripts could not find.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Score at most this many rows and leave the rest NaN. "
                        "For smoke tests only.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
