#!/usr/bin/env python
"""Train a MutPred-PPI checkpoint for 1 epoch on the tiny real Sahni+Fragoza subset
built by run_example.sh, and save it to output/.

This script does NOT modify src/training/train_final_model.py or
src/evaluation/mutpred_ppi_cv.py. It imports the real, unmodified model class
(GAT_mut_processor) and the generic (non-hardcoded-path) data-loading helpers
(_build_emb_dict / _load_graphs / _gather_labels_pos_neg) from mutpred_ppi_cv.py,
and feeds them data built directly from this directory's tiny
structure/FASTA/CSV inputs. The train/val loop below is a short, standard
PyTorch loop (Adam + BCEWithLogitsLoss + early stopping on val loss) that
mirrors train_final_model.py's `_train_loop` / mutpred_ppi_cv.py's `train_fold`
almost line for line — it is reimplemented here (rather than imported) only
because of the import bug documented below, not because the real training
logic differs.

Why this indirection is necessary (see README.md "Known issues" section):
1. train_final_model.py's only data entry point, load_dataset(), calls dataset
   loaders in mutpred_ppi_cv.py (e.g. load_sahni_fragoza()) that hardcode
   absolute paths to this machine's internal, non-Zenodo-distributed pre-cached
   embeddings/graphs (e.g. /data/.../swing_train), and align_to_vt_ids()
   requires a canonical vt_ids pickle from another hardcoded internal path
   (/home/rcstewart/gnn/ppi_interaction_loss/cv_splits). Neither has a CLI or
   config override, and both paths are outside this repo and not part of the
   public release, so train_final_model.py cannot be pointed at a custom CSV.
2. Separately, and more fundamentally: `import train_final_model` currently
   raises `ImportError: cannot import name '_SCALER_PATH' from
   'mutpred_ppi_cv'` — train_final_model.py (line 41-53) imports
   `_SCALER_PATH` and `_PRETRAINED_PATH` from mutpred_ppi_cv.py, but that
   module only defines `_V1_0_SCALER_PATH` / `_V1_0_PRETRAINED_PATH` (and
   `_MEGASCALE_SCALER_PATH` / `_MEGASCALE_PRETRAINED_PATH`) — no such names
   exist. This means train_final_model.py cannot be imported OR run at all
   right now, for any dataset/ablation/CLI combination — this is a
   pre-existing bug independent of this example, not something introduced by
   the workaround here.
This script therefore imports only the pieces that DO work standalone
(GAT_mut_processor + the generic loader helpers, all from mutpred_ppi_cv.py,
which does not import from train_final_model.py and has no such bug).

Usage:
    python train_on_subset.py --device cpu|cuda:0 [--epochs 1]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import dense_to_sparse

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent

sys.path.insert(0, str(_REPO_ROOT / "src" / "evaluation"))
sys.path.insert(0, str(_REPO_ROOT / "src" / "inference"))

from mutpred_ppi_cv import (  # noqa: E402
    GAT_mut_processor, _build_emb_dict, _load_graphs, _gather_labels_pos_neg,
)
from utils.inference_utils import read_h5  # noqa: E402


def train_loop(
    train_idx, val_idx,
    X, edge_mats, pos_labels, neg_labels, mutation_site_diffs, seq_lengths,
    device, seed,
    X_t, edge_t,
    batch_size=4, lr=0.001, lr_patience=3, es_patience=5, n_epochs=1,
):
    """Train GAT_mut_processor from scratch (ablation='scratch': random init,
    all params trainable) on train_idx, validate on val_idx, return the
    best-val-loss state_dict. Standard Adam + BCEWithLogitsLoss(pos_weight)
    loop with early stopping — mirrors train_final_model.py's `_train_loop`
    and mutpred_ppi_cv.py's `train_fold` (see module docstring for why this
    is reimplemented here instead of imported)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_amp = device.type == "cuda"
    model = GAT_mut_processor(input_dim=X[0].shape[1]).to(device)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    num_mut_residues = [lengths[0] for lengths in seq_lengths]

    def _get_tensors(indices):
        g = [X_t[j] for j in indices]
        e = [edge_t[j] for j in indices]
        d = [torch.tensor(mutation_site_diffs[j], dtype=torch.float) for j in indices]
        return g, e, d

    train_g, train_e, train_d = _get_tensors(train_idx)
    val_g, val_e, val_d = _get_tensors(val_idx)

    train_pos = [pos_labels[j] for j in train_idx]
    train_neg = [neg_labels[j] for j in train_idx]
    train_nm = [num_mut_residues[j] for j in train_idx]
    val_pos = [pos_labels[j] for j in val_idx]
    val_neg = [neg_labels[j] for j in val_idx]
    val_nm = [num_mut_residues[j] for j in val_idx]

    y_train = [1 if pos_labels[j] else 0 for j in train_idx]
    y_val = [1 if pos_labels[j] else 0 for j in val_idx]
    y_train_t = torch.tensor(y_train, dtype=torch.float, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float, device=device)

    num_pos = sum(y_train)
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos if num_pos > 0 else 1.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=lr_patience, min_lr=1e-7)

    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_loss = float("inf")
    patience_ctr = 0

    for epoch in range(n_epochs):
        model.train()
        shuffled = list(range(len(train_g)))
        random.shuffle(shuffled)
        total_loss = 0.0
        logits_buf, targets_buf = [], []

        for idx, i in enumerate(shuffled):
            mut_idx = train_pos[i][0] if train_pos[i] else train_neg[i][0]
            g_i, e_i, d_i = train_g[i].to(device), train_e[i].to(device), train_d[i].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(g_i, e_i, mut_idx, train_nm[i], d_i)
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

        print(f"Epoch {epoch + 1}: train loss={total_loss:.4f}", flush=True)

        model.eval()
        val_loss = 0.0
        vlogits_buf, vtargets_buf = [], []
        with torch.no_grad():
            for vi, i in enumerate(range(len(val_g))):
                mut_idx = val_pos[i][0] if val_pos[i] else val_neg[i][0]
                g_i, e_i, d_i = val_g[i].to(device), val_e[i].to(device), val_d[i].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(g_i, e_i, mut_idx, val_nm[i], d_i)
                vlogits_buf.append(out.squeeze())
                vtargets_buf.append(y_val_t[i])
                if (vi + 1) % batch_size == 0 or vi == len(val_g) - 1:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        val_loss += loss_fn(
                            torch.stack(vlogits_buf), torch.stack(vtargets_buf)
                        ).item()
                    vlogits_buf, vtargets_buf = [], []

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            print(f"  new best (val loss={val_loss:.4f})", flush=True)
        else:
            patience_ctr += 1
            if patience_ctr >= es_patience:
                print(f"  early stop epoch {epoch + 1}", flush=True)
                break
        scheduler.step(val_loss)

    return best_state


def _create_variant_marker_files(graph_dir: str) -> None:
    """Create the per-variant '.labels' marker files _build_emb_dict() globs for,
    from the .interaction_loss_pos / .interaction_loss_neg files already present."""
    import glob
    import os

    n_created = 0
    for label_ext in ("interaction_loss_pos", "interaction_loss_neg"):
        for f_path in glob.glob(f"{graph_dir}/*.{label_ext}"):
            complex_id = os.path.basename(f_path).split(f".{label_ext}")[0]
            with open(f_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    variant = line.split("\t")[-1]
                    marker = f"{graph_dir}/{complex_id}_interaction_loss_variant_{variant}.labels"
                    if not os.path.exists(marker):
                        Path(marker).touch()
                        n_created += 1
    print(f"created {n_created} variant marker files")


def split_complex_id_underscore(complex_id: str):
    """This example's complex IDs are UNIPROT_UNIPROT (from 01_make_contact_graphs_and_fasta.py),
    not the hyphen-delimited IDs used by the internal sahni_fragoza cache (split_complex_id_hyphen
    in mutpred_ppi_cv.py), so we use our own simple splitter matching our own file naming."""
    parts = complex_id.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Could not split: {complex_id}")
    return parts[0], parts[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device: {device}")

    graph_dir = str(_HERE / "af3_graphs")
    t5_emb_dict = read_h5(str(_HERE / "wt_and_vt_t5_embs.h5"))
    print(f"loaded {len(t5_emb_dict)} ProtT5 embeddings")

    # `_build_emb_dict` (imported below) expects one marker file per variant,
    # named "{complex_id}...interaction_loss_variant_{variant}....labels"
    # (content unused — only the filename is parsed). 01_make_contact_graphs_
    # and_fasta.py (the public inference pipeline) only writes a single
    # combined all_variants.labels file, not these per-variant markers (that
    # format comes from a different, internal-only data-prep script) — so we
    # create them here from the .interaction_loss_pos/_neg files 01's script
    # (and our own label-fixup step in run_example.sh) already wrote.
    _create_variant_marker_files(graph_dir)

    # -- build training data using the real, generic (non-hardcoded-path) loader
    #    helpers from mutpred_ppi_cv.py --------------------------------------
    # _build_emb_dict also returns complex_pair_dict -- the (interactor, partner)
    # split recorded at load time, which _load_graphs needs to derive C1/C2/C3.
    emb_dict, len_dict, pair_dict = _build_emb_dict(
        graph_dir, t5_emb_dict, split_complex_id_underscore)
    data = _load_graphs(graph_dir, emb_dict, len_dict, pair_dict)
    pos_labels, neg_labels = _gather_labels_pos_neg(graph_dir, data["all_vt_ids"])
    data["pos_labels"] = pos_labels
    data["neg_labels"] = neg_labels

    n = len(data["all_vt_ids"])
    n_pos = sum(1 for p in pos_labels if p)
    n_neg = sum(1 for p in neg_labels if p)
    print(f"{n} labeled rows loaded ({n_pos} disrupted / {n_neg} maintained)")
    assert n >= 10, f"expected >=10 rows, got {n}"

    # clusters: one cluster per complex (protein pair). This example's 40 rows
    # are already 40 distinct complexes, so this is equivalent in spirit to the
    # CD-HIT sequence-identity clustering train_final_model.py normally uses
    # (cluster_sequences()) for GroupKFold grouping, without requiring the
    # cd-hit binary (hardcoded to a different, non-"ppi" conda env in the
    # original pipeline: /home/rcstewart/miniconda3/envs/pytorch_env/bin/cd-hit).
    clusters = list(data["all_wt_ids"])

    X = data["prott5_embeddings"]
    edge_mats = data["edge_mats"]
    seq_lengths = data["seq_lengths"]
    mut_diffs_raw = data["mutation_site_diffs"]

    X_t = [torch.tensor(x, dtype=torch.float) for x in X]
    edge_t = [dense_to_sparse(torch.tensor(e))[0] for e in edge_mats]

    all_idx = np.arange(n)

    # -- ablation="scratch": fit a StandardScaler on this tiny dataset's own
    #    mutation-site diffs (matches train_final_model.py's train_all_data()
    #    behavior for --ablation scratch; no external pretrained checkpoint or
    #    scaler needed) --------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(np.array(mut_diffs_raw))
    mutation_site_diffs = scaler.transform(np.array(mut_diffs_raw))

    n_splits = min(9, len(set(clusters)))
    inner_kf = GroupKFold(n_splits=n_splits, shuffle=True)
    train_rel, val_rel = next(inner_kf.split(range(n), groups=clusters))
    train_idx, val_idx = all_idx[train_rel], all_idx[val_rel]
    print(f"train/val split: {len(train_idx)} train, {len(val_idx)} val ({n_splits}-group split)")

    best_state = train_loop(
        train_idx, val_idx,
        X, edge_mats, pos_labels, neg_labels, mutation_site_diffs, seq_lengths,
        device, seed=args.seed,
        X_t=X_t, edge_t=edge_t,
        batch_size=4, n_epochs=args.epochs, es_patience=args.epochs + 1,
    )

    out_dir = _HERE / "output"
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "MutPred-PPI_quickstart_scratch.pt"
    torch.save(best_state, save_path)
    print(f"\nSaved checkpoint: {save_path}")
    print(f"Checkpoint has {len(best_state)} state_dict tensors, "
          f"{sum(v.numel() for v in best_state.values())} total parameters")


if __name__ == "__main__":
    main()
