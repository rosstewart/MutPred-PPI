#!/usr/bin/env python
"""Train final MutPred-PPI model weights and save per-fold checkpoints.

Data loading is imported from src/evaluation/mutpred_ppi_cv.py, which holds
all hardcoded dataset paths.  This script only adds checkpoint saving and the
--no-cv (all-data) training mode on top of the standard GCV infrastructure.

Usage — 10-fold ensemble (Option A):
    conda run -n ppi python src/training/train_final_model.py \\
        --dataset sahni_fragoza --ablation megascale_all --seed 1 \\
        --save-models-dir weights/folds/ --device cuda:0

Usage — single model on all data (Option B):
    conda run -n ppi python src/training/train_final_model.py \\
        --dataset sahni_fragoza --ablation megascale_all --seed 1 \\
        --save-models-dir weights/ --device cuda:0 --no-cv
"""

from __future__ import annotations

import argparse
import gc
import pickle
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import dense_to_sparse

# ── import data loading infrastructure from the CV script ────────────────────
_EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"
sys.path.insert(0, str(_EVAL_DIR))
from mutpred_ppi_cv import (  # noqa: E402
    DATASET_CONFIGS,
    load_dataset,
    shuffle_data,
    align_to_vt_ids,
    canonical_vt_ids_path,
    canonical_clusters_path,
    make_fold_splits,
    GAT_mut_processor,
    GAT_mut_processor_no_gat,
    GAT_mut_processor_no_mut,
    _V1_0_SCALER_PATH,
    _MEGASCALE_SCALER_PATH,
    _V1_0_PRETRAINED_PATH,
    _MEGASCALE_PRETRAINED_PATH,
)


# ── model building helper ─────────────────────────────────────────────────────

def _load_ckpt(path, mdl, device):
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


def _build_model(ablation: str, input_dim: int, device: torch.device) -> nn.Module:
    if ablation in ("no-gat", "megascale_all_no-gat"):
        model = GAT_mut_processor_no_gat().to(device)
        if ablation == "megascale_all_no-gat":
            _load_ckpt(_MEGASCALE_PRETRAINED_PATH, model, device)
    elif ablation in ("no-mut", "megascale_all_no-mut"):
        model = GAT_mut_processor_no_mut(input_dim=input_dim).to(device)
        if ablation == "megascale_all_no-mut":
            _load_ckpt(_MEGASCALE_PRETRAINED_PATH, model, device)
    else:
        model = GAT_mut_processor(input_dim=input_dim)
        if ablation in ("full", "full_all"):
            _load_ckpt(_V1_0_PRETRAINED_PATH, model, device)
        elif ablation in ("megascale", "megascale_freeze_diff", "megascale_all",
                          "megascale_head", "megascale_all_wt-emb"):
            _load_ckpt(_MEGASCALE_PRETRAINED_PATH, model, device)
        model = model.to(device)

        if ablation in ("full", "megascale"):
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
            for param in model.parameters():
                param.requires_grad = False
            for param in model.binding_predictor.parameters():
                param.requires_grad = True
            for param in model.complex_gat1.parameters():
                param.requires_grad = True
            for param in model.complex_gat2.parameters():
                param.requires_grad = True
        elif ablation == "megascale_head":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.binding_predictor.parameters():
                param.requires_grad = True

    return model


# ── training loop (with optional checkpoint saving) ───────────────────────────

def _train_loop(
    train_idx,
    val_idx,
    X, edge_mats, pos_labels, neg_labels,
    mutation_site_diffs, seq_lengths,
    device, ablation, seed,
    X_t, edge_t,
    batch_size=16, lr=0.001, lr_patience=3, es_patience=5, n_epochs=100,
    label_prefix="",
):
    """Train on train_idx, validate on val_idx; return best_state dict."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_amp = device.type == "cuda"
    model = _build_model(ablation, X[0].shape[1], device)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    num_mut_residues = [lengths[0] for lengths in seq_lengths]

    def _get_tensors(indices):
        # Keep on CPU; move per-sample during training to avoid OOM with large datasets
        if X_t is not None:
            g = [X_t[j] for j in indices]
            e = [edge_t[j] for j in indices]
        else:
            g = [torch.tensor(X[j], dtype=torch.float) for j in indices]
            e = [dense_to_sparse(torch.tensor(edge_mats[j]))[0] for j in indices]
        d = [torch.tensor(mutation_site_diffs[j], dtype=torch.float) for j in indices]
        return g, e, d

    train_g, train_e, train_d = _get_tensors(train_idx)
    val_g,   val_e,   val_d   = _get_tensors(val_idx)

    train_pos = [pos_labels[j] for j in train_idx]
    train_neg = [neg_labels[j] for j in train_idx]
    train_nm  = [num_mut_residues[j] for j in train_idx]
    val_pos   = [pos_labels[j] for j in val_idx]
    val_neg   = [neg_labels[j] for j in val_idx]
    val_nm    = [num_mut_residues[j] for j in val_idx]

    y_train = [1 if pos_labels[j] else 0 for j in train_idx]
    y_val   = [1 if pos_labels[j] else 0 for j in val_idx]
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

    best_state   = {k: v.clone() for k, v in model.state_dict().items()}
    best_loss    = float("inf")
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
            g_i = train_g[i].to(device)
            e_i = train_e[i].to(device)
            d_i = train_d[i].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(g_i, e_i, mut_idx, train_nm[i], d_i)
            del g_i, e_i, d_i
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

        print(f"{label_prefix}Epoch {epoch + 1}: train loss={total_loss:.4f}", flush=True)

        model.eval()
        val_loss = 0.0
        vlogits_buf: list = []
        vtargets_buf: list = []

        with torch.no_grad():
            for vi, i in enumerate(range(len(val_g))):
                mut_idx = val_pos[i][0] if val_pos[i] else val_neg[i][0]
                g_i = val_g[i].to(device)
                e_i = val_e[i].to(device)
                d_i = val_d[i].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(g_i, e_i, mut_idx, val_nm[i], d_i)
                del g_i, e_i, d_i
                vlogits_buf.append(out.squeeze())
                vtargets_buf.append(y_val_t[i])

                if (vi + 1) % batch_size == 0 or vi == len(val_g) - 1:
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        val_loss += loss_fn(
                            torch.stack(vlogits_buf), torch.stack(vtargets_buf)
                        ).item()
                    vlogits_buf, vtargets_buf = [], []

        if val_loss < best_loss:
            best_loss    = val_loss
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            print(f"{label_prefix}  new best (val loss={val_loss:.4f})", flush=True)
        else:
            patience_ctr += 1
            if patience_ctr >= es_patience:
                print(f"{label_prefix}  early stop epoch {epoch + 1}", flush=True)
                break

        scheduler.step(val_loss)

    del model, optimizer, scheduler, amp_scaler
    gc.collect()
    torch.cuda.empty_cache()

    return best_state


def train_fold(
    train_val_idx, test_idx, fold,
    X, edge_mats, pos_labels, neg_labels, clusters,
    mut_diffs_raw, seq_lengths,
    device, ablation, seed,
    prefit_scaler, precomputed_diffs,
    X_t, edge_t,
    save_path: Optional[Path] = None,
):
    """Train one fold, save best checkpoint to save_path."""
    if ablation == "scratch":
        fold_scaler = StandardScaler()
        fold_scaler.fit(np.array([mut_diffs_raw[j] for j in train_val_idx]))
        mutation_site_diffs = fold_scaler.transform(np.array(mut_diffs_raw))
    else:
        mutation_site_diffs = precomputed_diffs

    inner_kf = GroupKFold(n_splits=9, shuffle=True)
    inner_clusters = [clusters[j] for j in train_val_idx]
    train_rel, val_rel = next(inner_kf.split(range(len(train_val_idx)), groups=inner_clusters))
    train_idx = train_val_idx[train_rel]
    val_idx   = train_val_idx[val_rel]

    best_state = _train_loop(
        train_idx, val_idx,
        X, edge_mats, pos_labels, neg_labels, mutation_site_diffs, seq_lengths,
        device, ablation, seed, X_t, edge_t,
        label_prefix=f"[fold {fold}] ",
    )

    if save_path is not None:
        torch.save(best_state, save_path)
        print(f"  [saved] {save_path}", flush=True)


def train_all_data(
    all_idx,
    X, edge_mats, pos_labels, neg_labels, clusters,
    mut_diffs_raw, seq_lengths,
    device, ablation, seed,
    precomputed_diffs,
    X_t, edge_t,
    save_path: Optional[Path] = None,
):
    """Train on all data (inner val split for early stopping only)."""
    if ablation == "scratch":
        fold_scaler = StandardScaler()
        fold_scaler.fit(np.array([mut_diffs_raw[j] for j in all_idx]))
        mutation_site_diffs = fold_scaler.transform(np.array(mut_diffs_raw))
    else:
        mutation_site_diffs = precomputed_diffs

    inner_kf = GroupKFold(n_splits=9, shuffle=True)
    inner_clusters = [clusters[j] for j in all_idx]
    train_rel, val_rel = next(inner_kf.split(range(len(all_idx)), groups=inner_clusters))
    train_idx = all_idx[train_rel]
    val_idx   = all_idx[val_rel]

    best_state = _train_loop(
        train_idx, val_idx,
        X, edge_mats, pos_labels, neg_labels, mutation_site_diffs, seq_lengths,
        device, ablation, seed, X_t, edge_t,
        label_prefix="[all-data] ",
    )

    if save_path is not None:
        torch.save(best_state, save_path)
        print(f"  [saved] {save_path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}  ablation: {args.ablation}  seed: {args.seed}", flush=True)

    cfg      = DATASET_CONFIGS[args.dataset]
    save_dir = Path(args.save_models_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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
    canonical = canonical_vt_ids_path(cfg)
    if canonical.exists():
        ordered = align_to_vt_ids(data, canonical, canonical_clusters_path(cfg))
    else:
        print(f"  no canonical ordering at {canonical}; deriving one by shuffle",
              flush=True)
        ordered = shuffle_data(data)
    print(f"  {len(ordered['all_vt_ids'])} rows", flush=True)

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

    print("Precomputing graph tensors...", flush=True)
    X_t    = [torch.tensor(x, dtype=torch.float) for x in X]
    edge_t = [dense_to_sparse(torch.tensor(e))[0] for e in edge_mats]

    precomputed_diffs = None
    if prefit_scaler is not None:
        precomputed_diffs = prefit_scaler.transform(np.array(mut_diffs_raw))

    ablation_tag = f"_{args.ablation}" if args.ablation != "full" else ""
    stem = f"MutPred-PPI_{cfg.name}{ablation_tag}"

    if args.no_cv:
        all_idx = np.arange(len(X))
        save_path = save_dir / f"{stem}_all.pt"
        print(f"\nTraining on all {len(all_idx)} samples → {save_path}", flush=True)
        train_all_data(
            all_idx, X, edge_mats, pos_labels, neg_labels, clusters,
            mut_diffs_raw, seq_lengths, device,
            args.ablation, args.seed, precomputed_diffs, X_t, edge_t,
            save_path=save_path,
        )
        print(f"\nDone. Checkpoint: {save_path}", flush=True)
    else:
        gcv_seed = args.seed
        fold_splits = make_fold_splits(ordered, gcv_seed, cfg=cfg)
        print(f"\nGenerated fold splits for seed={gcv_seed}: {len(fold_splits)} folds", flush=True)

        for fold, train_val_idx, test_idx in fold_splits:
            save_path = save_dir / f"{stem}_{fold}.pt"
            fold_seed = args.seed * 10000 + gcv_seed * 100 + fold
            print(f"\nFold {fold}: {len(train_val_idx)} train+val / {len(test_idx)} test",
                  flush=True)
            train_fold(
                train_val_idx, test_idx, fold,
                X, edge_mats, pos_labels, neg_labels, clusters,
                mut_diffs_raw, seq_lengths, device,
                args.ablation, fold_seed,
                prefit_scaler, precomputed_diffs, X_t, edge_t,
                save_path=save_path,
            )
            torch.cuda.empty_cache()
            gc.collect()

        print(f"\nAll 10 folds done. Checkpoints in {save_dir}/", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train final MutPred-PPI model and save checkpoints")
    p.add_argument("--dataset", default="sahni_fragoza", choices=list(DATASET_CONFIGS))
    p.add_argument("--device", default="")
    p.add_argument("--save-models-dir", required=True,
                   help="Directory to save .pt checkpoints")
    p.add_argument("--ablation", default="megascale_all",
                   choices=[
                       "full", "full_all",
                       "megascale", "megascale_freeze_diff", "megascale_all", "megascale_head",
                       "megascale_all_no-gat", "megascale_all_no-mut", "megascale_all_wt-emb",
                       "scratch", "no-gat", "no-mut", "wt-emb",
                   ])
    p.add_argument("--seed", type=int, default=1,
                   help="GCV seed (selects fold splits file) and base random seed (default: 1)")
    p.add_argument("--no-cv", action="store_true",
                   help="Train on all data (no held-out test); saves a single _all.pt checkpoint")
    p.add_argument("--data-cache", default=None,
                   help="Path to pickle cache of loaded dataset (read if exists, write on first load)")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
