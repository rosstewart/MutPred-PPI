#!/usr/bin/env python3
"""Pretrain GAT_mut_processor for ddG stability regression.

Uses data from preprocess_stability_data.py.
Outputs a .pt state-dict compatible with _MEGASCALE_PRETRAINED_PATH in mutpred_ppi_cv.py.

Usage:
    python pretrain_stability.py \\
        --data     megascale_preprocessed/preprocessed.pkl \\
        --scaler   weights/mutation_diff_scaler.pkl \\
        --outmodel weights/MutPred-PPI_stability_pretrain.pt \\
        --device   cuda:0
"""

from __future__ import annotations

import argparse
import gc
import pickle
import random
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import torch
from preprocess_stability_data import expand_emb
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import dense_to_sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── model ────────────────────────────────────────────────────────────────────
# Single definition lives in src/model.py; see its docstring for why.

import sys as _sys
from pathlib import Path as _Path
from model import GAT_mut_processor  # noqa: E402


# ── training ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.eval_only and not args.checkpoint:
        raise ValueError("--checkpoint is required when using --eval-only")

    device  = torch.device(args.device if args.device else
                           ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda"
    print(f"device: {device}  amp: {use_amp}", flush=True)

    # ── load data ─────────────────────────────────────────────────────────
    print("Loading preprocessed data...", flush=True)
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    scaler = joblib.load(args.scaler)

    # expand_emb handles both plain ndarray (full) and sparse dict (subgraph_hops>0)
    X               = [expand_emb(x) for x in data["prott5_embeddings"]]
    edge_mats       = data["edge_mats"]
    mut_diffs_raw   = data["mutation_site_diffs"]
    seq_lengths     = data["seq_lengths"]
    mutation_indices = data["mutation_indices"]   # np.int64 array
    ddg_labels      = data["ddg_labels"]          # np.float32 array
    splits          = data["splits"]

    train_idx = splits["train"]
    val_idx   = splits["val"]
    print(f"  train: {len(train_idx)}  val: {len(val_idx)}  test: {len(splits['test'])}",
          flush=True)

    # scale mutation diffs once
    mutation_site_diffs = scaler.transform(np.array(mut_diffs_raw))

    # ── precompute CPU tensors (avoids repeated conversion per epoch) ─────
    print("Precomputing CPU tensors...", flush=True)
    X_t    = [torch.tensor(x, dtype=torch.float) for x in X]
    edge_t = [dense_to_sparse(torch.tensor(e))[0] for e in edge_mats]
    print(f"  done ({len(X_t)} graphs)", flush=True)

    num_mut_residues = [lengths[0] for lengths in seq_lengths]
    ddg_t = torch.tensor(ddg_labels, dtype=torch.float)

    # ── model ─────────────────────────────────────────────────────────────
    model     = GAT_mut_processor(input_dim=X[0].shape[1]).to(device)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loss_fn   = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=3, min_lr=1e-7)

    if args.eval_only:
        print(f"Loading checkpoint from {args.checkpoint}...", flush=True)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        best_val_loss = 0.0
    else:
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        best_val_loss = float("inf")
        patience_ctr  = 0

    def _graphs_edges_diffs(indices) -> Tuple[list, list, list]:
        g = [X_t[j].to(device)    for j in indices]
        e = [edge_t[j].to(device) for j in indices]
        d = [torch.tensor(mutation_site_diffs[j], dtype=torch.float).to(device)
             for j in indices]
        return g, e, d

    if not args.eval_only:
        # pre-build val tensors (fixed across epochs)
        val_g, val_e, val_d = _graphs_edges_diffs(val_idx)
        val_labels_t = ddg_t[val_idx].to(device)
        val_nm       = [num_mut_residues[j] for j in val_idx]
        val_midx     = [int(mutation_indices[j]) for j in val_idx]

        # ── training loop ─────────────────────────────────────────────────────
        for epoch in range(args.epochs):
            model.train()
            shuffled = list(train_idx)
            random.shuffle(shuffled)
            total_loss  = 0.0
            logits_buf: list = []
            targets_buf: list = []

            for step_i, j in enumerate(shuffled):
                g  = X_t[j].to(device)
                e  = edge_t[j].to(device)
                d  = torch.tensor(mutation_site_diffs[j], dtype=torch.float).to(device)
                nm = num_mut_residues[j]
                mi = int(mutation_indices[j])

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(g, e, mi, nm, d)
                logits_buf.append(out.squeeze())
                targets_buf.append(ddg_t[j].to(device))

                if (step_i + 1) % args.batch_size == 0 or step_i == len(shuffled) - 1:
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

            # ── validation ────────────────────────────────────────────────────
            model.eval()
            val_loss   = 0.0
            vlogits_buf: list = []
            vtargets_buf: list = []

            with torch.no_grad():
                for vi in range(len(val_idx)):
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        out = model(val_g[vi], val_e[vi], val_midx[vi], val_nm[vi], val_d[vi])
                    vlogits_buf.append(out.squeeze())
                    vtargets_buf.append(val_labels_t[vi])

                    if (vi + 1) % args.batch_size == 0 or vi == len(val_idx) - 1:
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            val_loss += loss_fn(
                                torch.stack(vlogits_buf), torch.stack(vtargets_buf)
                            ).item()
                        vlogits_buf, vtargets_buf = [], []

            print(f"Epoch {epoch + 1}/{args.epochs}  "
                  f"train_loss={total_loss:.4f}  val_loss={val_loss:.4f}", flush=True)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr  = 0
            else:
                patience_ctr += 1
                if patience_ctr >= args.es_patience:
                    print(f"Early stopping at epoch {epoch + 1}", flush=True)
                    break

        model.load_state_dict(best_state)

    # ── test evaluation ──────────────────────────────────────────────────
    test_idx = splits["test"]
    te_g, te_e, te_d = _graphs_edges_diffs(test_idx)
    te_labels_t = ddg_t[test_idx].to(device)
    te_nm       = [num_mut_residues[j] for j in test_idx]
    te_midx     = [int(mutation_indices[j]) for j in test_idx]

    model.eval()
    te_preds: list = []
    with torch.no_grad():
        for ti in range(len(test_idx)):
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(te_g[ti], te_e[ti], te_midx[ti], te_nm[ti], te_d[ti])
            te_preds.append(out.squeeze().cpu().item())

    te_preds_arr = np.array(te_preds, dtype=np.float32)
    te_labels_arr = te_labels_t.cpu().numpy()

    mae = mean_absolute_error(te_labels_arr, te_preds_arr)
    rmse = np.sqrt(mean_squared_error(te_labels_arr, te_preds_arr))
    r_pearson, p_pearson = pearsonr(te_labels_arr, te_preds_arr)
    r_spearman, p_spearman = spearmanr(te_labels_arr, te_preds_arr)

    print(f"\nTest Set Metrics ({len(test_idx)} samples):", flush=True)
    print(f"  MAE:  {mae:.4f}", flush=True)
    print(f"  RMSE: {rmse:.4f}", flush=True)
    print(f"  Pearson r:  {r_pearson:.4f}  (p={p_pearson:.2e})", flush=True)
    print(f"  Spearman ρ: {r_spearman:.4f}  (p={p_spearman:.2e})", flush=True)

    # ── save checkpoint + plot ───────────────────────────────────────────
    if not args.eval_only:
        out_path = Path(args.outmodel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        print(f"Model saved to {out_path}  (best val_loss={best_val_loss:.4f})", flush=True)
    else:
        out_path = Path(args.outmodel) if args.outmodel else Path(args.checkpoint).parent / "eval"
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(te_labels_arr, te_preds_arr, alpha=0.3, s=5)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'k--', lw=1, label='perfect')
            ax.set_xlabel('Ground-truth ΔΔG', fontsize=12)
            ax.set_ylabel('Predicted ΔΔG', fontsize=12)
            ax.set_title(f'Pretrain Stability: Test Set (r={r_pearson:.3f}, RMSE={rmse:.3f})',
                        fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_path = out_path.parent / f"{out_path.stem}_test_predictions.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {plot_path}", flush=True)
            plt.close()
        except ImportError:
            print("(matplotlib not available; skipping plot)", flush=True)

    del model, optimizer, scheduler, amp_scaler
    gc.collect()
    torch.cuda.empty_cache()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pretrain GAT_mut_processor for ddG stability regression"
    )
    p.add_argument("--data",        required=True,
                   help="preprocessed.pkl from preprocess_stability_data.py")
    p.add_argument("--scaler",      required=True,
                   help="mutation_diff_scaler.pkl from preprocess_stability_data.py")
    p.add_argument("--outmodel",    required=True,
                   help="Output .pt path (plain state dict)")
    p.add_argument("--device",      default="",
                   help="PyTorch device (default: auto-detect)")
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--lr",          type=float, default=0.001)
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--es-patience", type=int, default=10,
                   help="Early stopping patience (default: 10)")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--plot", action="store_true",
                   help="Save a scatter plot of test predictions vs ground truth (requires matplotlib)")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; load checkpoint and evaluate on test set only")
    p.add_argument("--checkpoint", default="",
                   help="Path to checkpoint to load (required if --eval-only is set)")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
