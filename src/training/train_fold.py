#!/usr/bin/env python
"""The MutPred-PPI training loop, shared by every runner that fits the model.

`train_fold` is the only thing in here: it fits one GAT_mut_processor on a
train/test index pair and returns that fold's predictions. It is used by

    evaluation/mutpred_ppi_gcv.py     grouped cross-validation (Fig 3, S1)
    training/train_final_model.py     the shipped model

so the CV numbers and the released weights come from one implementation.

**Data loading no longer lives here.** This module used to carry a per-source
loader for each of sahni, sahni_fragoza, varchamp1p, cava, VarChAMP2026 and
VarChAMP_pooled -- 900 lines whose common shape was: glob a directory of `.mat`
files, parse the two accessions out of each FILENAME with a dataset-specific
`id_parts_fn`, read the adjacency with `loadmat`, split the concatenated chain
sequence at the bare integer `NRR`, hope chain A was the interactor, then
reconcile the resulting gene-symbol and UniProt namespaces against each other
with `_remap_vt_ids` and `_dedup_and_merge`. `_find_pair_graph` alone tried four
filename/orientation combinations.

All of it existed because identity lived in filenames. It is replaced by
`utils/mutpred_ppi_data.build_tensors`, which reads rows from the canonical
tables and graphs from `contact_graphs.ContactGraphStore` by sequence content:
one loader, no parsing, no namespaces to reconcile, and no orientation to
re-derive -- the store hands back a graph in which the interactor is always
nodes [0, len(interactor)).

The graph arrives as an `edge_index`, which is what the GAT consumes. The old
path stored a sparse matrix, densified it to feed this function, and then called
`dense_to_sparse` here to get back where it started.
"""

from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from paths import REPO_ROOT as _REPO_ROOT, WEIGHTS_DIR as _MODEL_WEIGHTS_DIR

_V1_0_SCALER_PATH          = _MODEL_WEIGHTS_DIR / "v1_0" / "mutation_diff_scaler_v1_0.pkl"
_MEGASCALE_SCALER_PATH     = _MODEL_WEIGHTS_DIR / "mutation_diff_scaler.pkl"
_V1_0_PRETRAINED_PATH      = _MODEL_WEIGHTS_DIR / "v1_0" / "MutPred-PPI_v1_0_stability_pretrain.pt"
_MEGASCALE_PRETRAINED_PATH = _MODEL_WEIGHTS_DIR / "MutPred-PPI_stability_pretrain.pt"

# The pre-090826 model this work supersedes, kept ONLY as the Fig S4 "Prior Best"
# ablation arm. It lives under archive/ rather than weights/ precisely so a
# production run cannot pick it up: nothing but `--ablation prior_best` reads it.
# Its layer names and shapes match GAT_mut_processor exactly, so all sixteen
# tensors transfer -- complex_gat1 included, unlike the monomer stability
# pretrains, whose first GAT is shape-filtered out.
_PRIOR_BEST_DIR = _REPO_ROOT / "archive" / "ablation_artifacts" / "prior_best"
_PRIOR_BEST_PRETRAINED_PATH = _PRIOR_BEST_DIR / "gnn_prott5_rasp4_scaledmutprocessor_whole_train.pt"
_PRIOR_BEST_SCALER_PATH     = _PRIOR_BEST_DIR / "mutation_diff_scaler.pkl"

# ── model ────────────────────────────────────────────────────────────────────
# Single definition lives in src/model.py; see its docstring for why. Re-exported
# here because the runners import the model and the loop from one place.
from model import (  # noqa: E402,F401
    apply_freeze_strategy,
    GAT_mut_processor,
    GAT_mut_processor_no_gat,
    GAT_mut_processor_no_mut,
)


# ── training ──────────────────────────────────────────────────────────────────

def train_fold(
    train_val_idx, test_idx, fold,
    X, edge_indices, pos_labels, neg_labels, clusters,
    mut_diffs_raw, seq_lengths,
    device: torch.device,
    ablation: str = "full",
    seed: int = 0,
    precomputed_diffs=None,   # pre-scaled diffs (non-scratch); None triggers per-fold fit
    X_t: Optional[list] = None,      # pre-built CPU float tensors for node features
    edge_t: Optional[list] = None,   # pre-built CPU COO edge_index tensors
    batch_size: int = 16,
    lr: float = 0.001,
    lr_patience: int = 3,
    es_patience: int = 5,
    n_epochs: int = 100,
    preload: bool | None = None,   # None = decide from free GPU memory
    fuse_batch: bool = True,       # one forward per optimiser step, not per sample
) -> Tuple[list, list]:
    # ── reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if ablation == "pretrain_zero_shot":
        # The MegaScale stability model applied to the PPI task with NO training:
        # zero epochs, so `best_state` stays the pretrained weights and the test
        # fold is scored by the checkpoint as shipped. It is a reference point --
        # how much of the task the stability pretraining already solves -- not a
        # fine-tuning arm, which is why it is reported unstratified (it never saw
        # a training fold, so C1/C2/C3 partitions rows by a property that means
        # nothing to it).
        n_epochs = 0

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
                "into weights/ (see Zenodo, docs/DATA_PREPARATION.md) or generate it "
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
        elif ablation in ("megascale", "freeze_mut_processor", "freeze_gat", "megascale_all",
                          "megascale_head", "megascale_all_wt-emb",
                          "pretrain_zero_shot"):
            _load_ckpt(_MEGASCALE_PRETRAINED_PATH, model)
        elif ablation == "prior_best":
            _load_ckpt(_PRIOR_BEST_PRETRAINED_PATH, model)
        # scratch / wt-emb: random init

        model = model.to(device)

        # ── freeze strategy ────────────────────────────────────────────────
        apply_freeze_strategy(model, ablation)

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

    def _fold_device_bytes(indices) -> int:
        """Bytes this fold's node features, edges and diffs would occupy on GPU."""
        total = 0
        for j in indices:
            total += X[j].size * 4                       # float32 node features
            total += edge_indices[j].size * 8            # int64 COO edges
            total += np.asarray(mutation_site_diffs[j]).size * 4
        return total

    def _should_preload(indices) -> bool:
        """Whether the whole fold fits on the device with room to train.

        Streaming per sample exists because full complexes did not fit: ~900
        nodes x 1024 dims x 4 B is ~4 MB each, and sfvca would need ~92 GB. The
        two-hop restriction cuts that ~30x, so the same fold is ~3 GB and the
        transfers can be dropped entirely rather than merely overlapped -- at
        100 epochs the streamed version moves hundreds of GB across PCIe per
        fold, which is what actually bounds the step time.

        The guard is deliberately conservative: activations, gradients, optimizer
        state and the model itself also need room, so the data is allowed at most
        a third of what is free. Anything larger keeps the pinned-CPU path, which
        stays correct at any size.
        """
        if preload is not None:
            return bool(preload)
        if device.type != "cuda":
            return False
        try:
            free, _total = torch.cuda.mem_get_info(device)
        except Exception:
            return False
        return _fold_device_bytes(indices) < free / 3

    def _graphs_edges_diffs(indices):
        # Resident on the device when the fold fits, pinned on the host otherwise.
        # Pinned memory enables async PCIe DMA via non_blocking=True, overlapping
        # transfers with compute; resident tensors skip the transfer altogether,
        # and `.to(device)` on an already-resident tensor returns it unchanged,
        # so the training loops need no special case.
        on_device = _should_preload(indices)
        if X_t is not None and not on_device:
            g = [X_t[j] for j in indices]   # already pinned from precompute
            e = [edge_t[j] for j in indices] # already pinned from precompute
        elif on_device:
            g = [torch.as_tensor(X[j], dtype=torch.float).to(device) for j in indices]
            e = [torch.as_tensor(edge_indices[j], dtype=torch.long).to(device)
                 for j in indices]
        else:
            g = [torch.tensor(X[j], dtype=torch.float).pin_memory() for j in indices]
            # The store already returns both directions plus self-loops, so this
            # is a view change, not a graph change.
            e = [torch.as_tensor(edge_indices[j], dtype=torch.long).pin_memory()
                 for j in indices]
        if on_device:
            d = [torch.as_tensor(mutation_site_diffs[j], dtype=torch.float).to(device)
                 for j in indices]
        else:
            d = [torch.tensor(mutation_site_diffs[j], dtype=torch.float).pin_memory()
                 for j in indices]
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


    # -- fused forward --------------------------------------------------------
    # The loops below already take ONE optimiser step per `batch_size` samples
    # (they buffer logits and stack them), so the only serial thing about them is
    # the forward pass -- `batch_size` kernel launches on ~30-node two-hop
    # subgraphs, where launch overhead dwarfs the arithmetic.
    #
    # Those graphs have different sizes, so they cannot be stacked. They can be
    # CONCATENATED: several graphs become one disconnected graph with the edge
    # indices offset. Message passing never crosses components and the model
    # pools nothing -- it reads only the mutated nodes -- so every node sees
    # exactly the neighbourhood it would have seen alone. Same loss, same
    # gradient, same single optimiser step; far fewer launches.
    #
    # Exactness, and the one caveat -- dropout draws one mask per batch rather
    # than one per sample, so a fused run matches a serial one in distribution
    # rather than bitwise -- are pinned in `tests/test_batched_forward.py`.
    # `fuse_batch=False` restores the per-sample path unchanged.

    def _fused(g, e, d, sites, idxs):
        """One forward over `idxs`, returning a 1-d tensor of logits."""
        if len(idxs) == 1 or not fuse_batch:
            return torch.stack(
                [model(g[i].to(device, non_blocking=True),
                       e[i].to(device, non_blocking=True),
                       sites[k], None,
                       d[i].to(device, non_blocking=True)).reshape(())
                 for k, i in enumerate(idxs)])
        xs, eis, centres, offset = [], [], [], 0
        for k, i in enumerate(idxs):
            x = g[i].to(device, non_blocking=True)
            xs.append(x)
            eis.append(e[i].to(device, non_blocking=True) + offset)
            centres.append(sites[k] + offset)
            offset += x.shape[0]
        return model(torch.cat(xs, dim=0),
                     torch.cat(eis, dim=1),
                     torch.as_tensor(centres, dtype=torch.long, device=device),
                     None,
                     torch.stack([d[i].to(device, non_blocking=True).reshape(-1)
                                  for i in idxs])).reshape(len(idxs))

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
            logits_buf.append(i)          # sample indices; the forward is deferred
            targets_buf.append(y_train_t[i])

            if (idx + 1) % batch_size == 0 or idx == len(shuffled) - 1:
                sites = [train_pos[j][0] if train_pos[j] else train_neg[j][0]
                         for j in logits_buf]
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = loss_fn(
                        _fused(train_g, train_e, train_d, sites, logits_buf),
                        torch.stack(targets_buf))
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
                vlogits_buf.append(i)
                vtargets_buf.append(y_val_t[i])

                if (vi + 1) % batch_size == 0 or vi == len(val_g) - 1:
                    sites = [val_pos[j][0] if val_pos[j] else val_neg[j][0]
                             for j in vlogits_buf]
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        val_loss += loss_fn(
                            _fused(val_g, val_e, val_d, sites, vlogits_buf),
                            torch.stack(vtargets_buf)
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
        for start in range(0, len(test_g), batch_size):
            idxs = list(range(start, min(start + batch_size, len(test_g))))
            sites = [test_pos[j][0] if test_pos[j] else test_neg[j][0] for j in idxs]
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = _fused(test_g, test_e, test_d, sites, idxs)
            fold_preds.extend(torch.sigmoid(out.float()).cpu().tolist())
            fold_labels.extend(float(y_test[j]) for j in idxs)

    del model, optimizer, scheduler, amp_scaler
    gc.collect()
    torch.cuda.empty_cache()

    return fold_preds, fold_labels
