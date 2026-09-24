"""MutPred-PPI model definitions — the single source of truth.

Previously `GAT_mut_processor` was defined verbatim in three places
(`training/train_fold.py`, `training/pretrain_stability.py`, and
`inference/pipeline/model_loader.py` under the name `MutPred_PPI`).  All three were
layer-for-layer identical, so a checkpoint written by one loaded into any other --
but editing `hidden_dim`, `num_heads`, or the `binding_predictor` head in one copy
would have silently broken `load_state_dict` for the others, or worse, loaded into
a subtly different graph.  One definition removes that failure mode.

The layer names and shapes here are exactly those of the published checkpoints in
`weights/`; do not rename attributes without re-training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GAT_mut_processor(nn.Module):
    """Two-layer GAT over the complex graph, fused with a ProtT5 mutation-diff MLP.

    forward() accepts both historical call signatures:
        (x, edge_index, mutation_idx, num_mut_res, mutation_site_diff)   # training/CV
        (x, edge_index, mutation_idx, mutation_site_diff)                # inference
    `num_mut_res` has never been read by any implementation -- it is threaded
    through by the training loops but unused -- so the 4-argument inference form
    is equivalent, not a different model.

    `mutation_idx` may be a single site (int or 0-d tensor) or a 1-d tensor of
    sites. The latter lets several samples share one pass: because this module
    pools nothing and reads only the mutated nodes, concatenating their graphs
    into one disconnected graph -- with edge indices offset -- gives each node
    exactly the neighbourhood it would have had alone. The int path below is
    byte-for-byte the released one; see `tests/test_batched_forward.py`.
    """

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

    def forward(self, x, edge_index, mutation_idx, num_mut_res, mutation_site_diff=None):
        if mutation_site_diff is None:          # 4-arg inference call
            mutation_site_diff = num_mut_res
        if mutation_site_diff.dim() == 1:
            mutation_site_diff = mutation_site_diff.unsqueeze(0)
        processed_mut_diff = self.mutation_diff_processor(mutation_site_diff)
        h = torch.relu(self.complex_gat1(x, edge_index))
        h = torch.relu(self.complex_gat2(h, edge_index))
        if torch.is_tensor(mutation_idx) and mutation_idx.dim() > 0:
            features_at_mutation = h[mutation_idx]          # batched: (B, F)
        else:
            features_at_mutation = h[mutation_idx:mutation_idx + 1]
        combined = torch.cat([features_at_mutation, processed_mut_diff], dim=-1)
        return self.binding_predictor(combined)


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

    def forward(self, x, edge_index, mutation_idx, num_mut_res, mutation_site_diff=None):
        if mutation_site_diff is None:
            mutation_site_diff = num_mut_res
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

    def forward(self, x, edge_index, mutation_idx, num_mut_res=None, mutation_site_diff=None):
        h = torch.relu(self.complex_gat1(x, edge_index))
        h = torch.relu(self.complex_gat2(h, edge_index))
        if torch.is_tensor(mutation_idx) and mutation_idx.dim() > 0:
            return self.binding_predictor(h[mutation_idx])
        return self.binding_predictor(h[mutation_idx:mutation_idx + 1])


class GAT_mut_processor_v1(nn.Module):
    """The v1.0 architecture -- RECOMB 2026 and bioRxiv v1/v2.

    Kept ONLY so the published v1.0 checkpoints remain scoreable. It is not a
    hyperparameter variant of `GAT_mut_processor` and cannot be expressed as one: the
    head has three Linear layers rather than two (hence the extra `binding_predictor.6`
    parameters, 18 tensors against 16), the GAT is four times wider (`hidden_dim=256`),
    and the mutation-diff MLP ends at 128 rather than 32.

    Everything AROUND the model -- contact graphs, ProtT5 embeddings, the mutation-diff
    scaling -- is shared with the current generation, and verified to give bit-identical
    scores to the retired v1.0 code tree on the published example. So only the weights,
    their scaler, and this class are actually legacy; the data pipeline is not.

    Scores from this generation are not comparable with the current model.
    """

    def __init__(self, input_dim, hidden_dim=256, output_dim=1,
                 num_heads=4, mutation_diff_dim=1024):
        super().__init__()

        self.mutation_diff_processor = nn.Sequential(
            nn.Linear(mutation_diff_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
        )

        self.complex_gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.complex_gat2 = GATConv(hidden_dim * num_heads, hidden_dim // 2, heads=1,
                                    concat=False)

        self.binding_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim),
        )

    def forward(self, x, edge_index, mutation_idx, num_mut_res=None,
                mutation_site_diff=None):
        # Same 4-arg/5-arg tolerance as the current class, so the shared inference path
        # can call either without knowing which generation it holds.
        if mutation_site_diff is None:
            mutation_site_diff = num_mut_res
        if mutation_site_diff.dim() == 1:
            mutation_site_diff = mutation_site_diff.unsqueeze(0)
        processed = self.mutation_diff_processor(mutation_site_diff)

        h = torch.relu(self.complex_gat1(x, edge_index))
        h = torch.relu(self.complex_gat2(h, edge_index))

        if torch.is_tensor(mutation_idx) and mutation_idx.dim() > 0:
            features = h[mutation_idx]
        else:
            features = h[mutation_idx:mutation_idx + 1]
        return self.binding_predictor(torch.cat([features, processed], dim=-1))


# Historical alias used by the public inference pipeline.
MutPred_PPI = GAT_mut_processor
MutPred_PPI_v1 = GAT_mut_processor_v1

__all__ = [
    "GAT_mut_processor",
    "GAT_mut_processor_no_gat",
    "GAT_mut_processor_no_mut",
    "GAT_mut_processor_v1",
    "MutPred_PPI",
    "MutPred_PPI_v1",
]


# ── fine-tuning freeze policy ─────────────────────────────────────────────────

def apply_freeze_strategy(model, ablation: str):
    """Freeze parameters according to `ablation`, in place, returning the model.

    This was duplicated verbatim in `mutpred_ppi_cv.train_fold` (the CV numbers)
    and `train_final_model._build_model` (the shipped model). Keeping one copy
    protects the invariant that the released model was trained the same way the
    reported numbers were -- this codebase has already drifted once where a
    definition was duplicated (`pretrain_stability` imported the canonical GAT
    while `run_stability_inference` kept its own).

    | ablation                  | trainable                                        |
    |---------------------------|--------------------------------------------------|
    | full, megascale,          | mutation_diff_processor[-1], head, both GATs     |
    | prior_best                | (only the first Linear of the diff processor is  |
    |                           | frozen -- the prior published model's own recipe)|
    | freeze_mut_processor      | head + both GATs (mutation representation fixed) |
    | freeze_gat                | head + mutation processor (structure fixed)      |
    | megascale_head            | head only -- 'Freeze Both' in the figures: the    |
    |                           | mutation processor AND both GATs are held fixed   |
    | anything else, incl.      | everything -- no freezing                        |
    | megascale_all (default)   |                                                  |
    """
    if ablation in ("full", "megascale", "prior_best"):
        for p in model.parameters():
            p.requires_grad = False
        for group in (model.mutation_diff_processor[-1], model.binding_predictor,
                      model.complex_gat1, model.complex_gat2):
            for p in group.parameters():
                p.requires_grad = True
    elif ablation == "freeze_mut_processor":
        # The whole mutation-diff processor stays fixed -- both Linears -- so the
        # pretrained mutation representation is used exactly as learned and only
        # the structural half plus the head adapt.
        for p in model.parameters():
            p.requires_grad = False
        for group in (model.binding_predictor, model.complex_gat1, model.complex_gat2):
            for p in group.parameters():
                p.requires_grad = True
    elif ablation == "freeze_gat":
        # The complement: both GAT layers fixed, mutation processor and head free.
        # Together with freeze_mut_processor this brackets which half of the model
        # the fine-tuning actually needs to move.
        for p in model.parameters():
            p.requires_grad = False
        for group in (model.mutation_diff_processor, model.binding_predictor):
            for p in group.parameters():
                p.requires_grad = True
    elif ablation == "megascale_head":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.binding_predictor.parameters():
            p.requires_grad = True
    # full_all / megascale_all / scratch / wt-emb: everything stays trainable
    return model
