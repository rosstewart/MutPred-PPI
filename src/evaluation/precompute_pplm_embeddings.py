#!/usr/bin/env python
"""Pre-compute PPLM per-residue embeddings for SWING dataset sequences.

Run this BEFORE pplm_gcv_iter.py to populate the cache that PPLMSeqDiff and
PPLMSiteDiff read at inference time.

Cache format (.pkl dict) — nested, one entry per key:

  "{id_a}_{id_b}"             → WT entry:
      mean:    (1280,) float32      full-pair mean (all La+Lb residues)
      embed_A: (La, 1280) float16   per-residue chain-A embeddings
      embed_B: (Lb, 1280) float16   per-residue chain-B embeddings

  "{id_a}_{id_b}_{var_zero}"  → MUT entry:
      mean:    (1280,) float32
      embed_A: (La, 1280) float16
      embed_B: (Lb, 1280) float16

Variant positions are 0-based in all keys (e.g. 'E79K' for 1-based 'E80K'),
matching pplm_mlp.py's zero_based_variant() convention.

PPLM uses gated cross-chain attention via a learned per-head scalar
(inter_attn_weight) in each TransformerLayer. The inter-chain mask separates
intra- and inter-chain attention paths, but does NOT block cross-chain attention:
  attn = attn_intra + inter_attn_weight * attn_inter
In the pretrained checkpoint, inter_attn_weight ranges from -1.7 to +2.4 (one
value per attention head), so cross-chain information genuinely flows. embed_A
and embed_B are therefore both cross-chain-context-aware.
Note: embed_B does change slightly between WT and MUT because A's changed token
affects the shared inter-chain attention path, though the effect is small.

No sequence truncation is applied: full-length sequences are processed.
Pairs that OOM are skipped with a warning; pplm_mlp.py returns prior for them.

Usage:
    conda run -n ppi python precompute_pplm_embeddings.py \\
        --dataset sahni_fragoza --device cuda:0

    # Extend an existing cache (only embeds missing keys):
    conda run -n ppi python precompute_pplm_embeddings.py \\
        --dataset sahni_fragoza_varchamp1p_cava --device cuda:0

Model weights: /data/ross/ppi_lossgain/interaction_loss/2026/PPLM/weights/pplm_t33_650M.pt
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── dataset configs (vendored in-repo) ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from esignet_gcv_iter_legacy import DATASET_CONFIGS, load_data  # noqa: E402

# ── PPLM package ─────────────────────────────────────────────────────────────

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import REVISIONS_DIR, cache_file# noqa: E402

_PPLM_DIR = REVISIONS_DIR / "PPLM"
_WEIGHTS_PATH = str(_PPLM_DIR / "weights" / "pplm_t33_650M.pt")
sys.path.insert(0, str(_PPLM_DIR))
from pplm.pplm import PPLM, Alphabet  # noqa: E402




# ── mutation helper ───────────────────────────────────────────────────────────

def _zero_based_variant(mutation: str) -> str:
    """'E80K' (1-based) → 'E79K' (0-based), matching pplm_mlp.py convention."""
    pos_1based = int("".join(filter(str.isdigit, mutation)))
    return f"{mutation[0]}{pos_1based - 1}{mutation[-1]}"


# ── PPLM model loading ────────────────────────────────────────────────────────

def load_pplm(device: torch.device):
    """Load PPLM base model and batch converter."""
    data = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=False)
    alphabet = Alphabet.from_architecture()
    batch_converter = alphabet.get_batch_converter()
    model = PPLM(
        num_layers=data["param"]["encoder_layers"],
        embed_dim=data["param"]["encoder_embed_dim"],
        attention_heads=data["param"]["encoder_attention_heads"],
        token_dropout=False,
        alphabet=alphabet,
    )
    model.load_state_dict(data["model"], strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, batch_converter


# ── PPLM forward ─────────────────────────────────────────────────────────────

@torch.no_grad()
def _pplm_forward(model, batch_converter, seq_a: str, seq_b: str,
                  device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Single PPLM forward pass for a sequence pair.

    Token layout:
      [BOS_A, a1..aLa, EOS_A, BOS_B, b1..bLb, EOS_B]
      La+Lb+4 tokens total.

    The inter-chain mask separates intra- and inter-chain attention paths.
    Cross-chain attention is gated by the learned inter_attn_weight scalar
    (not blocked), so each chain's embeddings are influenced by the other.

    Returns embed_A (La, 1280) and embed_B (Lb, 1280) as float32 arrays.
    """
    La, Lb = len(seq_a), len(seq_b)
    tokens_a = batch_converter([("seqA", seq_a)])[2]  # (1, La+2)
    tokens_b = batch_converter([("seqB", seq_b)])[2]  # (1, Lb+2)
    tokens = torch.cat([tokens_a, tokens_b], dim=-1).to(device)  # (1, La+Lb+4)

    L = La + 2 + Lb + 2
    mask = torch.ones((L, L), device=device)
    mask[: La + 2, : La + 2] = 0   # intra-A attention allowed
    mask[La + 2:,  La + 2:]  = 0   # intra-B attention allowed

    out = model(tokens, mask, repr_layers=[33],
                need_head_weights=False, return_contacts=False)
    rep = out["representations"][33][0]  # (L, 1280)

    embed_a = rep[1 : La + 1].cpu().float().numpy()              # (La, 1280)
    embed_b = rep[La + 3 : La + 3 + Lb].cpu().float().numpy()   # (Lb, 1280)
    return embed_a, embed_b


# ── cache population ──────────────────────────────────────────────────────────

def build_cache(
    df,
    model,
    batch_converter,
    device: torch.device,
    existing_cache: dict,
    log_interval: int,
) -> dict:
    """Compute PPLM embeddings for all unique WT pairs and MUT triplets in df.

    Skips keys already present in existing_cache (incremental update).
    OOM pairs are skipped with a warning; they will fall back to prior in pplm_mlp.py.

    df must have columns: refseq_id, partner, Mutation, Target_Seq, Interactor_Seq, Mutated_Seq
    """
    cache = dict(existing_cache)

    # ── unique WT pairs ───────────────────────────────────────────────────────
    wt_pairs: dict[tuple, tuple] = {}  # (id_a, id_b) → (wt_seq_a, partner_seq)
    for _, row in df.iterrows():
        key = (str(row["refseq_id"]), str(row["partner"]))
        if key not in wt_pairs:
            wt_pairs[key] = (str(row["Target_Seq"]), str(row["Interactor_Seq"]))

    wt_to_embed = [
        (id_a, id_b, seq_a, seq_b)
        for (id_a, id_b), (seq_a, seq_b) in wt_pairs.items()
        if not _entry_has_embeds(cache, f"{id_a}_{id_b}")
    ]

    print(f"\nWT pairs: {len(wt_pairs)} unique, {len(wt_to_embed)} to embed", flush=True)
    n_skipped = 0
    for i, (id_a, id_b, seq_a, seq_b) in enumerate(wt_to_embed):
        key = f"{id_a}_{id_b}"
        try:
            emb_a, emb_b = _pplm_forward(model, batch_converter, seq_a, seq_b, device)
            mean = (emb_a.sum(0) + emb_b.sum(0)) / (len(emb_a) + len(emb_b))
            entry = cache.setdefault(key, {})
            entry["mean"]    = mean.astype(np.float32)
            entry["embed_A"] = emb_a.astype(np.float16)
            entry["embed_B"] = emb_b.astype(np.float16)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                print(f"  WARNING: OOM skipped WT {key} "
                      f"(La={len(seq_a)}, Lb={len(seq_b)})", flush=True)
                n_skipped += 1
            else:
                raise
        if (i + 1) % log_interval == 0 or (i + 1) == len(wt_to_embed):
            print(f"  {i + 1}/{len(wt_to_embed)} WT pairs embedded "
                  f"({n_skipped} skipped)", flush=True)

    print(f"  WT done. Cache now has {len(cache)} entries.", flush=True)

    # ── unique MUT triplets ───────────────────────────────────────────────────
    mut_triplets: dict[tuple, tuple] = {}  # (id_a, id_b, var_zero) → (mut_seq_a, partner_seq)
    for _, row in df.iterrows():
        id_a = str(row["refseq_id"])
        id_b = str(row["partner"])
        var_zero = _zero_based_variant(str(row["Mutation"]))
        key = (id_a, id_b, var_zero)
        if key not in mut_triplets:
            mut_triplets[key] = (str(row["Mutated_Seq"]), str(row["Interactor_Seq"]))

    mut_to_embed = [
        (id_a, id_b, var_zero, seq_a_mut, seq_b)
        for (id_a, id_b, var_zero), (seq_a_mut, seq_b) in mut_triplets.items()
        if not _entry_has_embeds(cache, f"{id_a}_{id_b}_{var_zero}")
    ]

    print(f"\nMUT triplets: {len(mut_triplets)} unique, {len(mut_to_embed)} to embed",
          flush=True)
    n_skipped = 0
    for i, (id_a, id_b, var_zero, seq_a_mut, seq_b) in enumerate(mut_to_embed):
        key = f"{id_a}_{id_b}_{var_zero}"
        try:
            emb_a, emb_b = _pplm_forward(model, batch_converter, seq_a_mut, seq_b, device)
            mean = (emb_a.sum(0) + emb_b.sum(0)) / (len(emb_a) + len(emb_b))
            entry = cache.setdefault(key, {})
            entry["mean"]    = mean.astype(np.float32)
            entry["embed_A"] = emb_a.astype(np.float16)
            entry["embed_B"] = emb_b.astype(np.float16)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                print(f"  WARNING: OOM skipped MUT {key} "
                      f"(La={len(seq_a_mut)}, Lb={len(seq_b)})", flush=True)
                n_skipped += 1
            else:
                raise
        if (i + 1) % log_interval == 0 or (i + 1) == len(mut_to_embed):
            print(f"  {i + 1}/{len(mut_to_embed)} MUT triplets embedded "
                  f"({n_skipped} skipped)", flush=True)

    print(f"  MUT done. Cache now has {len(cache)} entries.", flush=True)
    return cache


def _entry_has_embeds(cache: dict, key: str) -> bool:
    """Return True if cache[key] already has embed_A and embed_B."""
    entry = cache.get(key)
    return (entry is not None
            and entry.get("embed_A") is not None
            and entry.get("embed_B") is not None)


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    cfg = DATASET_CONFIGS[args.dataset]
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)

    # ── load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset '{cfg.name}'...", flush=True)
    df = load_data(cfg, Path(args.data_root))
    print(f"  {len(df)} rows loaded", flush=True)

    required_cols = {"refseq_id", "partner", "Mutation",
                     "Target_Seq", "Interactor_Seq", "Mutated_Seq"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    # ── load or initialise existing cache ─────────────────────────────────────
    output_path = Path(args.output)
    existing_cache: dict = {}
    if output_path.exists():
        print(f"Loading existing cache from {output_path}...", flush=True)
        with open(output_path, "rb") as f:
            existing_cache = pickle.load(f)
        n_with_emb = sum(1 for v in existing_cache.values()
                         if isinstance(v, dict) and v.get("embed_A") is not None)
        print(f"  {len(existing_cache)} total entries, "
              f"{n_with_emb} already have embeddings", flush=True)

    # ── backfill mean for pre-existing entries that lack it ───────────────────
    n_backfilled = 0
    for entry in existing_cache.values():
        if (isinstance(entry, dict)
                and entry.get("embed_A") is not None
                and entry.get("embed_B") is not None
                and entry.get("mean") is None):
            ea = entry["embed_A"].astype(np.float32)
            eb = entry["embed_B"].astype(np.float32)
            entry["mean"] = ((ea.sum(0) + eb.sum(0)) / (len(ea) + len(eb))).astype(np.float32)
            n_backfilled += 1
    if n_backfilled:
        print(f"  Backfilled mean for {n_backfilled} existing entries.", flush=True)

    # ── load PPLM model ───────────────────────────────────────────────────────
    print(f"Loading PPLM model from {_WEIGHTS_PATH}", flush=True)
    model, batch_converter = load_pplm(device)

    # ── embed and cache ───────────────────────────────────────────────────────
    cache = build_cache(
        df, model, batch_converter, device,
        existing_cache, log_interval=args.log_interval,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    print(f"\nSaved PPLM cache to: {output_path}", flush=True)
    n_with_emb = sum(1 for v in cache.values()
                     if isinstance(v, dict) and v.get("embed_A") is not None)
    print(f"  Total entries: {len(cache)} ({n_with_emb} with embeddings)", flush=True)
    print(
        "\nTo use: pass this path via --pplm-cache to pplm_gcv_iter.py.",
        flush=True,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-compute PPLM per-residue embeddings for SWING datasets"
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIGS),
        help="Dataset to generate embeddings for",
    )
    p.add_argument(
        "--output",
        default=str(cache_file("pplm_cache.pkl")),
        help="Output .pkl cache file path (created or extended if it exists)",
    )
    p.add_argument(
        "--data-root",
        default="/data/ross/ppi_lossgain/interaction_loss/home/data_interaction_loss",
        help="Root directory containing pos/neg/extra files",
    )
    p.add_argument(
        "--device",
        default="",
        help="PyTorch device (e.g. 'cuda:0'). Defaults to auto-detect.",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print progress every N pairs (default: 100).",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
