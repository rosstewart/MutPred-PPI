#!/usr/bin/env python
"""Pre-compute MINT pair embeddings for SWING dataset sequences.

Run this BEFORE mint_gcv_iter.py to populate the cache that MINTSeqDiff and
MINTSiteDiff read at inference time.

Cache format (.pkl dict):
  mean_{id_a}_{id_b}                      → np.ndarray (1280,)  float32
      WT full-pair mean: mean over all La+Lb residues (sep_chains=False)
  mean_{id_a}_{id_b}_{var}           → np.ndarray (1280,)  float32
      MUT full-pair mean: mean over all La+Lb residues (sep_chains=False)
  res_wt_pair_{id_a}_{id_b}              → np.ndarray (La, 1280) float32
      Per-residue chain-A embeddings in WT context (only with --compute-residue)
  res_mut_pair_{var}_{id_a}_{id_b}  → np.ndarray (La, 1280) float32
      Per-residue chain-A embeddings in MUT context (only with --compute-residue)

Variant positions are 0-based in all cache keys (e.g. 'E79K' for 1-based 'E80K'),
Cache keys use the 1-based mutation exactly as the canonical tables store it.

Usage:
    # Seq-diff only (fast, small cache):
    conda run -n ppi python precompute_mint_embeddings.py \\
        --dataset sahni_fragoza --device cuda:0

    # Seq-diff + site-diff (larger cache, enables MINTSiteDiff):
    conda run -n ppi python precompute_mint_embeddings.py \\
        --dataset sahni_fragoza --device cuda:0 --compute-residue

    # Extend an existing cache with new keys only:
    conda run -n ppi python precompute_mint_embeddings.py \\
        --dataset sahni_fragoza_varchamp1p_cava --device cuda:0 --compute-residue

Model checkpoint: $MUTPRED_DATA_ROOT/2026/mint/mint.ckpt
Config:           $MUTPRED_DATA_ROOT/2026/mint/esm2_t33_650M_UR50D.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

# ── dataset configs (vendored in-repo) ───────────────────────────────────────
# Shared GCV data-loading layer (see src/evaluation/gcv_common.py).
from evaluation.gcv_common import DATASET_CONFIGS, add_mutated_sequence, load_data  # noqa: E402

# ── MINT package ─────────────────────────────────────────────────────────────

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import DATASETS_DIR, EXTERNAL_DIR, REVISIONS_DIR, cache_file  # noqa: E402

_MINT_DIR = REVISIONS_DIR / "mint"
sys.path.insert(0, str(_MINT_DIR))
import mint                                              # noqa: E402
from mint.model.esm2 import ESM2                        # noqa: E402



_CKPT_PATH   = str(_MINT_DIR / "mint.ckpt")
_CONFIG_PATH = str(_MINT_DIR / "esm2_t33_650M_UR50D.json")

# ── mutation helper ───────────────────────────────────────────────────────────



# ── MINT model wrapper ────────────────────────────────────────────────────────

class MINTEmbedder(nn.Module):
    """MINT model wrapper returning full-pair mean and per-residue chain-A embeddings."""

    def __init__(self, cfg: argparse.Namespace, checkpoint_path: str, device: torch.device):
        super().__init__()
        self.model = ESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            token_dropout=cfg.token_dropout,
            use_multimer=True,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        new_ckpt = OrderedDict(
            (k.replace("model.", ""), v) for k, v in checkpoint["state_dict"].items()
        )
        self.model.load_state_dict(new_ckpt)
        self._device = device

    @property
    def _cls_idx(self):
        return self.model.cls_idx

    @property
    def _eos_idx(self):
        return self.model.eos_idx

    @property
    def _pad_idx(self):
        return self.model.padding_idx


def _tokenize_pair(seq_a: str, seq_b: str, alphabet) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a single (seq_a, seq_b) pair for MINT input.

    Returns chains (1, La+Lb), chain_ids (1, La+Lb).
    Chain A occupies positions 0..La-1; chain B occupies La..La+Lb-1.
    CLS/EOS tokens are included; actual residues are between them.
    """
    def _encode(seq):
        return alphabet.encode(f"<cls>{seq.replace('J', 'L')}<eos>")

    enc_a = torch.tensor(_encode(seq_a), dtype=torch.int64)
    enc_b = torch.tensor(_encode(seq_b), dtype=torch.int64)

    chains    = torch.cat([enc_a, enc_b]).unsqueeze(0)      # (1, La+Lb)
    chain_ids = torch.zeros(1, len(enc_a) + len(enc_b), dtype=torch.int32)
    chain_ids[0, len(enc_a):] = 1                           # chain B = 1
    return chains, chain_ids


@torch.no_grad()
def embed_pairs(
    model: MINTEmbedder,
    items: list[tuple[str, str, str]],  # (label, seq_a, seq_b)
    alphabet,
    log_interval: int = 100,
    compute_residue: bool = True,
) -> dict[str, dict]:
    """Embed a list of (label, seq_a, seq_b) pairs.

    Returns dict: label → {"mean": np.ndarray(1280,), "res_a": np.ndarray(La,1280)|None}

    Full-pair mean follows embeddings_mint.py's sep_chains=False path: mean over all
    La+Lb residues (excluding CLS/EOS/PAD tokens) → 1280-dim.

    Pairs are processed one at a time to avoid padding complexity when
    extracting per-residue chain-A embeddings of variable-length sequences.
    """
    results: dict[str, dict] = {}
    model.eval()
    device = model._device

    for i, (label, seq_a, seq_b) in enumerate(items):
        chains, chain_ids = _tokenize_pair(seq_a, seq_b, alphabet)
        chains    = chains.to(device)
        chain_ids = chain_ids.to(device)

        out = model.model(chains, chain_ids, repr_layers=[33])["representations"][33]
        # out: (1, L_total, 1280)

        mask = (
            (~chains.eq(model._cls_idx))
            & (~chains.eq(model._eos_idx))
            & (~chains.eq(model._pad_idx))
        )  # (1, L_total)

        mask_a = (chain_ids.eq(0) & mask)[0]  # (L_total,) — for per-residue chain-A

        # Full-pair mean: verbatim sep_chains=False path from ESMMultimerWrapper.forward
        # (embeddings_mint.py) — mean over all La+Lb residues, excluding CLS/EOS/PAD.
        mask_expanded = mask.unsqueeze(-1).expand_as(out)
        sum_masked    = (out * mask_expanded).sum(dim=1)       # (1, 1280)
        mask_counts   = mask.sum(dim=1, keepdim=True).float()  # (1, 1)
        mean          = (sum_masked / mask_counts)[0]          # (1280,)

        # Per-residue chain-A embeddings (for site_diff): extract from same chain_out.
        res_out_a = out[0][mask_a]  # (La, 1280)

        results[label] = {
            "mean":  mean.cpu().float().numpy(),
            "res_a": res_out_a.cpu().float().numpy() if compute_residue else None,
        }

        if (i + 1) % log_interval == 0 or (i + 1) == len(items):
            print(f"  {i + 1}/{len(items)} pairs embedded", flush=True)

    return results


# ── cache population ──────────────────────────────────────────────────────────

def build_cache(
    df,
    model: MINTEmbedder,
    alphabet,
    existing_cache: dict,
    log_interval: int,
    compute_residue: bool,
) -> dict:
    """Compute MINT embeddings for all unique WT pairs and MUT triplets in df.

    Skips keys already present in existing_cache (incremental update).

    df must have columns: interactor, partner, mutation, interactor_sequence, partner_sequence, mutated_sequence
    """
    cache = dict(existing_cache)  # copy

    # ── collect unique WT pairs ───────────────────────────────────────────────
    wt_pairs: dict[tuple, tuple] = {}  # (id_a, id_b) → (wt_seq_a, partner_seq)
    for _, row in df.iterrows():
        key = (str(row["interactor"]), str(row["partner"]))
        if key not in wt_pairs:
            wt_pairs[key] = (str(row["interactor_sequence"]), str(row["partner_sequence"]))

    # Determine which WT pairs need embedding
    wt_to_embed: list[tuple[str, str, str]] = []  # (label, seq_a, seq_b)
    for (id_a, id_b), (seq_a, seq_b) in wt_pairs.items():
        mean_key = f"mean_{id_a}_{id_b}"
        res_key  = f"res_wt_pair_{id_a}_{id_b}"
        need_mean = mean_key not in cache
        need_res  = compute_residue and res_key not in cache
        if need_mean or need_res:
            wt_to_embed.append((f"{id_a}||{id_b}", seq_a, seq_b))

    print(f"\nWT pairs: {len(wt_pairs)} unique, {len(wt_to_embed)} to embed", flush=True)
    if wt_to_embed:
        wt_results = embed_pairs(model, wt_to_embed, alphabet, log_interval, compute_residue)
        for lbl, vals in wt_results.items():
            id_a, id_b = lbl.split("||", 1)
            cache[f"mean_{id_a}_{id_b}"] = vals["mean"]
            if compute_residue and vals["res_a"] is not None:
                cache[f"res_wt_pair_{id_a}_{id_b}"] = vals["res_a"]
    print(f"  WT done. Cache now has {len(cache)} entries.", flush=True)

    # ── collect unique MUT triplets ───────────────────────────────────────────
    mut_triplets: dict[tuple, tuple] = {}  # (id_a, id_b, var) → (mut_seq_a, partner_seq)
    for _, row in df.iterrows():
        id_a = str(row["interactor"])
        id_b = str(row["partner"])
        var = str(row["mutation"])      # 1-based, as stored in the tables
        key = (id_a, id_b, var)
        if key not in mut_triplets:
            mut_triplets[key] = (str(row["mutated_sequence"]), str(row["partner_sequence"]))

    mut_to_embed: list[tuple[str, str, str]] = []
    for (id_a, id_b, var), (mut_seq_a, seq_b) in mut_triplets.items():
        mean_key = f"mean_{id_a}_{id_b}_{var}"
        res_key  = f"res_mut_pair_{var}_{id_a}_{id_b}"
        need_mean = mean_key not in cache
        need_res  = compute_residue and res_key not in cache
        if need_mean or need_res:
            mut_to_embed.append((f"{id_a}||{id_b}||{var}", mut_seq_a, seq_b))

    print(f"\nMUT triplets: {len(mut_triplets)} unique, {len(mut_to_embed)} to embed", flush=True)
    if mut_to_embed:
        mut_results = embed_pairs(model, mut_to_embed, alphabet, log_interval, compute_residue)
        for lbl, vals in mut_results.items():
            id_a, id_b, var = lbl.split("||", 2)
            cache[f"mean_{id_a}_{id_b}_{var}"] = vals["mean"]
            if compute_residue and vals["res_a"] is not None:
                cache[f"res_mut_pair_{var}_{id_a}_{id_b}"] = vals["res_a"]
    print(f"  MUT done. Cache now has {len(cache)} entries.", flush=True)

    return cache


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    cfg = DATASET_CONFIGS[args.dataset]
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)

    # ── load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset '{cfg.name}'...", flush=True)
    df = add_mutated_sequence(load_data(cfg))
    print(f"  {len(df)} rows loaded", flush=True)

    # load_data drops rows with missing/synonymous mutations;
    # it also populates interactor_sequence, partner_sequence, mutated_sequence
    required_cols = {"interactor", "partner", "mutation",
                     "interactor_sequence", "partner_sequence", "mutated_sequence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    # ── load or initialise existing cache ─────────────────────────────────────
    # Default: one cache per dataset, beside the canonical tables.
    output_path = Path(args.output or
                       DATASETS_DIR / "mapped090826" / f"{args.dataset}_mint.pkl")
    existing_cache: dict = {}
    if output_path.exists():
        print(f"Loading existing cache from {output_path}...", flush=True)
        with open(output_path, "rb") as f:
            existing_cache = pickle.load(f)
        print(f"  {len(existing_cache)} entries already cached", flush=True)

    # ── load MINT model ───────────────────────────────────────────────────────
    print(f"Loading MINT config from {_CONFIG_PATH}", flush=True)
    with open(_CONFIG_PATH) as f:
        cfg_dict = json.load(f)
    model_cfg = argparse.Namespace(**cfg_dict)

    print(f"Loading MINT checkpoint from {_CKPT_PATH}", flush=True)
    embedder = MINTEmbedder(model_cfg, _CKPT_PATH, device).to(device).eval()

    alphabet = mint.data.Alphabet.from_architecture("ESM-1b")

    # ── embed and cache ───────────────────────────────────────────────────────
    cache = build_cache(
        df,
        embedder,
        alphabet,
        existing_cache,
        log_interval=args.log_interval,
        compute_residue=args.compute_residue,
    )

    # ── save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    n_mean = sum(1 for k in cache if k.startswith("mean_"))
    print(f"\nSaved MINT cache to: {output_path}", flush=True)
    print(f"  Total entries: {len(cache)} ({n_mean} full-pair mean_* entries)", flush=True)
    if args.compute_residue:
        n_res = sum(1 for k in cache if k.startswith("res_"))
        print(f"  Of which per-residue (res_*) entries: {n_res}", flush=True)
    print(
        "\nTo use: pass this path via --mint-cache to mint_gcv_iter.py.",
        flush=True,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-compute MINT pair embeddings for SWING datasets"
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIGS),
        help="Dataset to generate embeddings for",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output .pkl cache file path (created or extended if it exists)",
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
        help=(
            "Print progress every N pairs (default: 100). "
            "Pairs are processed individually (no padding) to simplify "
            "per-residue extraction."
        ),
    )
    p.add_argument(
        "--compute-residue",
        action="store_true",
        default=False,
        help=(
            "Also compute per-residue chain-A embeddings (res_wt_pair_* and "
            "res_mut_pair_* cache keys). Required for MINTSiteDiff / site_diff GCV. "
            "Significantly increases cache size and runtime (default: off)."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
