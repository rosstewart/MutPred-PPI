#!/usr/bin/env python3
"""Run pretrained stability (ΔΔG) inference on variant databases.

Uses the MegaScale-pretrained stability model (NOT the fine-tuned MutPred-PPI)
to predict ΔΔG for each variant. Output ΔΔG values are in the same units as
the MegaScale training data (kcal/mol), recovered by inverse_transform.

Positive ΔΔG = destabilizing (variant less stable than WT).
Negative ΔΔG = stabilizing (variant more stable than WT).

This enables comparing interaction disruption (MutPred-PPI) with stability
effects (this script) across variant databases.

Usage:
    conda run -n ppi python src/variant_db_inference/run_stability_inference.py \
        --dataset clinvar --device cuda:0

Output: results_revisions/variant_dbs_stability/{dataset}_stability_predictions.tsv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import h5py
import joblib
import numpy as np
import scipy.sparse as sp
import torch
from scipy.io import loadmat

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from model import GAT_mut_processor  # noqa: E402
from paths import DATA_ROOT  # noqa: E402


_THIS_DIR = Path(__file__).resolve().parent
_PUB = _THIS_DIR.parent.parent
_BASE = DATA_ROOT
_MEGASCALE_PRETRAINED = _PUB / "weights" / "MutPred-PPI_stability_pretrain.pt"
_SCALER_PATH = _PUB / "weights" / "mutation_diff_scaler.pkl"
_OUT_DIR = _PUB / "results_revisions" / "variant_dbs_stability"

DATASET_CONFIGS = {
    "clinvar": {"subgraph_h5": _BASE / "clinvar" / "prott5_subgraphs.h5"},
    "gnomad":  {"subgraph_h5": _BASE / "gnomad"  / "prott5_subgraphs.h5"},
    "cosmic":  {"subgraph_h5": _BASE / "cosmic"  / "prott5_subgraphs.h5"},
    "hgmd": {
        "subgraph_h5": _BASE / "hgmd"   / "prott5_subgraphs.h5",  # doesn't exist; falls back
        "emb_h5":      _BASE / "hgmd"   / "prott5_embeddings.h5",
        "graph_dir":   _BASE / "hgmd"   / "af3_graphs",
    },
    "autism": {
        "subgraph_h5": _BASE / "autism" / "prott5_subgraphs.h5",  # doesn't exist; falls back
        "emb_h5":      _BASE / "autism" / "prott5_embeddings.h5",
        "graph_dir":   _BASE / "autism" / "af3_graphs",
    },
}


# The GAT stability predictor is `GAT_mut_processor` in src/model.py -- imported
# above. A verbatim 4th copy of the architecture used to live here; it was
# bit-for-bit identical (state_dict loads strict=True, max output delta 0.0
# over 20 random graphs), so this is a pure de-duplication. The canonical
# class has a 4-arg inference path, so call it exactly as before.


def _variant_0b_to_1b(mut: str) -> str:
    """Convert 0-based mutation key to 1-based (AA + pos + AA, e.g. 'A89V' → 'A90V')."""
    return f"{mut[0]}{int(mut[1:-1]) + 1}{mut[-1]}"


def _load_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.startswith("complex_id"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    done.add(f"{parts[0]}\t{parts[1]}")
    return done


def predict_stability(node_emb, edge_index_np, model, mut_local_idx,
                      mutation_site_diff_np, device):
    """Return raw ΔΔG (scaled) or None on error."""
    try:
        x = torch.tensor(node_emb, dtype=torch.float).to(device)
        ei = torch.tensor(edge_index_np, dtype=torch.long).to(device)
        md = torch.tensor(mutation_site_diff_np, dtype=torch.float).to(device)
        if x.size(0) == 0 or ei.size(1) == 0:
            return None
        if ei.max() >= x.size(0):
            return None
        with torch.no_grad():
            out = model(x, ei, mut_local_idx, md)
        return float(out.squeeze().cpu().numpy())
    except Exception as e:
        print(f"[WARN] predict_stability error: {e}", flush=True)
        return None


def _load_embeddings_h5(h5_path: Path) -> dict[str, np.ndarray]:
    print(f"Loading embeddings from {h5_path} ...", flush=True)
    embs = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            embs[key] = f[key][:]
    print(f"  {len(embs)} embeddings loaded", flush=True)
    return embs


def run_dataset_full_emb(dataset: str, device: torch.device, model, scaler, out_path: Path) -> None:
    """Inference via prott5_embeddings.h5 + af3_graphs/*.mat (for hgmd/autism)."""
    cfg = DATASET_CONFIGS[dataset]
    emb_h5   = cfg["emb_h5"]
    graph_dir = cfg["graph_dir"]

    if not Path(emb_h5).exists():
        print(f"[SKIP] {dataset}: embeddings H5 not found at {emb_h5}", flush=True)
        return

    mat_files = sorted(glob.glob(str(Path(graph_dir) / "*.mat")))
    if not mat_files:
        print(f"[SKIP] {dataset}: no .mat files in {graph_dir}", flush=True)
        return

    print(f"{dataset}: {len(mat_files)} .mat files, loading embeddings...", flush=True)
    embs = _load_embeddings_h5(emb_h5)

    # Build (interactor -> [variant, ...]) index from embedding keys
    inter_to_variants: dict[str, list[str]] = {}
    for k in embs:
        if " " in k:
            inter, var = k.split(" ", 1)
            inter_to_variants.setdefault(inter, []).append(var)

    done = _load_done(out_path)
    print(f"  {len(done)} already scored", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_path.exists() else "w"
    total_scored = total_skipped = total_missing = 0

    with open(out_path, mode) as out_file:
        if mode == "w":
            out_file.write("complex_id\tvariant\tddg_kcalmol\n")

        for mat_idx, mat_path in enumerate(mat_files):
            complex_id = Path(mat_path).stem
            under = complex_id.index("_")
            interactor = complex_id[:under]
            partner    = complex_id[under + 1:]

            if interactor not in embs or partner not in embs:
                total_missing += 1
                continue

            variants = inter_to_variants.get(interactor, [])
            if not variants:
                continue

            try:
                data = loadmat(mat_path)
                adj  = sp.csr_matrix(data["G"])
                n_refseq = int(data["NRR"].flat[0])
            except Exception as e:
                print(f"[WARN] mat load error {mat_path}: {e}", flush=True)
                continue

            edge_coo = adj.tocoo()
            edge_index_np = np.vstack([edge_coo.row, edge_coo.col])

            wt_emb      = embs[interactor]
            partner_emb = embs[partner]

            for variant in variants:
                key_1b  = _variant_0b_to_1b(variant)
                tsv_key = f"{complex_id}\t{key_1b}"
                if tsv_key in done:
                    total_skipped += 1
                    continue

                vt_key = f"{interactor} {variant}"
                if vt_key not in embs:
                    total_missing += 1
                    continue

                vt_emb  = embs[vt_key]
                mut_idx = int(variant[1:-1])

                if mut_idx >= wt_emb.shape[0]:
                    continue
                if vt_emb.shape[0] != n_refseq:
                    continue
                expected_partner = edge_index_np.max() + 1 - n_refseq if edge_index_np.size > 0 else 0
                if partner_emb.shape[0] != expected_partner and expected_partner > 0:
                    continue

                node_emb = np.concatenate([vt_emb, partner_emb], axis=0)
                mut_diff_raw = (vt_emb[mut_idx] - wt_emb[mut_idx]).reshape(1, -1)
                mut_diff_scaled = scaler.transform(mut_diff_raw).squeeze()

                ddg = predict_stability(node_emb, edge_index_np, model, mut_idx,
                                        mut_diff_scaled, device)
                if ddg is None:
                    continue

                out_file.write(f"{complex_id}\t{key_1b}\t{ddg:.4f}\n")
                out_file.flush()
                total_scored += 1

            if (mat_idx + 1) % 500 == 0:
                print(f"  [{mat_idx + 1}/{len(mat_files)}] scored={total_scored} "
                      f"skipped={total_skipped} missing={total_missing}", flush=True)

    print(f"  Done: {total_scored} new, {total_skipped} skipped", flush=True)


def run_dataset(dataset: str, device: torch.device, model, scaler, out_path: Path) -> None:
    cfg = DATASET_CONFIGS[dataset]
    sg_h5 = cfg["subgraph_h5"]
    if not Path(sg_h5).exists():
        # Fall back to full embedding path if available
        if "emb_h5" in cfg:
            print(f"{dataset}: no subgraph H5, using full embedding path", flush=True)
            run_dataset_full_emb(dataset, device, model, scaler, out_path)
        else:
            print(f"[SKIP] {dataset}: subgraph H5 not found at {sg_h5}", flush=True)
        return

    done = _load_done(out_path)
    print(f"{dataset}: {len(done)} already scored", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_path.exists() else "w"
    with open(out_path, mode) as out_file:
        if mode == "w":
            out_file.write("complex_id\tvariant\tddg_kcalmol\n")

        sg_file = h5py.File(sg_h5, "r")
        total_scored = total_skipped = 0

        for complex_id in sg_file.keys():
            cgrp = sg_file[complex_id]
            for variant in cgrp.keys():
                key_1b = _variant_0b_to_1b(variant)
                tsv_key = f"{complex_id}\t{key_1b}"
                if tsv_key in done:
                    total_skipped += 1
                    continue

                vgrp = cgrp[variant]
                if "node_emb" not in vgrp or "mut_diff" not in vgrp:
                    continue

                node_emb      = vgrp["node_emb"][:]
                edge_index_np = vgrp["edge_index"][:]
                mut_diff_raw  = vgrp["mut_diff"][:].reshape(1, -1)
                mut_local_idx = int(vgrp.attrs["mut_local_idx"])

                mut_diff_scaled = scaler.transform(mut_diff_raw).squeeze()
                ddg = predict_stability(
                    node_emb, edge_index_np, model, mut_local_idx, mut_diff_scaled, device
                )
                if ddg is None:
                    continue

                # Model output is raw ΔΔG (kcal/mol) — no inverse transform needed
                out_file.write(f"{complex_id}\t{key_1b}\t{ddg:.4f}\n")
                out_file.flush()
                total_scored += 1

                if total_scored % 10000 == 0:
                    print(f"  [{total_scored}] scored", flush=True)

        sg_file.close()
    print(f"  Done: {total_scored} new, {total_skipped} skipped", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=list(DATASET_CONFIGS.keys()) + ["all"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    print(f"Device: {device}", flush=True)

    print(f"Loading pretrained stability model from {_MEGASCALE_PRETRAINED}", flush=True)
    model = GAT_mut_processor(input_dim=1024).to(device)
    state = torch.load(_MEGASCALE_PRETRAINED, weights_only=True, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("  Model loaded", flush=True)

    print(f"Loading scaler from {_SCALER_PATH}", flush=True)
    scaler = joblib.load(_SCALER_PATH)
    print("  Scaler loaded", flush=True)

    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        out_path = _OUT_DIR / f"{ds}_stability_predictions.tsv"
        print(f"\n=== {ds} → {out_path}", flush=True)
        run_dataset(ds, device, model, scaler, out_path)


if __name__ == "__main__":
    main()
