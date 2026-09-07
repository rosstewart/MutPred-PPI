#!/usr/bin/env python
"""Run MutPred-PPI inference on a variant database using precomputed ProtT5 embeddings.

Prerequisites:
1. ProtT5 embeddings precomputed with precompute_prott5.py
2. AlphaFold3 contact graphs in af3_graphs/ (.mat files + all_variants.labels)
3. Trained model checkpoints in weights/ (the SFVCFP model — variant-DB inference
   is not a blind test, so the model trained on the most data is used)

Usage (nohup recommended for large datasets):
    nohup conda run -n ppi python run_variant_db_inference.py \\
        --dataset gnomad --device cuda:0 \\
        --embeddings-h5 /data/ross/ppi_lossgain/interaction_loss/gnomad/prott5_embeddings.h5 \\
        >> inference_gnomad.log 2>&1 &

    # Or with explicit paths:
    nohup conda run -n ppi python run_variant_db_inference.py \\
        --graph-dir /path/to/af3_graphs \\
        --embeddings-h5 /path/to/embeddings.h5 \\
        --out /path/to/results.tsv \\
        --device cuda:0 >> inference.log 2>&1 &

NOTE: HGMD and COSMIC datasets require licensed input data that cannot be
redistributed.  The scripts that generate their graph directories take the
licensed source files as input.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import glob
import h5py
import joblib
import numpy as np
import scipy.sparse as sp
import torch
from scipy.io import loadmat

# Resolve the models directory relative to this file
_THIS_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _THIS_DIR.parent.parent / "weights"
_SCALER_PATH = _MODELS_DIR / "mutation_diff_scaler.pkl"

# Add inference/utils to path so we can import model_loader
sys.path.insert(0, str(_THIS_DIR.parent / "inference"))
from utils.model_loader import get_models, model_predict, model_predict_subgraph  # noqa: E402

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA_ROOT  # noqa: E402


# ── dataset path registry ─────────────────────────────────────────────────────

_BASE = DATA_ROOT
DATASET_CONFIGS = {
    "clinvar": {
        "graph_dir":          _BASE / "clinvar" / "af3_graphs",
        "default_emb_h5":     _BASE / "clinvar" / "prott5_embeddings.h5",
        "default_subgraph_h5": _BASE / "clinvar" / "prott5_subgraphs.h5",
        "default_out":        _BASE / "clinvar" / "mutpred_ppi_predictions.tsv",
    },
    "gnomad": {
        "graph_dir":          _BASE / "gnomad" / "af3_graphs",
        "default_emb_h5":     _BASE / "gnomad" / "prott5_embeddings.h5",
        "default_subgraph_h5": _BASE / "gnomad" / "prott5_subgraphs.h5",
        "default_out":        _BASE / "gnomad" / "mutpred_ppi_predictions.tsv",
    },
    "hgmd": {
        "graph_dir":          _BASE / "hgmd" / "af3_graphs",
        "default_emb_h5":     _BASE / "hgmd" / "prott5_embeddings.h5",
        "default_subgraph_h5": _BASE / "hgmd" / "prott5_subgraphs.h5",
        "default_out":        _BASE / "hgmd" / "mutpred_ppi_predictions.tsv",
    },
    "cosmic": {
        "graph_dir":          _BASE / "cosmic" / "af3_graphs",
        "default_emb_h5":     _BASE / "cosmic" / "prott5_embeddings.h5",
        "default_subgraph_h5": _BASE / "cosmic" / "prott5_subgraphs.h5",
        "default_out":        _BASE / "cosmic" / "mutpred_ppi_predictions.tsv",
    },
    "autism": {
        "graph_dir":          _BASE / "autism" / "af3_graphs",
        "default_emb_h5":     _BASE / "autism" / "prott5_embeddings.h5",
        "default_subgraph_h5": _BASE / "autism" / "prott5_subgraphs.h5",
        "default_out":        _BASE / "autism" / "mutpred_ppi_predictions.tsv",
    },
    "neurodev": {
        "graph_dir":          _BASE / "neurodev" / "af3_graphs",
        "default_emb_h5":     _BASE / "neurodev" / "prott5_embeddings.h5",
        "default_subgraph_h5": _BASE / "neurodev" / "prott5_subgraphs.h5",
        "default_out":        _BASE / "neurodev" / "mutpred_ppi_predictions.tsv",
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_embeddings_h5(h5_path: str) -> dict[str, np.ndarray]:
    print(f"Loading embeddings from {h5_path} ...", flush=True)
    embs: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            embs[key] = f[key][:]
    print(f"  {len(embs)} sequences loaded", flush=True)
    return embs


def _parse_variants_labels(labels_path: str) -> dict[str, list[str]]:
    """Parse all_variants.labels → {complex_id: [variant, ...]}."""
    wt_to_vt: dict[str, list[str]] = {}
    with open(labels_path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            entry = line[1:].strip()
            # format: REFSEQ_PARTNER_interaction_loss_variant_G89R
            pdb_id = entry.split("_interaction_loss")[0]
            variant = entry.split("_")[-1]
            wt_to_vt.setdefault(pdb_id, []).append(variant)
    return wt_to_vt


def _load_done(out_path: str) -> set[str]:
    """Return set of 'complex_id\tvariant' keys already in the output TSV."""
    done: set[str] = set()
    if not Path(out_path).exists():
        return done
    with open(out_path) as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                done.add(f"{parts[0]}\t{parts[1]}")
    return done


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(
    graph_dir: str,
    emb_h5: str,
    out_path: str,
    models_dir: str,
    device_str: str,
    subgraph_h5: str | None = None,
) -> None:
    device = torch.device(device_str if device_str else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}", flush=True)

    models = get_models(models_dir, device)
    print(f"Loaded {len(models)} MutPred-PPI models from {models_dir}", flush=True)

    scaler_path = Path(models_dir) / "mutation_diff_scaler.pkl"
    scaler = joblib.load(str(scaler_path))
    print(f"Loaded scaler from {scaler_path}", flush=True)

    # Prefer subgraph H5 if available — avoids loading multi-TB full embeddings into RAM
    use_subgraph = False
    if subgraph_h5 and Path(subgraph_h5).exists():
        use_subgraph = True
        print(f"Subgraph H5 found: {subgraph_h5} — using compact 2-hop inference mode",
              flush=True)
    elif not Path(emb_h5).exists():
        print(f"[ERROR] Neither subgraph H5 nor embedding H5 found", flush=True)
        sys.exit(1)
    else:
        print(f"Using full embedding H5: {emb_h5}", flush=True)

    done = _load_done(out_path)
    print(f"  {len(done)} already scored — skipping", flush=True)

    out_file = open(out_path, "a")
    if len(done) == 0:
        out_file.write("complex_id\tvariant\tscore\n")

    mat_files = sorted(glob.glob(str(Path(graph_dir) / "*.mat")))
    print(f"  {len(mat_files)} .mat files in {graph_dir}", flush=True)

    total_scored = 0
    total_skipped = 0
    total_missing_emb = 0

    if use_subgraph:
        _run_inference_subgraph(
            mat_files, subgraph_h5, scaler, models, device,
            done, out_file, out_path,
        )
    else:
        # Legacy path: bulk-load full embeddings (only feasible for small datasets)
        embs = _load_embeddings_h5(emb_h5)
        _run_inference_full_emb(
            mat_files, embs, scaler, models, device,
            done, out_file, out_path,
        )

    out_file.close()


def _run_inference_subgraph(mat_files, subgraph_h5, scaler, models, device,
                             done, out_file, out_path):
    """Inference using pre-computed 2-hop subgraph H5 (compact, low RAM)."""
    total_scored = 0
    total_skipped = 0
    total_missing = 0

    sg_file = h5py.File(subgraph_h5, "r")

    for mat_idx, mat_path in enumerate(mat_files):
        complex_id = Path(mat_path).stem

        if complex_id not in sg_file:
            continue

        cgrp = sg_file[complex_id]

        for variant in cgrp.keys():
            key_1b = _variant_0b_to_1b(variant)
            tsv_key = f"{complex_id}\t{key_1b}"
            if tsv_key in done:
                total_skipped += 1
                continue

            vgrp = cgrp[variant]
            if "node_emb" not in vgrp or "mut_diff" not in vgrp:
                total_missing += 1
                continue

            node_emb      = vgrp["node_emb"][:]
            edge_index_np = vgrp["edge_index"][:]
            mut_diff_raw  = vgrp["mut_diff"][:].reshape(1, -1)
            mut_local_idx = int(vgrp.attrs["mut_local_idx"])

            mutation_site_diff = scaler.transform(mut_diff_raw).squeeze()

            score = model_predict_subgraph(
                node_emb, edge_index_np, models, mut_local_idx,
                mutation_site_diff, device,
            )
            if score is None:
                print(f"[WARN] model returned None for {complex_id} {variant}",
                      flush=True)
                continue

            out_file.write(f"{complex_id}\t{key_1b}\t{float(score):.6f}\n")
            out_file.flush()
            total_scored += 1

        if (mat_idx + 1) % 500 == 0:
            print(f"[{mat_idx + 1}/{len(mat_files)}] "
                  f"scored={total_scored}  skipped={total_skipped}  "
                  f"missing={total_missing}", flush=True)

    sg_file.close()
    print(f"\nDone. {total_scored} new predictions written to {out_path}", flush=True)
    print(f"  skipped (already done): {total_skipped}", flush=True)
    print(f"  missing subgraphs:      {total_missing}", flush=True)


def _variant_0b_to_1b(variant_0b: str) -> str:
    """Convert 0-based variant string to 1-based: 'A822V' → 'A823V'."""
    aa_from = variant_0b[0]
    aa_to   = variant_0b[-1]
    pos_1b  = int(variant_0b[1:-1]) + 1
    return f"{aa_from}{pos_1b}{aa_to}"


def _run_inference_full_emb(mat_files, embs, scaler, models, device,
                             done, out_file, out_path):
    """Legacy inference path using bulk-loaded full embeddings."""
    # Infer variants from emb keys (no all_variants.labels required)
    inter_to_variants: dict[str, list[str]] = {}
    for k in embs:
        if " " in k:
            inter, var = k.split(" ", 1)
            inter_to_variants.setdefault(inter, []).append(var)

    total_scored = 0
    total_skipped = 0
    total_missing_emb = 0

    for mat_idx, mat_path in enumerate(mat_files):
        complex_id = Path(mat_path).stem
        parts = complex_id.split("_")
        refseq_id  = parts[0]
        partner_id = "_".join(parts[1:])

        if refseq_id not in embs or partner_id not in embs:
            total_missing_emb += 1
            continue

        variants = inter_to_variants.get(refseq_id, [])
        if not variants:
            continue

        data = loadmat(mat_path)
        adj = sp.csr_matrix(data["G"])
        edge_mat = adj.toarray()
        n_refseq = int(data["NRR"].flat[0])

        wt_emb      = embs[refseq_id]
        partner_emb = embs[partner_id]

        for variant in variants:
            key_1b = _variant_0b_to_1b(variant)
            tsv_key = f"{complex_id}\t{key_1b}"
            if tsv_key in done:
                total_skipped += 1
                continue

            mut_idx = int(variant[1:-1])
            vt_id   = f"{refseq_id} {variant}"

            if vt_id not in embs:
                total_missing_emb += 1
                continue

            vt_emb = embs[vt_id]

            if vt_emb.shape[0] != n_refseq or partner_emb.shape[0] != (edge_mat.shape[0] - n_refseq):
                continue
            if mut_idx >= wt_emb.shape[0]:
                continue

            combined_emb = np.concatenate([vt_emb, partner_emb], axis=0)
            mut_diff_raw = (vt_emb[mut_idx] - wt_emb[mut_idx]).reshape(1, -1)
            mutation_site_diff = scaler.transform(mut_diff_raw).squeeze()

            score = model_predict(
                combined_emb, edge_mat, models, mut_idx, mutation_site_diff, device
            )
            if score is None:
                continue

            out_file.write(f"{complex_id}\t{key_1b}\t{float(score):.6f}\n")
            out_file.flush()
            total_scored += 1

        if (mat_idx + 1) % 100 == 0:
            print(f"[{mat_idx + 1}/{len(mat_files)}] "
                  f"scored={total_scored}  skipped={total_skipped}  "
                  f"missing={total_missing_emb}", flush=True)

    print(f"\nDone. {total_scored} new predictions written to {out_path}", flush=True)
    print(f"  skipped (already done): {total_skipped}", flush=True)
    print(f"  missing embeddings:     {total_missing_emb}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    if args.dataset:
        cfg = DATASET_CONFIGS[args.dataset]
        graph_dir    = str(args.graph_dir    or cfg["graph_dir"])
        emb_h5       = str(args.embeddings_h5 or cfg["default_emb_h5"])
        subgraph_h5  = str(args.subgraphs_h5 or cfg["default_subgraph_h5"])
        out_path     = str(args.out          or cfg["default_out"])
    else:
        if not args.graph_dir or not args.out:
            print("ERROR: --graph-dir and --out are required when --dataset is not specified.",
                  file=sys.stderr)
            sys.exit(1)
        graph_dir   = str(args.graph_dir)
        emb_h5      = str(args.embeddings_h5 or "")
        subgraph_h5 = str(args.subgraphs_h5  or "")
        out_path    = str(args.out)

    models_dir = str(args.models_dir or _MODELS_DIR)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    run_inference(
        graph_dir=graph_dir,
        emb_h5=emb_h5,
        out_path=out_path,
        models_dir=models_dir,
        device_str=args.device,
        subgraph_h5=subgraph_h5,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run MutPred-PPI inference on variant databases")
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS),
                   help="Named dataset (sets default paths; overridden by explicit path args)")
    p.add_argument("--graph-dir",
                   help="Directory containing .mat files and all_variants.labels")
    p.add_argument("--embeddings-h5",
                   help="Path to full per-protein ProtT5 H5 (legacy; not needed when subgraphs H5 exists)")
    p.add_argument("--subgraphs-h5",
                   help="Path to compact 2-hop subgraph H5 (preferred; auto-detected per dataset)")
    p.add_argument("--out",
                   help="Output TSV path (default: dataset-specific path under /data)")
    p.add_argument("--models-dir", default=None,
                   help=f"Directory containing .pt model files and scaler (default: {_MODELS_DIR})")
    p.add_argument("--device", default="",
                   help="PyTorch device string (e.g. 'cuda:0'). Defaults to auto-detect.")
    main(p.parse_args())
