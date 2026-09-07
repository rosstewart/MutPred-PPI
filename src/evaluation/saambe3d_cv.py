#!/usr/bin/env python
"""Unified SAAMBE-3D inference script for all four interaction-loss datasets.

Replaces run_sahni_preds.py, run_sahni_preds_three_datasets.py, and
run_sahni_dn_preds_three_datasets.py with a single --dataset-driven script.

Iterates through the canonical base fold_splits.pkl for the chosen dataset,
runs SAAMBE-3D on each test variant in fold order, and saves a flat numpy
prediction array aligned to that iteration order (matching the format expected
by the eval notebook).

MUST be run from the SAAMBE-3D directory so that saambe-3d.py is on the path.

Usage:
    cd .../SAAMBE-3D
    conda run -n pytorch_env python saambe3d_preds.py \\
        --dataset sahni --model-type regression --outdir ./results/
    conda run -n pytorch_env python saambe3d_preds.py \\
        --dataset sahni_fragoza_varchamp1p_cava \\
        --model-type classification --outdir ./results/
"""

from __future__ import annotations

import argparse
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import sys

import numpy as np

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CV_DIR, DATA_ROOT  # noqa: E402


# ── fixed paths ───────────────────────────────────────────────────────────────
_CV_DIR = CV_DIR
_AF3_PDBS = DATA_ROOT / "three_datasets_af3_models" / "pdbs"
_SAHNI_PDBS = DATA_ROOT / "sahni_pdbs"
_ALL_TO_UNIPROT = DATA_ROOT / "all_to_uniprot.pkl"
# ── dataset configuration ─────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    name: str
    fold_splits_file: str    # in _CV_DIR, base split (no seed suffix)
    labels_file: str         # all_vt_ids_and_labels.txt variant, in _CV_DIR
    use_sahni_pdbs: bool     # True → sahni_pdbs/; False → three_datasets AF3


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "sahni": DatasetConfig(
        name="sahni",
        fold_splits_file="fold_splits.pkl",
        labels_file="all_vt_ids_and_labels.txt",
        use_sahni_pdbs=True,
    ),
    "sahni_fragoza": DatasetConfig(
        name="sahni_fragoza",
        fold_splits_file="sahni_fragoza_train_fold_splits.pkl",
        labels_file="sahni_fragoza_all_vt_ids_and_labels.txt",
        use_sahni_pdbs=False,
    ),
    "sahni_varchamp1p_cava": DatasetConfig(
        name="sahni_varchamp1p_cava",
        fold_splits_file="sahni_varchamp1p_cava_train_fold_splits.pkl",
        labels_file="combined_sahni_varchamp1p_cava_seq_confirmed_all_vt_ids_and_labels.txt",
        use_sahni_pdbs=False,
    ),
    "sahni_fragoza_varchamp1p_cava": DatasetConfig(
        name="sahni_fragoza_varchamp1p_cava",
        fold_splits_file="sahni_fragoza_varchamp1p_cava_train_fold_splits.pkl",
        labels_file="combined_sahni_fragoza_varchamp1p_cava_seq_confirmed_all_vt_ids_and_labels.txt",
        use_sahni_pdbs=False,
    ),
}


# ── ID helpers (matching eval notebook) ──────────────────────────────────────

def get_gene_name(gene_name_and_orf_id: str) -> str:
    """Strip numeric ORF suffix; pass RefSeq IDs through unchanged."""
    if gene_name_and_orf_id.startswith(("NP_", "np_")):
        return gene_name_and_orf_id
    if "_" not in gene_name_and_orf_id:
        return gene_name_and_orf_id
    return "_".join(gene_name_and_orf_id.split("_")[:-1])


def split_wt_id(wt_id: str) -> tuple[str, str]:
    """Split a complex_id into (protein_1, protein_2)."""
    if wt_id.startswith(("NP_", "np_")):
        return "_".join(wt_id.split("_")[:2]), "_".join(wt_id.split("_")[2:])
    delim = "_" if "_" in wt_id else "-"
    parts = wt_id.split(delim)
    if len(parts) == 2:
        return (parts[0], parts[1])
    part_split_idx = -1
    for idx, part in enumerate(parts):
        try:
            int(part)
            part_split_idx = idx + 1
            break
        except ValueError:
            continue
    assert part_split_idx != -1, f"Could not split wt_id: {wt_id}"
    if part_split_idx == len(parts):
        part_split_idx = 1
    return (delim.join(parts[:part_split_idx]), delim.join(parts[part_split_idx:]))


def is_uniprot_accession(id_str: str) -> bool:
    p1 = r"^[OPQ][0-9][A-Z0-9]{3}[0-9](?:-[0-9]+)?$"
    p2 = r"^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(?:-[0-9]+)?$"
    return bool(re.match(p1, id_str) or re.match(p2, id_str))


def _norm(s: str) -> str:
    """Lowercase and replace hyphens with underscores for PDB filenames."""
    return s.lower().replace("-", "_")


# ── PDB path resolution ───────────────────────────────────────────────────────

def _build_uniprot_to_gene(all_to_uniprot: dict) -> dict[str, list[str]]:
    """Invert all_to_uniprot (gene_symbol → UniProt) to UniProt → [gene_symbols]."""
    inv: dict[str, list[str]] = {}
    for gene, uniprot in all_to_uniprot.items():
        inv.setdefault(uniprot, []).append(gene)
    return inv


def _try_af3_pdb(g1: str, g2: str) -> Path:
    return _AF3_PDBS / f"fold_{_norm(g1)}_{_norm(g2)}_model_0.pdb"


def resolve_pdb_three_datasets(part_1: str, part_2: str,
                                u2g: dict[str, list[str]]) -> Path:
    """Resolve AF3 PDB path for non-sahni datasets."""
    # Primary: use IDs as-is (sahni_fragoza UniProt IDs, gene-symbol IDs)
    g1 = get_gene_name(part_1)
    g2 = get_gene_name(part_2)
    cand = _try_af3_pdb(g1, g2)
    if cand.exists():
        return cand

    # Fallback: strip isoform suffix, then look up UniProt → gene_symbol
    base1 = part_1.split("-")[0]
    base2 = part_2.split("-")[0]
    for g1_alt in u2g.get(base1, [g1]):
        for g2_alt in u2g.get(base2, [g2]):
            cand2 = _try_af3_pdb(get_gene_name(g1_alt), get_gene_name(g2_alt))
            if cand2.exists():
                return cand2

    return cand  # best guess; caller checks existence


def resolve_pdb(complex_id: str, cfg: DatasetConfig,
                u2g: dict[str, list[str]]) -> Path:
    part_1, part_2 = split_wt_id(complex_id)
    if cfg.use_sahni_pdbs:
        return _SAHNI_PDBS / f"fold_{_norm(part_1)}_{_norm(part_2)}_model_0.pdb"
    return resolve_pdb_three_datasets(part_1, part_2, u2g)


# ── SAAMBE-3D subprocess call ─────────────────────────────────────────────────

def _call_saambe(pdb_path: Path, chain: str, pos_str: str,
                 wt_res: str, mt_res: str, model_flag: str,
                 tmp_out: Path) -> tuple[float, int]:
    """Run SAAMBE-3D and return (score, binary_label).

    Regression (-d 1): output is "{ddg} Destabilizing|Stabilizing"
        score = ddG float; binary_label = 1 if Destabilizing else 0
    Classification (-d 0): output is "Disruptive|Nondisruptive" (no numeric)
        score = float(binary_label); binary_label = 1 if Disruptive else 0
    """
    _SAAMBE3D_PY = Path(__file__).parent / "saambe-3d.py"
    subprocess.run(
        [sys.executable, str(_SAAMBE3D_PY),
         "-i", str(pdb_path), "-c", chain,
         "-r", pos_str, "-w", wt_res, "-m", mt_res,
         "-d", model_flag, "-o", str(tmp_out)],
        check=True, capture_output=True,
        cwd=str(Path(__file__).parent),
    )
    with open(tmp_out) as fh:
        line = fh.readline().strip()
    tmp_out.unlink()

    tokens = line.split()
    if model_flag == "1":
        # regression: "{ddg} Destabilizing|Stabilizing"
        score = float(tokens[0])
        binary = 1 if len(tokens) > 1 and tokens[1] == "Destabilizing" else 0
    else:
        # classification: "Disruptive|Nondisruptive"
        binary = 1 if tokens[0] == "Disruptive" else 0
        score = float(binary)

    return score, binary


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    cfg = DATASET_CONFIGS[args.dataset]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_flag = "1" if args.model_type == "regression" else "0"
    stem = "SAAMBE-3D" if args.model_type == "regression" else "SAAMBE-3D_dn"
    out_npy = outdir / f"{cfg.name}_{stem}_preds.npy"

    if out_npy.exists() and not args.overwrite:
        print(f"Output already exists: {out_npy}  (use --overwrite to rerun)", flush=True)
        return

    with open(_ALL_TO_UNIPROT, "rb") as f:
        all_to_uniprot = pickle.load(f)
    u2g = _build_uniprot_to_gene(all_to_uniprot)

    with open(_CV_DIR / cfg.fold_splits_file, "rb") as f:
        fold_splits = pickle.load(f)

    # canonical vt_id ordering from labels file
    vt_ids: list[tuple[str, str]] = []
    with open(_CV_DIR / cfg.labels_file) as f:
        for line in f:
            parts = line.strip().split()
            vt_ids.append((parts[0], parts[1]))  # (complex_id, variant)

    total_test = sum(len(test_idx) for _, _, test_idx in fold_splits)
    print(f"Dataset: {cfg.name}  model_type: {args.model_type}", flush=True)
    print(f"Total test variants: {total_test}", flush=True)

    all_preds: list[float] = []
    all_binary: list[int]  = []
    n_ok = n_missing = n_error = 0

    for fold, train_idx, test_idx in fold_splits:
        print(f"\nfold {fold}: {len(test_idx)} test variants", flush=True)
        for idx in test_idx:
            complex_id, variant = vt_ids[idx]
            wt_res  = variant[0]
            pos_str = str(int(variant[1:-1]) + 1)  # labels file is 0-indexed; PDB residues are 1-based
            mt_res  = variant[-1]

            pdb_path = resolve_pdb(complex_id, cfg, u2g)
            if not pdb_path.exists():
                print(f"  MISSING PDB: {complex_id} → {pdb_path.name}", flush=True)
                all_preds.append(float("nan"))
                all_binary.append(-1)
                n_missing += 1
                continue

            tmp_out = outdir / f"_tmp_{model_flag}_{idx}.txt"
            try:
                score, binary = _call_saambe(pdb_path, "A", pos_str, wt_res, mt_res,
                                             model_flag, tmp_out)
                all_preds.append(score)
                all_binary.append(binary)
                n_ok += 1
            except Exception as exc:
                print(f"  ERROR {complex_id} {variant}: {exc}", flush=True)
                all_preds.append(float("nan"))
                all_binary.append(-1)
                n_error += 1
                if tmp_out.exists():
                    tmp_out.unlink()

    binary_npy = outdir / f"{cfg.name}_{stem}_binary_labels.npy"
    print(f"\n{'='*50}", flush=True)
    print(f"Done: {n_ok} ok  {n_missing} missing PDB  {n_error} errors", flush=True)
    np.save(out_npy,    np.array(all_preds,  dtype=np.float32))
    np.save(binary_npy, np.array(all_binary, dtype=np.int8))
    print(f"Saved: {out_npy}  shape={np.array(all_preds).shape}", flush=True)
    print(f"Saved: {binary_npy}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAAMBE-3D inference — all four interaction-loss datasets"
    )
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    p.add_argument(
        "--model-type", default="regression",
        choices=["regression", "classification"],
        help="regression → ddG (model -d 1); classification → disruptive/non-disruptive (model -d 0)",
    )
    p.add_argument("--outdir", default=".",
                   help="Output directory for pred arrays (default: current dir)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output file")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
