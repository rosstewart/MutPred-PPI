#!/usr/bin/env python
"""SAAMBE-3D inference for all 090826 canonical interaction-loss datasets.

Structures are resolved by chain sequence, not by filename, so the lookup is
robust to accession changes introduced by the 090826 re-mapping. The interactor
chain is identified from the structure index and passed to SAAMBE-3D; previously
it was hardcoded to "A".

SAAMBE-3D scores a PDB rather than the contact matrix, so `graphs.Structures`
resolves one straight from the canonical structure manifest, on the same pair of
sequence hashes the graph store uses. It previously took the sequence-matched
`.mat` and turned its FILENAME into a structure name, which put a name-based hop
back in front of an exact one. Coverage is partial, and rows with no structure
score NaN and are dropped by the shared per-class AUC.

Mutation positions come from the canonical tables (1-based) and are passed to
SAAMBE-3D directly, without the +1 adjustment the old labels-file code needed.

MUST be run from the SAAMBE-3D directory so that saambe-3d.py is on the path.

Usage:
    cd .../SAAMBE-3D
    conda run -n pytorch_env python saambe3d_cv.py \\
        --dataset sahni_fragoza_varchamp_all_mapped090826 \\
        --model-type regression --outdir ./results/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

from utils import mutations  # noqa: E402
from utils.gcv_common import DATASET_CONFIGS, load_data, load_splits  # noqa: E402
from utils.structures import Structures  # noqa: E402

_SAAMBE3D_PY = _HERE / "saambe-3d.py"


def _call_saambe(pdb_path: Path, chain: str, pos: str,
                 wt: str, mt: str, model_flag: str,
                 tmp_out: Path) -> tuple[float, int]:
    """Run SAAMBE-3D subprocess, return (score, binary_label)."""
    subprocess.run(
        [sys.executable, str(_SAAMBE3D_PY),
         "-i", str(pdb_path), "-c", chain,
         "-r", pos, "-w", wt, "-m", mt,
         "-d", model_flag, "-o", str(tmp_out)],
        check=True, capture_output=True,
        cwd=str(_HERE),
    )
    with open(tmp_out) as fh:
        line = fh.readline().strip()
    tmp_out.unlink()
    tokens = line.split()
    if model_flag == "1":
        score = float(tokens[0])
        binary = 1 if len(tokens) > 1 and tokens[1] == "Destabilizing" else 0
    else:
        binary = 1 if tokens[0] == "Disruptive" else 0
        score = float(binary)
    return score, binary


def run(args: argparse.Namespace) -> None:
    cfg = DATASET_CONFIGS[args.dataset]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_flag = "1" if args.model_type == "regression" else "0"
    stem = "SAAMBE-3D" if args.model_type == "regression" else "SAAMBE-3D_dn"
    out_npy    = outdir / f"{cfg.name}_{stem}_preds.npy"
    binary_npy = outdir / f"{cfg.name}_{stem}_binary_labels.npy"

    if out_npy.exists() and not args.overwrite:
        print(f"Output already exists: {out_npy}  (use --overwrite to rerun)", flush=True)
        return

    rows = load_data(cfg)
    fold_splits, _ = load_splits(cfg, seed=args.seed)
    structures = Structures(pdb_cache=outdir / "_pdb_cache")
    skipped: Counter = Counter()

    n_rows = len(rows)
    n_test_total = sum(len(test_idx) for _, _, test_idx in fold_splits)
    print(f"Dataset: {cfg.name}  model: {args.model_type}  "
          f"rows: {n_rows}  test: {n_test_total}", flush=True)

    all_preds:  list[float] = []
    all_binary: list[int]   = []
    n_ok = n_no_struct = n_error = 0

    for fold, _train_idx, test_idx in fold_splits:
        print(f"\nfold {fold}: {len(test_idx)} test rows", flush=True)
        for idx in test_idx:
            row = rows.loc[idx]
            mutation = row["mutation"]          # 1-based, e.g. "E80K"
            wt, pos_int, mt = mutations.parse(mutation)
            pos = str(pos_int)

            # (path, interactor chain id). SAAMBE-3D parses PDB only, so a
            # gzipped mmCIF from the canonical tree is converted once per pair.
            pdb_path, chain = structures.find_pdb(
                interactor=row["interactor_sequence"],
                partner=row["partner_sequence"])
            if pdb_path is None:
                skipped["no_structure"] += 1
                all_preds.append(float("nan"))
                all_binary.append(-1)
                n_no_struct += 1
                continue

            tmp_out = outdir / f"_tmp_{model_flag}_{idx}.txt"
            try:
                score, binary = _call_saambe(pdb_path, chain, pos, wt, mt,
                                             model_flag, tmp_out)
                all_preds.append(score)
                all_binary.append(binary)
                n_ok += 1
            except Exception as exc:
                print(f"  ERROR {row['interactor']} {mutation}: {exc}", flush=True)
                all_preds.append(float("nan"))
                all_binary.append(-1)
                n_error += 1
                if tmp_out.exists():
                    tmp_out.unlink()

    print(f"\n{'='*50}", flush=True)
    print(f"Done: {n_ok} ok  {n_no_struct} no structure  {n_error} errors",
          flush=True)
    for reason, count in sorted(skipped.items()):
        print(f"  skipped, {reason}: {count}", flush=True)
    np.save(out_npy,    np.array(all_preds,  dtype=np.float32))
    np.save(binary_npy, np.array(all_binary, dtype=np.int8))
    print(f"Saved: {out_npy}  shape={np.array(all_preds).shape}", flush=True)
    print(f"Saved: {binary_npy}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAAMBE-3D inference — 090826 canonical datasets"
    )
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    p.add_argument("--model-type", default="regression",
                   choices=["regression", "classification"],
                   help="regression → ddG (model -d 1); classification → Disruptive (model -d 0)")
    p.add_argument("--seed", type=int, default=0,
                   help="GCV split seed (default: 0; all rows appear in test exactly once)")
    p.add_argument("--outdir", default=".", help="Output directory")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
