#!/usr/bin/env python3
"""Import MutPred2 blind-test scores for the new VCFP entries into the main arrays.

MutPred2 has no trainable model to run in-repo — it's an external tool whose
output is a CSV of (protein, mutation, score) scored offline. This script
parses that CSV, covers the ~2,936 new VCFP entries (VC1p+CAVA, UniProt-remapped)
that were absent from the original MutPred2 blind test, and then automatically
merges the result into the main MutPred2 VCFP arrays and restratifies C1/C2/C3
classification — the same one-command experience as
src/evaluation/run_vcfp_blind_test.py provides for the other methods.

The input FASTA (data/mutpred2_vc1pcava_supplement_input.fasta, used to
generate the externally-run CSV) uses 1-based positions. The blind test
vt_ids use 0-based positions. This script converts back to match vt_ids when
looking up scores.

Usage:
    conda run -n ppi python src/analysis/import_mutpred2_vcfp_scores.py \\
        --csv mutpred2_vc1pcava_output.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ANALYSIS_DIR = Path(__file__).resolve().parent  # src/analysis


# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import VCFP_RESULTS_DIR  # noqa: E402


_EVAL = VCFP_RESULTS_DIR
_FULL_METHOD = "MutPred-PPI (megascale_all, all-data) (varchamp_full_pooled)"
_MP2_METHOD  = "MutPred2 (varchamp_full_pooled)"
_SUPP_TAG    = "vc1pcava"


def load_mutpred2_csv(csv_path: Path) -> dict[tuple[str, str], float]:
    """Return {(uniprot_id, mutation_1based): score} from MutPred2 output CSV."""
    scores: dict[tuple[str, str], float] = {}
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                prot_id = parts[0].strip()
                mutation = parts[1].strip()
                score = float(parts[2].strip())
                scores[(prot_id, mutation)] = score
            except ValueError:
                continue
    return scores


def mut_0based_to_1based(mut0: str) -> str:
    """Convert 0-based mutation string to 1-based (as used by MutPred2)."""
    wt_aa = mut0[0]
    pos0 = int(mut0[1:-1])
    mut_aa = mut0[-1]
    return f"{wt_aa}{pos0 + 1}{mut_aa}"


def run(csv_path: Path, dry_run: bool = False) -> str | None:
    """Parse the MutPred2 output CSV, save the vc1pcava supplement npy arrays,
    then merge + restratify into the main MutPred2 VCFP arrays.

    Returns _MP2_METHOD (the method description / key), or None if no
    supplement entries were written (nothing to merge/restratify).
    """
    print(f"Loading MutPred2 output: {csv_path}")
    mp2_scores = load_mutpred2_csv(csv_path)
    print(f"  {len(mp2_scores)} (protein, mutation) scores loaded")

    # Load full-coverage MutPred-PPI arrays to get the canonical set of new vt_ids
    full_vt_ids: set[str] = set()
    for c in [1, 2, 3]:
        vf = _EVAL / f"{_FULL_METHOD}_c{c}_vt_ids.npy"
        full_vt_ids.update(np.load(vf, allow_pickle=True).tolist())

    # Load existing MutPred2 vt_ids (14,116 entries)
    mp2_vt_ids: set[str] = set()
    for c in [1, 2, 3]:
        vf = _EVAL / f"{_MP2_METHOD}_c{c}_vt_ids.npy"
        mp2_vt_ids.update(np.load(vf, allow_pickle=True).tolist())

    new_vt_ids = sorted(full_vt_ids - mp2_vt_ids)
    print(f"\nNew vc1pcava entries: {len(new_vt_ids)}")

    # Get canonical class for new entries from restratified MutPred-PPI arrays
    mp2_full_class: dict[str, int] = {}
    for c in [1, 2, 3]:
        vf = _EVAL / f"{_FULL_METHOD}_c{c}_vt_ids.npy"
        for vid in np.load(vf, allow_pickle=True):
            if vid not in mp2_vt_ids:
                mp2_full_class[vid] = c

    # Build per-class predictions
    per_class: dict[int, list] = {c: [] for c in [1, 2, 3]}

    # Load labels from MutPred-PPI arrays (they share the same ground truth)
    mp2_full_labels: dict[str, int] = {}
    for c in [1, 2, 3]:
        vf = _EVAL / f"{_FULL_METHOD}_c{c}_vt_ids.npy"
        lf = _EVAL / f"{_FULL_METHOD}_c{c}_labels.npy"
        vids = np.load(vf, allow_pickle=True)
        labs = np.load(lf)
        for vid, lab in zip(vids, labs):
            mp2_full_labels[vid] = int(lab)

    n_found = n_missing = 0
    for vt in new_vt_ids:
        parts = vt.split()
        uid, mut0 = parts[0], parts[2]
        mut1 = mut_0based_to_1based(mut0)
        score = mp2_scores.get((uid, mut1))
        c = mp2_full_class.get(vt, 3)  # default C3 if not classified
        label = mp2_full_labels.get(vt, -1)

        if score is not None:
            per_class[c].append((score, label, vt))
            n_found += 1
        else:
            n_missing += 1

    print(f"\nScored: {n_found}, Missing: {n_missing}")
    for c in [1, 2, 3]:
        entries = per_class[c]
        print(f"  C{c}: {len(entries)} entries")

    if dry_run:
        print("\n[dry-run] Not writing files")
        return None

    any_saved = False
    for c in [1, 2, 3]:
        entries = per_class[c]
        if not entries:
            print(f"  C{c}: no entries — skipping")
            continue
        preds  = np.array([e[0] for e in entries], dtype=np.float32)
        labels = np.array([e[1] for e in entries], dtype=np.int8)
        vids   = np.array([e[2] for e in entries])
        pf = _EVAL / f"{_MP2_METHOD}_{_SUPP_TAG}_c{c}_preds.npy"
        lf = _EVAL / f"{_MP2_METHOD}_{_SUPP_TAG}_c{c}_labels.npy"
        vf = _EVAL / f"{_MP2_METHOD}_{_SUPP_TAG}_c{c}_vt_ids.npy"
        np.save(pf, preds)
        np.save(lf, labels)
        np.save(vf, vids)
        print(f"  C{c}: saved {len(preds)} entries → {pf.name}")
        any_saved = True

    return _MP2_METHOD if any_saved else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="MutPred2 output CSV file")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse/report only — skip writing files and skip merge/restratify.")
    args = ap.parse_args()

    method = run(Path(args.csv), dry_run=args.dry_run)

    if method is None:
        print("\nNo entries written.")
        return

    # The merge + restratify steps that used to run here are obsolete.  Every row
    # now carries its blind_test_class in datasets/sfvcfp_rows.csv.gz, and the
    # canonical class assignment was verified identical to what restratify wrote
    # (16,277/16,277).  Scores are aligned to canonical rows by
    # repro_test/map_old_vcfp_preds_to_canonical.py instead.
    print("\nDone.")


if __name__ == "__main__":
    main()
