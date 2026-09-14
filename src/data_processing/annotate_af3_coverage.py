#!/usr/bin/env python
"""Record, in the mapping CSVs, which complexes have no AlphaFold3 structure.

Stage 2a of the data-preparation chain: it runs after the structures have been
canonicalised and before `training_sets/prepare_gcv_tables.py` builds the GCV
layer.

A row whose (interactor, partner) pair is absent from
`datasets/af3_structures_canonical/manifest.csv` cannot be scored by any
structure-based method -- there is no contact graph for it. Those rows are
marked `af3_failed = True` **in the raw mapping CSVs**, where they are kept for
provenance, and `prepare_gcv_tables.py` then drops them before assigning
`row_index`. Nothing downstream ever sees them.

Writing the flag here rather than inside `prepare_gcv_tables.py` keeps the two
concerns separate: this one answers "what did AlphaFold3 actually produce?",
which is a property of the structure set and changes every time a fold batch
lands; the other builds a train/eval layer from whatever is available.

Pairs are compared unordered -- `(A, B)` and `(B, A)` are the same complex, and
the canonical store resolves chain orientation at load time.

Usage:
    conda run -n ppi python src/data_processing/annotate_af3_coverage.py
    conda run -n ppi python src/data_processing/annotate_af3_coverage.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

from paths import DATASETS_DIR, MAPPING_DIR

MANIFEST = DATASETS_DIR / "af3_structures_canonical" / "manifest.csv"
MAPPING_CSV_DIR = MAPPING_DIR / "datasets"
COLUMN = "af3_failed"


def folded_pairs(manifest: Path = MANIFEST) -> set[tuple[str, str]]:
    """Unordered accession pairs the canonical structure manifest covers."""
    if not manifest.exists():
        sys.exit(f"ERROR: {manifest} not found -- run "
                 f"src/data_processing/canonicalize_structures.py first.")
    pairs = set()
    with open(manifest) as f:
        for row in csv.DictReader(f):
            pairs.add(tuple(sorted((row["chain_a_accession"],
                                    row["chain_b_accession"]))))
    return pairs


def mapping_csvs(root: Path = MAPPING_CSV_DIR) -> list[Path]:
    """Every mapped dataset CSV, including the single_source/ variants."""
    if not root.is_dir():
        sys.exit(f"ERROR: {root} not found -- run the mapping notebook first "
                 f"(notebooks/map_ppi_datasets.py).")
    return sorted(list(root.glob("*.csv")) + list(root.glob("*/*.csv")))


def annotate(dry_run: bool = False, manifest: Path = MANIFEST,
             root: Path = MAPPING_CSV_DIR) -> int:
    folded = folded_pairs(manifest)
    print(f"canonical manifest: {len(folded)} folded pairs\n")

    total_failed = 0
    for path in mapping_csvs(root):
        df = pd.read_csv(path)
        if not {"interactor", "partner"} <= set(df.columns):
            print(f"  skip (no interactor/partner): {path.name}")
            continue
        failed = [tuple(sorted((a, b))) not in folded
                  for a, b in zip(df["interactor"], df["partner"])]
        n = sum(failed)
        total_failed += n
        pct = 100.0 * n / len(df) if len(df) else 0.0
        rel = path.relative_to(root)
        print(f"  {str(rel):52s} rows={len(df):6d}  {COLUMN}={n:5d} ({pct:.2f}%)")
        if not dry_run:
            df[COLUMN] = failed
            df.to_csv(path, index=False)

    print(f"\n{'[dry-run] ' if dry_run else ''}{total_failed} row(s) flagged "
          f"{COLUMN}=True across {len(mapping_csvs(root))} file(s)")
    if not dry_run:
        print(f"Column written in place. `prepare_gcv_tables.py` drops these "
              f"rows before assigning row_index; they stay here for provenance.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--mapping-dir", type=Path, default=MAPPING_CSV_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="report coverage without modifying the CSVs")
    args = ap.parse_args()
    return annotate(args.dry_run, args.manifest, args.mapping_dir)


if __name__ == "__main__":
    sys.exit(main())
