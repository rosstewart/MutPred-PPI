#!/usr/bin/env python
"""Prepare AlphaFold3 inputs for every complex not already folded.

A standing pipeline step, not an anomaly report: it always runs, always reports
coverage, and exits cleanly when nothing is missing. Run it after any change to
the canonical datasets or after merging a new batch of AF3 output, fold whatever
it emits on a GPU machine, then run `canonicalize_structures.py` to merge the
results back in. See `docs/DATA_PREPARATION.md` for the full chain.

    required  = every unordered (interactor, partner) pair the datasets need
    present   = every pair the canonical structure manifest already covers
    to fold   = required - present

**Required pairs are the UNION of all five canonical datasets, not any one of
them alone.** `sahni_fragoza_varchamp_all_mapped090826` is the largest and
usually a superset, but it is a POOLED, conflict-resolved table: a row whose
label disagrees across sources can be dropped during pooling even though the
same (interactor, partner, mutation) is present and unconflicted in a smaller
dataset. Relying on the pooled table alone would silently under-count on the day
such a pair exists -- the union is the only definition of "required" that cannot
miss one. (Verified 2026-09-10: zero pairs are lost this way today, i.e. the
pooled table happens to be a strict superset -- but nothing guarantees that
survives a future mapping change, so this does not rely on it.)

A pair counts as PRESENT if the manifest has it in EITHER chain order: `(A, B)`
and `(B, A)` are the same unordered pair, and `canonicalize_structures.py`
already normalises orientation at load time rather than trusting the filename.

JSON EMISSION IS DELEGATED, NOT DUPLICATED
------------------------------------------
This replaces `find_missing_af3_structures.py` (2026-09-10), which had its own
inline JSON writer. That writer named files `{a}-{b}.json` -- the single-hyphen
scheme the repo deliberately abandoned, because `O14787-2-Q13207` cannot be
split back into two accessions. It also called its output "AF3-server-format"
while emitting the *local* dialect, and hardcoded `modelSeeds: [1]`.

Rather than patch three bugs in a second copy, this hands the pairs to
`inference/00_make_af3_json_input.py`, the one JSON writer: it already produces
`{id_a}__{id_b}.json`, supports `--format local|server` and `--seeds`, and
normalises non-standard residues. One writer, already covered by tests.

Usage:
    conda run -n ppi python src/data_processing/prepare_af3_inputs.py
    conda run -n ppi python src/data_processing/prepare_af3_inputs.py \\
        --out-dir datasets/af3_inputs_to_fold --format server --seeds 1
"""
from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path

from paths import DATASETS_DIR
from utils.gcv_common import DATASET_CONFIGS, load_data, load_sequences

# `00_make_af3_json_input` is not a valid identifier, so it cannot be imported
# with `from ... import`; importlib takes the name as a string. Importing rather
# than shelling out keeps the failure modes (missing sequence, bad residue) as
# exceptions we can report per pair.
_af3 = importlib.import_module("inference.00_make_af3_json_input")

CANONICAL_DIR = DATASETS_DIR / "af3_structures_canonical"
MANIFEST = CANONICAL_DIR / "manifest.csv"
MISSING_CSV = CANONICAL_DIR / "missing_pairs.csv"
DEFAULT_OUT_DIR = DATASETS_DIR / "af3_inputs_to_fold"


def required_pairs() -> set[tuple[str, str]]:
    """Union of unordered (interactor, partner) pairs across all five datasets."""
    pairs: set[tuple[str, str]] = set()
    for cfg in DATASET_CONFIGS.values():
        df = load_data(cfg)
        pairs.update(tuple(sorted((a, b)))
                     for a, b in zip(df["interactor"], df["partner"]))
    return pairs


def present_pairs(manifest_path: Path = MANIFEST) -> set[tuple[str, str]]:
    """Pairs the canonical manifest already covers, in either chain order.

    A missing manifest means nothing has been folded yet -- a legitimate cold
    start, in which case every required pair needs folding. That is not an
    error, so it is not raised as one.
    """
    if not manifest_path.exists():
        print(f"[note] no manifest at {manifest_path} -- treating every pair as "
              f"unfolded (cold start)")
        return set()
    pairs: set[tuple[str, str]] = set()
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            pairs.add(tuple(sorted((row["chain_a_accession"],
                                    row["chain_b_accession"]))))
    return pairs


def write_pair_csv(pairs: list[tuple[str, str]], sequences: dict, path: Path
                   ) -> tuple[int, list[tuple[str, str]]]:
    """4-column CSV in the schema `00_make_af3_json_input.py --csv` expects.

    Returns `(n_written, no_sequence_pairs)`. A pair whose sequence we do not
    have cannot be folded and is reported rather than silently dropped.
    """
    no_seq = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["interactor", "partner",
                    "interactor_sequence", "partner_sequence"])
        n = 0
        for a, b in pairs:
            sa, sb = sequences.get(a), sequences.get(b)
            if not sa or not sb:
                no_seq.append((a, b))
                continue
            w.writerow([a, b, sa, sb])
            n += 1
    return n, no_seq


def run(out_dir: Path | None = None, fmt: str = "local", seeds: int = 1,
        skip_existing: bool = False, dry_run: bool = False) -> list[tuple[str, str]]:
    required = required_pairs()
    present = present_pairs()
    to_fold = sorted(required - present)

    covered = len(required & present)
    pct = 100.0 * covered / len(required) if required else 100.0
    print(f"required (union of all {len(DATASET_CONFIGS)} canonical datasets): "
          f"{len(required)}")
    print(f"already folded: {covered} ({pct:.1f}%)   "
          f"manifest holds {len(present)} pairs")
    print(f"to fold: {len(to_fold)}")

    # The pair list is written unconditionally -- including when empty -- so it
    # is always an accurate record of the last run rather than a stale one.
    MISSING_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(MISSING_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["accession_a", "accession_b"])
        w.writerows(to_fold)
    print(f"pair list -> {MISSING_CSV}")

    if not to_fold:
        print("\nAll required complexes are already folded. Nothing to prepare.")
        return []

    if dry_run:
        print("\n[dry-run] Not writing AF3 inputs")
        return to_fold

    out_dir = Path(out_dir or DEFAULT_OUT_DIR)
    sequences = load_sequences()
    pair_csv = out_dir / "pairs_to_fold.csv"
    n_pairs, no_seq = write_pair_csv(to_fold, sequences, pair_csv)
    if no_seq:
        print(f"\n[warn] {len(no_seq)} pair(s) have no sequence and cannot be "
              f"folded; they are in {MISSING_CSV} but not in the AF3 inputs:")
        for a, b in no_seq[:10]:
            print(f"    {a} / {b}")
    print(f"\n{n_pairs} pairs -> {pair_csv}")

    # Delegate to the one JSON writer, exactly as a user would invoke it.
    argv = [str(_af3.__file__), "--csv", str(pair_csv), str(out_dir),
            "--format", fmt, "--seeds", str(seeds)]
    if skip_existing:
        argv.append("--skip-existing")
    old_argv, sys.argv = sys.argv, argv
    try:
        rc = _af3.main()
    finally:
        sys.argv = old_argv
    if rc:
        raise RuntimeError(
            f"00_make_af3_json_input.py reported failures for some pairs "
            f"(exit {rc}); see the FAILED lines above")

    print(f"\nNext: fold {out_dir}/*.json on a GPU machine, then run\n"
          f"  src/data_processing/canonicalize_structures.py\n"
          f"to merge the results into {CANONICAL_DIR}.")
    return to_fold


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=None,
                   help=f"where to write AF3 input JSONs "
                        f"(default: {DEFAULT_OUT_DIR})")
    p.add_argument("--format", choices=("local", "server"), default="local",
                   help="AF3 JSON dialect. 'local' for the AlphaFold3 executable, "
                        "'server' for the AlphaFold Server web UI. A file in one "
                        "dialect will NOT run under the other.")
    p.add_argument("--seeds", type=int, default=1, help="model seeds (default 1)")
    p.add_argument("--skip-existing", action="store_true",
                   help="leave already-written JSONs alone (resume a batch)")
    p.add_argument("--dry-run", action="store_true",
                   help="report coverage and write the pair list only")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    run(out_dir=a.out_dir, fmt=a.format, seeds=a.seeds,
        skip_existing=a.skip_existing, dry_run=a.dry_run)
