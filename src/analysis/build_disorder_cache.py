#!/usr/bin/env python
"""Per-residue intrinsic disorder for the benchmark sequences, from AIUPred.

WHY NOT ALPHAFOLD'S OWN MEASURE. AlphaFold 3 reports `fraction_disordered` in
its summary confidences, but that is one scalar per COMPLEX and it exists only
where a summary file was kept, which is 66% of the benchmark rows. Supplementary
Figure S5B asks whether the complex-confidence trend is explained by disorder,
and answering that needs the disorder state of the MUTATED RESIDUE, for every
row. AIUPred predicts from sequence, so coverage is 100%.

THRESHOLD. AIUPred applies a logistic calibration internally so that 0.5 is the
binarisation cutoff (`aiupred/predictor.py::_apply_disorder_cutoff`), and uses
0.5 for its own region calls. It is the tool's value, not one chosen here.

Output format (pickle)::

    {"<sha1 of the sequence>": float32[len(sequence)], ...}

KEYED BY SEQUENCE HASH, not by accession. The same protein appears under several
accessions across the row tables (isoforms, and the interactor/partner roles
swap), while disorder is a property of the sequence alone. Hashing the sequence
means each distinct sequence is predicted once and every accession that carries
it resolves to the same array, with no isoform fallback.

Usage:
  python src/analysis/build_disorder_cache.py --output /tmp/disorder_cache.pkl
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

from paths import ANNOTATIONS_DIR
from utils.legacy_guard import DATASET_SUFFIX

DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "disorder_cache.pkl"
DEFAULT_DATASET = f"sahni_fragoza{DATASET_SUFFIX}"

#: AIUPred's own calibrated cutoff; see the module docstring.
DISORDER_CUTOFF = 0.5


def sequence_key(sequence: str) -> str:
    return hashlib.sha1(str(sequence).encode()).hexdigest()


def lookup(cache: dict, sequence: str):
    """Per-residue disorder for a sequence, or None if it was never predicted."""
    return cache.get(sequence_key(sequence))


def residue_disorder(cache: dict, sequence: str, pos1: int):
    """Disorder score at a 1-BASED position, or None when out of range."""
    arr = lookup(cache, sequence)
    if arr is None or pos1 < 1 or pos1 > len(arr):
        return None
    return float(arr[pos1 - 1])


def collect_sequences(dataset: str) -> dict[str, str]:
    """Every distinct sequence in the dataset, on either side of the pair."""
    from utils.gcv_common import dataset_config, load_data

    df = load_data(dataset_config(dataset))
    out: dict[str, str] = {}
    for column in ("interactor_sequence", "partner_sequence"):
        for seq in df[column]:
            out[sequence_key(seq)] = str(seq)
    return out


def build(sequences: dict[str, str], force_cpu: bool = False) -> dict:
    import aiupred

    predictor = aiupred.AIUPred(force_cpu=force_cpu)
    cache, failed = {}, []
    for n, (key, seq) in enumerate(sequences.items(), 1):
        try:
            cache[key] = predictor.predict_disorder(seq).astype(np.float32)
        except Exception as exc:                  # noqa: BLE001 -- reported, not raised
            failed.append((key, f"{type(exc).__name__}: {exc}"))
        if n % 200 == 0:
            print(f"  {n}/{len(sequences)}", flush=True)
    for key, err in failed[:10]:
        print(f"  FAILED {key[:12]}: {err}")
    if failed:
        print(f"  {len(failed)} sequences failed")
    return cache


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"dataset whose sequences to cover (default: {DEFAULT_DATASET})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"output pickle (default: {DEFAULT_OUTPUT})")
    p.add_argument("--force-cpu", action="store_true", help="do not use a GPU")
    p.add_argument("--force", action="store_true",
                   help="allow overwriting a file under datasets/annotations")
    args = p.parse_args()

    out = args.output.resolve()
    if not args.force and ANNOTATIONS_DIR.resolve() in out.parents and out.exists():
        p.error(f"refusing to overwrite {out}; write to scratch and diff first, or pass --force")

    sequences = collect_sequences(args.dataset)
    print(f"{len(sequences)} distinct sequences in {args.dataset}")
    cache = build(sequences, force_cpu=args.force_cpu)

    allv = np.concatenate(list(cache.values()))
    print(f"Built {len(cache)} sequences, {len(allv):,} residues; "
          f"{100.0 * float((allv > DISORDER_CUTOFF).mean()):.1f}% disordered at "
          f"> {DISORDER_CUTOFF}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
