#!/usr/bin/env python
"""Rebuild the contact-graph store from the canonical structures.

Replaces the `.mat` migration. That migration was lossless per file, but it had
to CHOOSE among files, and the choice was arbitrary: **10,006 of the 19,978
variant-DB pair_keys have more than one source `.mat`**, and on an
orientation-invariant check (edge count at equal size, so reciprocal `A_B`/`B_A`
files are not falsely flagged) **28 of 150 sampled multi-source keys hold
genuinely different graphs** -- e.g. `Q9BXJ9_Q9BSU3` 10,362 edges vs 10,256. The
store kept whichever was written first, so roughly 1,900 pairs scored against an
arbitrarily selected contact map.

`canonicalize_structures.py` already solved the same problem one level up: one
structure per pair, chosen by mean pLDDT with a `.cif` tie-break. Deriving the
graphs from those structures makes graph identity follow structure identity by
construction, so there is no second tie-break to invent and no `.mat` tree to
keep alive.

The contact rule is transcribed from `src/inference/01_make_contact_graphs_and_fasta.py`:
ANY pair of atoms within 4.5 A puts an edge between their residues, self-residue
pairs excluded, via a KD-tree. Same threshold, same metric, same exclusion.

Self-loops are NOT stored -- they are exactly the N diagonal entries and the
store adds them on read, unconditionally.

Usage:
    python src/data_processing/rebuild_graphs_from_structures.py \\
        --structures datasets/af3_structures_canonical \\
        --out datasets/training_eval/contact_graphs_rebuilt.h5

    # verify against the migrated store before promoting
    python src/data_processing/rebuild_graphs_from_structures.py \\
        --structures datasets/af3_structures_variant_dbs_canonical \\
        --out datasets/variant_dbs/contact_graphs_rebuilt.h5 \\
        --compare-to datasets/variant_dbs/contact_graphs.h5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from joblib import Parallel, delayed

from contact_graphs import (
    DEFAULT_THRESHOLD, ContactGraphStore, contact_graph_from_structure,
)

# The contact rule lives in `contact_graphs.contact_graph_from_structure` -- one
# definition, shared with the inference pipeline, so the two cannot drift.
EDGE_DIST_THRESHOLD = DEFAULT_THRESHOLD


def _one(path: Path):
    try:
        return path, contact_graph_from_structure(path, EDGE_DIST_THRESHOLD)
    except Exception as e:
        return path, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structures", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-to", default=None,
                    help="existing store to diff against (edge counts per pair_key)")
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(f for d in args.structures for f in Path(d).glob("*.cif.gz"))
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} canonical structures", flush=True)

    results = Parallel(n_jobs=args.n_jobs, verbose=1)(delayed(_one)(f) for f in files)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    store = ContactGraphStore(out, mode="w")
    n_ok = n_skip = 0
    errors: list[str] = []
    built: dict[str, int] = {}
    for path, res in results:
        if res is None or isinstance(res, str):
            n_skip += 1
            if isinstance(res, str) and len(errors) < 5:
                errors.append(f"{path.name}: {res}")
            continue
        seq_a, seq_b, ei = res
        key = store.put(seq_a, seq_b, ei, source=str(path),
                        threshold=EDGE_DIST_THRESHOLD)
        built[key] = int(ei.shape[1]) if ei.size else 0
        n_ok += 1
    print(f"\nbuilt {n_ok} graphs, skipped {n_skip}")
    for e in errors:
        print(f"   skip {e}")

    if args.compare_to and Path(args.compare_to).exists():
        old = ContactGraphStore(args.compare_to)
        same = diff = only_new = 0
        examples = []
        old_counts = {m["key"]: m["nnz"] for m in old.meta()}
        for key, nnz in built.items():
            if key not in old_counts:
                only_new += 1
            elif old_counts[key] == nnz:
                same += 1
            else:
                diff += 1
                if len(examples) < 5:
                    examples.append(f"{key}: migrated {old_counts[key]} vs rebuilt {nnz}")
        only_old = len(set(old_counts) - set(built))
        print(f"\nvs {args.compare_to}:")
        print(f"  identical edge count : {same}")
        print(f"  DIFFERENT edge count : {diff}")
        print(f"  only in rebuilt      : {only_new}")
        print(f"  only in migrated     : {only_old}")
        for e in examples:
            print(f"     {e}")
        old.close()

    store.close()
    print(f"\nstore -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
