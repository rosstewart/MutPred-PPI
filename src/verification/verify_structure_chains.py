#!/usr/bin/env python3
"""Does `Structures.find_pdb` hand each row the chain its mutation belongs to?

The structure-based comparators (SAAMBE-3D, MutPPI, MutPPI+) are given a PDB and
a CHAIN LETTER, and apply the mutation at a position within that chain. The
letter comes from the canonical manifest, and `Structures.find` falls back to
"A" when the lookup misses -- so a complex stored partner-first, or a manifest
row whose chain order disagrees with its own file, silently puts every mutation
on the wrong protein. Positions past the end of the wrong chain error out (which
is visible); positions inside it are scored against the wrong residue (which is
not).

This checks the property end to end: read back the chain `find_pdb` returned and
require its sequence to BE the interactor's. Run it after any rebuild of the
canonical structure set.

    conda run -n ppi python src/verification/verify_structure_chains.py \
        --dataset sahni_fragoza --out chains.txt
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from joblib import Parallel, delayed

from contact_graphs import _is_polymer_residue, residue_to_one, sha
from utils.gcv_common import dataset_config, load_data
from utils.structures import Structures

_BAD = ("FLIPPED", "WRONG_CHAIN", "CHAIN_ABSENT", "seq_differs")


def _check(structures, interactor, partner, seq_i, seq_p):
    """Classify one pair: does the returned chain really hold the interactor?"""
    from Bio.PDB import PDBParser
    try:
        path, chain = structures.find_pdb(interactor=seq_i, partner=seq_p)
        if path is None:
            return ("no_structure", interactor, partner, "")
        model = PDBParser(QUIET=True).get_structure("x", str(path))[0]
        chains = {c.id: "".join(residue_to_one(r.get_resname())
                                for r in c if _is_polymer_residue(r))
                  for c in model}
        if chain not in chains:
            return ("CHAIN_ABSENT", interactor, partner,
                    f"got {chain!r}, have {list(chains)}")
        got = chains[chain]
        if sha(got) == sha(seq_i):
            return ("ok", interactor, partner, "")
        if sha(got) == sha(seq_p):
            return ("FLIPPED", interactor, partner, f"chain {chain} holds the PARTNER")
        if len(got) == len(seq_i):
            return ("seq_differs", interactor, partner, "same length, different seq")
        return ("WRONG_CHAIN", interactor, partner,
                f"chain {chain} len={len(got)} vs interactor len={len(seq_i)}")
    except Exception as exc:                                  # noqa: BLE001
        return ("error", interactor, partner, f"{type(exc).__name__}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="dataset name or short alias (see utils.gcv_common)")
    ap.add_argument("--out", default="-",
                    help="report file; '-' (default) writes to stdout")
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--pdb-cache", default="/tmp/_pdbc_verify")
    args = ap.parse_args()

    table = load_data(dataset_config(args.dataset))
    pairs = table.drop_duplicates(subset=["interactor", "partner"])[
        ["interactor", "partner", "interactor_sequence", "partner_sequence"]]

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    try:
        out.write(f"{args.dataset}: {len(table)} rows, {len(pairs)} unique pairs\n")
        out.flush()

        structures = Structures(pdb_cache=args.pdb_cache)
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(_check)(structures, r.interactor, r.partner,
                            r.interactor_sequence, r.partner_sequence)
            for r in pairs.itertuples())

        for kind, count in Counter(k for k, _, _, _ in results).most_common():
            out.write(f"  {kind:<14} {count}\n")

        bad = [r for r in results if r[0] in _BAD]
        out.write(f"\nproblems ({len(bad)}):\n")
        for kind, i, p, msg in bad[:25]:
            out.write(f"   {kind:<13} {i}/{p}  {msg}\n")
    finally:
        if out is not sys.stdout:
            out.close()

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
