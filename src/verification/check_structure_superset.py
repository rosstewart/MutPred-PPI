#!/usr/bin/env python3
"""Would promoting a new canonical structure set cost any row its structure?

Gates the promotion in `run_post_canonicalization_chain.sh`.

The check is ROW-DRIVEN, and deliberately not a manifest-to-manifest diff. Two
weaker checks were tried first and both gave false answers:

  * comparing filenames / accession pairs said "superset" while six VarChAMP
    pairs had silently lost their usable structure -- a ProtVar model of a
    DIFFERENT ISOFORM won the pLDDT tie-break for the same accession pair, so the
    filename matched and the chain sequence did not;
  * comparing every sequence-pair in the old manifest flagged 28 "regressions"
    that no current row actually needs, which would block a promotion that loses
    nothing.

What matters is coverage of the pairs the canonical tables ask for, keyed the way
`StructureIndex` resolves them: the pair of chain-sequence hashes. Exits non-zero
only if a dataset would end up with fewer resolvable rows than it has now.
"""
import argparse
import csv
import sys


def _seq_pairs(manifest):
    with open(manifest) as fh:
        return {frozenset((r["seq_a_sha"], r["seq_b_sha"])) for r in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True)
    ap.add_argument("--live", required=True)
    args = ap.parse_args()

    from contact_graphs import sha
    from utils.gcv_common import DATASET_CONFIGS, dataset_config, load_data

    new = _seq_pairs(f"{args.new}/manifest.csv")
    live = _seq_pairs(f"{args.live}/manifest.csv")
    print(f"live: {len(live)} sequence-pairs    new: {len(new)} sequence-pairs\n")

    regressed = False
    for ds in DATASET_CONFIGS:
        df = load_data(dataset_config(ds))
        need = {frozenset((sha(a), sha(b)))
                for a, b in zip(df["interactor_sequence"], df["partner_sequence"])}
        in_live, in_new = len(need & live), len(need & new)
        delta = in_new - in_live
        flag = ""
        if delta < 0:
            flag = "   <-- REGRESSION"
            regressed = True
        print(f"  {ds:<45} pairs {len(need):>6}   live {in_live:>6}   "
              f"new {in_new:>6}   {delta:+d}{flag}")

    if regressed:
        print("\nREFUSING to promote: at least one dataset loses coverage.")
        return 1
    print("\nOK: no dataset loses a resolvable pair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
