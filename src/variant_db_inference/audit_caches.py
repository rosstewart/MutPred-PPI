#!/usr/bin/env python
"""Audit a variant database's caches and say exactly why each row is unscoreable.

A row may legitimately be skipped for only two reasons: the pair has no AF3
structure, or the mutation does not apply to the interactor sequence. Everything
else -- a missing ProtT5 embedding, an embedding left over from an older version
of a sequence, a subgraph that was never written -- is a cache defect and is
repairable. This script separates the two so a repair run can target the second
group and a reader can trust the first.

    conda run -n ppi python src/variant_db_inference/audit_caches.py --dataset clinvar
    conda run -n ppi python src/variant_db_inference/audit_caches.py --dataset all --proteins-out fix.txt

`--proteins-out` writes the accessions whose embeddings must be (re)generated,
one per line, ready for `precompute_prott5.py`.

Staleness is detected by length: a per-residue embedding must have exactly one
row per residue of the sequence it claims to describe (`check_embedding_lengths`).
An embedding generated from a different version of a protein therefore fails the
check rather than silently producing predictions for the wrong sequence.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py

from contact_graphs import ContactGraphStore, check_embedding_lengths
from paths import DATA_ROOT, DATASETS_DIR
from utils import mutations
from variant_db_inference import variant_rows as vr

STORE = DATASETS_DIR / "variant_dbs" / "contact_graphs.h5"

# Reasons a row cannot be scored. Only the first two are legitimate; the rest
# mean a cache needs rebuilding.
LEGITIMATE = ("no AF3 structure", "mutation does not apply")
REPAIRABLE = ("interactor embedding missing", "partner embedding missing",
              "interactor embedding stale", "partner embedding stale",
              "subgraph missing")


def _embedding_index(db: str) -> tuple[dict, str]:
    """{accession: n_residues} from whichever embedding cache the db has."""
    emb = DATA_ROOT / db / "prott5_embeddings.h5"
    if emb.exists():
        with h5py.File(emb, "r") as f:
            return {k: f[k].shape[0] for k in f}, str(emb)
    return {}, str(emb)


def _subgraph_index(db: str) -> tuple[dict, str]:
    """{pair_group: {variant_keys}} from the subgraph H5, or {} when absent."""
    sg = DATA_ROOT / db / "prott5_subgraphs.h5"
    if not sg.exists():
        return {}, str(sg)
    with h5py.File(sg, "r") as f:
        return {k: set(f[k].keys()) for k in f}, str(sg)


def audit(db: str, store_path: str = str(STORE)) -> tuple[Counter, set]:
    """Per-row reasons, and the accessions needing (re)embedding."""
    stats: Counter = Counter()
    need: set[str] = set()

    emb_len, emb_path = _embedding_index(db)
    subgraphs, sg_path = _subgraph_index(db)
    use_subgraph = bool(subgraphs)
    print(f"{db}: embeddings={len(emb_len)} proteins ({emb_path})", flush=True)
    print(f"{db}: subgraphs={len(subgraphs)} pairs ({sg_path})", flush=True)

    store = ContactGraphStore(store_path)
    rows_path = DATASETS_DIR / "variant_dbs" / f"{db}_rows.csv.gz"
    for r in vr.iter_table_rows(db, str(rows_path), stats=stats):
        stats["rows"] += 1
        i, p, m = r["interactor"], r["partner"], r["mutation"]
        iseq, pseq = r["interactor_sequence"], r["partner_sequence"]

        # 1. structure -- a legitimate skip
        if store.get(iseq, pseq) is None:
            stats["no AF3 structure"] += 1
            continue

        # 2. mutation validity -- a legitimate skip
        if mutations.apply(iseq, m) is None:
            stats["mutation does not apply"] += 1
            continue

        # 3. subgraph presence, when the db uses that path
        if use_subgraph:
            variants = subgraphs.get(f"{i}_{p}")
            if variants is None or vr.to_zero_based(m) not in variants:
                stats["subgraph missing"] += 1
                need.update((i, p))
                continue

        # 4. embedding presence and freshness
        for acc, seq, side in ((i, iseq, "interactor"), (p, pseq, "partner")):
            n = emb_len.get(acc)
            if emb_len and n is None:
                stats[f"{side} embedding missing"] += 1
                need.add(acc)
            elif n is not None and n != len(seq):
                stats[f"{side} embedding stale"] += 1
                need.add(acc)
        stats["scoreable"] += 1

    store.close()
    return stats, need


def report(db: str, stats: Counter, need: set) -> None:
    rows = stats["rows"]
    print(f"\n=== {db}: {rows:,} rows ===")
    legit = sum(stats[k] for k in LEGITIMATE)
    repair = sum(stats[k] for k in REPAIRABLE)
    print(f"  scoreable                    {stats['scoreable']:>10,}")
    print(f"  legitimately skipped         {legit:>10,}")
    for k in LEGITIMATE:
        if stats[k]:
            print(f"    {k:<26} {stats[k]:>10,}")
    print(f"  REPAIRABLE (cache defect)    {repair:>10,}")
    for k in REPAIRABLE:
        if stats[k]:
            print(f"    {k:<26} {stats[k]:>10,}")
    other = {k: v for k, v in stats.items()
             if k not in LEGITIMATE + REPAIRABLE + ("rows", "scoreable")}
    for k, v in sorted(other.items()):
        print(f"    {k:<26} {v:>10,}")
    print(f"  proteins needing (re)embedding: {len(need):,}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    choices=sorted(vr.DB_SOURCES) + ["all"])
    ap.add_argument("--store", default=str(STORE))
    ap.add_argument("--proteins-out", default=None,
                    help="write accessions needing (re)embedding, one per line")
    args = ap.parse_args()

    dbs = sorted(vr.DB_SOURCES) if args.dataset == "all" else [args.dataset]
    all_need: dict[str, set] = {}
    for db in dbs:
        stats, need = audit(db, args.store)
        report(db, stats, need)
        all_need[db] = need

    if args.proteins_out:
        out = Path(args.proteins_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            for db in dbs:
                for acc in sorted(all_need[db]):
                    fh.write(f"{db}\t{acc}\n")
        total = sum(len(v) for v in all_need.values())
        print(f"\nWrote {total:,} (db, accession) pairs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
