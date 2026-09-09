#!/usr/bin/env python
"""Emit one self-contained table per variant database.

    datasets/variant_dbs/{db}_rows.csv.gz

Replaces the scattered `*.pkl` annotation files at the point of USE. Each row is
a `(interactor, partner, mutation)` triplet that has already been mutation-
validated against the sequence its own graph was built from, plus every
annotation a downstream analysis needs, so an analysis can read one file instead
of joining six pickles with three different key conventions.

The pickles are not deleted -- they remain the source, and this is a derived
view. What goes away is the need for every consumer to know their formats.

WHAT IS AND IS NOT IN A ROW
---------------------------
Sequences are NOT inlined. They live once per pair in the contact-graph store,
and `pair_key` points at them; inlining would repeat a 1,000-residue sequence
across every one of gnomad's 10.5M rows. Identity still comes from content --
`pair_key` IS the content address.

BASE CONVENTION
---------------
`mutation` is 1-BASED, matching the triplet tables and every annotation file
(verified: 99-100% of keys validate at `pos-1` against the WT sequence, against
an ~8% coincidence floor at `pos`). The 0-based half of the pipeline -- FASTA
headers, ProtT5 keys, subgraph H5 variant keys -- is converted only through
`variant_rows.to_zero_based`, never inline.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

_PUB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PUB / "src"))
from contact_graphs import ContactGraphStore, pair_key  # noqa: E402
from paths import ANNOTATIONS_DIR, ANNOTATIONS_LICENSED_DIR, DATASETS_DIR  # noqa: E402
from variant_db_inference import variant_rows as vr  # noqa: E402

_ROOT = Path("/data/ross/ppi_lossgain/interaction_loss")
STORE = DATASETS_DIR / "variant_dbs" / "contact_graphs.h5"
OUT_DIR = DATASETS_DIR / "variant_dbs"
BIOGRID_DIRBIND = _ROOT / "biogrid" / "biogrid_dirbind_uniprot_to_interactors.pkl"

# BioGRID direct-binding is a FILTER, not a column. Every emitted row is a
# physical-direct-binding pair, matching the legacy pipeline, so there is nothing
# to flag -- a constant column is noise. Rows failing it are dropped and counted
# in `stats`. This is load-bearing rather than a formality: cosmic, gnomad, hgmd
# and autism are already 100% dirbind, but **clinvar is only 93.3% -- 455 of its
# 6,836 pairs are not dirbind edges** (349 where both proteins are in the map but
# the pair is not an edge, 92 with one accession absent, 14 with both). clinvar
# is also the one database whose derivation notebook was never migrated, which
# is consistent. All 355 clinvar homodimers ARE dirbind edges, so self-pairs are
# not the explanation.
#
# `variant_subset.pkl` is deliberately NOT surfaced. It is a folding-budget cap
# (`interactor_count < 10`, stopping at `len(complex_subset) == 600`), and the
# question it appears to answer -- was this complex folded? -- is already
# answered by the pair having a graph, which every row requires by construction.
EXTRA_COLUMNS = {
    "clinvar": ["clinical_significance", "allele_frequency"],
    "gnomad":  ["allele_frequency"],
    "cosmic":  ["recurrence", "tumor_sites", "onco_tsg"],
    "hgmd":    [],
    "autism":  ["neurodev_label"],
}
# `interactor_len` / `partner_len` are deliberately absent: both are one store
# lookup from `pair_key`, and duplicating derived state is what this refactor
# removes. `pair_key` itself is NOT redundant -- accession -> sequence is
# ambiguous for the 111 conflated accessions, so it is the only record of which
# sequence the row was actually validated and scored against.
CORE_COLUMNS = ["interactor", "partner", "mutation", "pair_key",
                "clingen_moi", "in_embedding_store"]


def _load_pkl(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ── the one place the legacy "ACC MUT" string key is understood ──────────────
#
# Several annotation files key on a space-joined string. That is the composite
# key this refactor exists to remove, so it is parsed HERE, once, and every
# structure below is keyed on the canonical `(accession, mutation)` tuple. No
# code outside these three helpers builds or splits an "ACC MUT" string, and the
# emitted table has explicit columns instead.

def _variant_key(s: str) -> tuple[str, str] | None:
    """'P25054 S305R' -> ('P25054', 'S305R'). None when it is not that shape."""
    parts = s.split()
    return (parts[0], parts[1]) if len(parts) == 2 else None


def _keyed_set(strings) -> set[tuple[str, str]]:
    return {k for k in map(_variant_key, strings) if k is not None}


def _keyed_map(pairs) -> dict[tuple[str, str], object]:
    return {k: v for s, v in pairs for k in (_variant_key(s),) if k is not None}


def _load_tsv_map(path) -> dict[tuple[str, str], str]:
    """'ACC MUT\\tvalue' lines -> {(acc, mut): value}."""
    def rows():
        with open(path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    yield parts[0], parts[1]
    return _keyed_map(rows())


def load_annotations(db: str) -> dict:
    """Everything needed to annotate one DB's rows.

    Every lookup is keyed canonically: `(accession, mutation)` for per-variant
    annotations, `(interactor, mutation, partner)` for the dirbind subsets, bare
    accession for mode-of-inheritance. All mutations are 1-based.
    """
    a: dict = {}
    clingen = _load_pkl(ANNOTATIONS_DIR / "clingen_ar_ad_uniprot_sets.pkl")
    a["ar"], a["ad"] = clingen.get("AR", set()), clingen.get("AD", set())

    # The physical-direct-binding interactome, as an unordered pair set. This is
    # the legacy filter, recomputed rather than assumed present.
    dirbind = _load_pkl(BIOGRID_DIRBIND)
    a["dirbind"] = {frozenset((str(x), str(y)))
                    for x, ys in dirbind.items() for y in ys}

    if db == "clinvar":
        # The three significance tiers ARE clinvar's dirbind subsets, so
        # membership and significance are the same lookup.
        a["sig"] = {}
        for tier in ("pathogenic", "benign", "vus"):
            for i, m, p in _load_pkl(ANNOTATIONS_DIR / "clinvar"
                                     / f"{tier}_dirbind_variant_subset.pkl"):
                a["sig"][(i, m, p)] = tier
        a["af"] = _load_tsv_map(ANNOTATIONS_DIR / "benign_allele_frequencies.tsv")
    elif db == "gnomad":
        a["af"] = _load_tsv_map(ANNOTATIONS_DIR / "gnomad_allele_frequencies.tsv")
    elif db == "cosmic":
        a["sites"] = _keyed_map(
            _load_pkl(ANNOTATIONS_LICENSED_DIR / "vt_to_tumor_site.pkl").items())
        ot = _load_pkl(ANNOTATIONS_LICENSED_DIR / "onco_tsg_dict.pkl")
        a["onco"] = _keyed_set(ot.get("oncogene", set()))
        a["tsg"] = _keyed_set(ot.get("TSG", set()))
    elif db == "autism":
        a["labels"] = _keyed_map(
            _load_pkl(ANNOTATIONS_DIR / "autism" / "variant_label_dict.pkl").items())
    return a


def embedding_lookup(db: str):
    """`(interactor, partner, mutation_1b) -> bool`, or None if nothing to read.

    This is what actually limited the published runs: measured across clinvar,
    cosmic and gnomad, 100% of published predictions lie inside the store and
    none outside it. So the column records scoreability, and a 0 is a GPU cost
    rather than a defect.

    Two on-disk layouts, because the databases were built at different times:

        prott5_subgraphs.h5   clinvar/cosmic/gnomad   group per PAIR, one
                              dataset per variant -- a triplet is present or not
        prott5_embeddings.h5  hgmd/autism             one dataset per PROTEIN,
                              keyed 'ACC' and 'ACC MUT' -- a triplet is scoreable
                              when the interactor's WT and variant embeddings and
                              the partner's WT embedding all exist

    Variant keys are 0-based on disk in both, converted through
    `variant_rows.to_zero_based` and nowhere else.
    """
    import h5py

    sub = _ROOT / db / "prott5_subgraphs.h5"
    if sub.exists():
        triplets = set()
        with h5py.File(sub, "r") as f:
            for g in f:
                i, _, p = g.partition("_")
                if not p:
                    continue
                for v in f[g]:
                    try:
                        triplets.add((i, p, vr.to_one_based(v)))
                    except ValueError:
                        pass
        print(f"  embedding store: {len(triplets):,} triplets (subgraphs)", flush=True)
        return lambda i, p, m: (i, p, m) in triplets

    emb = _ROOT / db / "prott5_embeddings.h5"
    if emb.exists():
        with h5py.File(emb, "r") as f:
            keys = list(f.keys())
        # Normalised at the boundary: bare keys become the WT set, 'ACC MUT'
        # keys become (accession, mutation) tuples in 1-based form. Nothing
        # downstream rebuilds the string.
        wt = {k for k in keys if " " not in k}
        variants = set()
        for k in keys:
            kv = _variant_key(k)
            if kv is None:
                continue
            try:
                variants.add((kv[0], vr.to_one_based(kv[1])))
            except ValueError:
                pass
        print(f"  embedding store: {len(wt):,} proteins / {len(variants):,} "
              f"variants (per-protein)", flush=True)
        return lambda i, p, m: i in wt and p in wt and (i, m) in variants

    print("  embedding store: unavailable", flush=True)
    return None


def build(db: str, out_path: Path, store: ContactGraphStore,
          check_embeddings: bool = True) -> Counter:
    ann = load_annotations(db)
    seqs = vr.load_wt_sequences(db)
    variants = vr.load_fasta_variants(db)
    pairs = vr.store_pairs(db, OUT_DIR / "aliases.csv")

    has_embedding = embedding_lookup(db) if check_embeddings else None

    cols = CORE_COLUMNS + EXTRA_COLUMNS[db]
    stats, n = Counter(), 0
    with gzip.open(out_path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in vr.iter_rows(db, store, seqs=seqs, variants=variants,
                              pairs=pairs, stats=stats):
            i, p, m = r["interactor"], r["partner"], r["mutation"]
            iseq, pseq = r["interactor_sequence"], r["partner_sequence"]
            # Canonical keys. The triplet table's own order is (i, mutation, p).
            variant, triplet = (i, m), (i, m, p)

            if frozenset((i, p)) not in ann["dirbind"]:
                stats["pair is not a BioGRID direct-binding edge"] += 1
                continue

            row = {
                "interactor": i, "partner": p, "mutation": m,
                "pair_key": pair_key(iseq, pseq),
                "clingen_moi": "AR" if i in ann["ar"] else "AD" if i in ann["ad"] else "",
                "in_embedding_store": ("" if has_embedding is None
                                       else int(has_embedding(i, p, m))),
            }
            if db == "clinvar":
                row["clinical_significance"] = ann["sig"].get(triplet, "")
                row["allele_frequency"] = ann["af"].get(variant, "")
            elif db == "gnomad":
                row["allele_frequency"] = ann["af"].get(variant, "")
            elif db == "cosmic":
                sites = ann["sites"].get(variant, [])
                row["recurrence"] = len(sites)
                row["tumor_sites"] = ";".join(sites)
                row["onco_tsg"] = ("oncogene" if variant in ann["onco"]
                                   else "TSG" if variant in ann["tsg"] else "")
            elif db == "autism":
                lab = ann["labels"].get(variant)
                row["neurodev_label"] = "" if lab is None else int(lab)
            w.writerow(row)
            n += 1
    stats["rows written"] = n
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="all", choices=sorted(vr.DB_SOURCES) + ["all"])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--store", default=str(STORE))
    ap.add_argument("--no-embedding-check", action="store_true",
                    help="skip the ProtT5 subgraph scan (it reads a ~175 GB H5)")
    args = ap.parse_args()

    dbs = sorted(vr.DB_SOURCES) if args.db == "all" else [args.db]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ContactGraphStore(args.store)
    for db in dbs:
        out = out_dir / f"{db}_rows.csv.gz"
        print(f"=== {db} -> {out}", flush=True)
        stats = build(db, out, store, check_embeddings=not args.no_embedding_check)
        for k, v in stats.most_common():
            print(f"    {k}: {v:,}")
        print(flush=True)
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
