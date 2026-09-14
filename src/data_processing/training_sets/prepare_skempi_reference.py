#!/usr/bin/env python
"""Derive the SKEMPI training-protein set that defines C1/C2/C3 for the
SKEMPI-pretrained comparison methods.

SAAMBE-3D, MutPPI and MutPPI+ are not retrained per dataset -- they ship
pretrained on SKEMPI 2.0. Their test-class stratification therefore depends on
overlap with *SKEMPI's* proteins, not with ours, which is why this set exists
separately from the per-dataset training pairs (see
`utils.gcv_common.skempi_test_class`).

Inputs (both under `datasets/source_data/`, see docs/DATA_PREPARATION.md):

    skempi_v2.csv            SKEMPI 2.0, semicolon-delimited. The `#Pdb` column
                             is `PDB_<chains1>_<chains2>`, e.g. `1CSE_E_I`.
    pdb_chain_uniprot.csv    SIFTS PDB-chain -> UniProt, columns
                             `PDB,CHAIN,SP_PRIMARY,...`.

EVERY CHARACTER IN A CHAIN GROUP IS A CHAIN
-------------------------------------------
122 of SKEMPI's 348 complexes have multi-chain groups -- `3SE8_HL_G`,
`1BD2_ABC_DE`, `4FTV_ABC_DE` -- overwhelmingly antibodies, where `H` and `L`
are the heavy and light chains of one partner. They must be expanded character
by character.

The previous reference (`results/gcv/SAAMBE_train_uniprots.npy`, 258 accessions,
2026-08-10) did not do this: it yielded exactly the set you get by keeping only
*single-character* groups, i.e. it silently dropped every multi-chain complex.
That under-counted SKEMPI's proteins by 84 and made SAAMBE-3D/MutPPI/MutPPI+
look less overlapped with our test pairs than they are -- C1/C2/C3 too
optimistic. It also contained one accession (`Q7A260`) that SKEMPI does not
reference at all. Both are corrected here; see `docs/DATA_PREPARATION.md`.

A chain may map to more than one accession (chimeric constructs), so the
resolution is a set union rather than a last-wins lookup.

Usage:
    conda run -n ppi python src/data_processing/training_sets/prepare_skempi_reference.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

from paths import ANNOTATIONS_DIR, SOURCE_DATA_DIR

SKEMPI_CSV = SOURCE_DATA_DIR / "skempi_v2.csv"
SIFTS_CSV = SOURCE_DATA_DIR / "pdb_chain_uniprot.csv"
OUT_CSV = ANNOTATIONS_DIR / "skempi_train_uniprots.csv"


def load_sifts(path: Path = SIFTS_CSV) -> dict[tuple[str, str], set[str]]:
    """`(pdb_lower, chain) -> {accessions}` from the SIFTS per-chain table.

    The file's first line is a `# <date> | PDB: ... | UniProt: ...` provenance
    comment, ahead of the real header.
    """
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.\n"
                 f"  Download the SIFTS 'pdb_chain_uniprot' CSV (per-chain form,\n"
                 f"  columns PDB,CHAIN,SP_PRIMARY,...) from\n"
                 f"  https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html\n"
                 f"  NOTE the similarly named 'uniprot_pdb' file is a\n"
                 f"  UniProt->PDB-list mapping with NO chain column; it cannot\n"
                 f"  resolve SKEMPI's per-chain identifiers.")
    out: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    with open(path) as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        for row in csv.DictReader(f):
            out[(row["PDB"].lower(), row["CHAIN"])].add(row["SP_PRIMARY"])
    if not out:
        sys.exit(f"ERROR: {path} parsed to zero chain mappings -- wrong file?")
    return out


def skempi_complexes(path: Path = SKEMPI_CSV) -> set[str]:
    """The unique `#Pdb` tags in SKEMPI."""
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. SKEMPI 2.0 is downloadable from "
                 f"https://life.bsc.es/pid/skempi2/ -- see docs/DATA_PREPARATION.md.")
    with open(path) as f:
        return {r["#Pdb"] for r in csv.DictReader(f, delimiter=";") if r.get("#Pdb")}


def derive(skempi_csv: Path = SKEMPI_CSV, sifts_csv: Path = SIFTS_CSV):
    """`(accessions, unresolved_chains, n_complexes)`."""
    sifts = load_sifts(sifts_csv)
    tags = skempi_complexes(skempi_csv)

    accessions: set[str] = set()
    unresolved: set[tuple[str, str]] = set()
    for tag in tags:
        parts = tag.split("_")
        pdb, groups = parts[0].lower(), parts[1:]
        for group in groups:
            for chain in group:            # every character is a chain
                hit = sifts.get((pdb, chain))
                if hit:
                    accessions |= hit
                else:
                    unresolved.add((pdb, chain))
    return accessions, unresolved, len(tags)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skempi", type=Path, default=SKEMPI_CSV)
    ap.add_argument("--sifts", type=Path, default=SIFTS_CSV)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    accessions, unresolved, n_complexes = derive(args.skempi, args.sifts)
    print(f"SKEMPI complexes: {n_complexes}")
    print(f"resolved UniProt accessions: {len(accessions)}")
    print(f"unresolved (pdb, chain) pairs: {len(unresolved)}"
          + (f"  e.g. {sorted(unresolved)[:5]}" if unresolved else ""))

    if args.dry_run:
        print("\n[dry-run] not writing")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        f.write(f"# SKEMPI 2.0 training proteins, for the C1/C2/C3 stratification of\n"
                f"# the SKEMPI-pretrained methods (SAAMBE-3D, MutPPI, MutPPI+).\n"
                f"# Derived by src/data_processing/training_sets/prepare_skempi_reference.py\n"
                f"# from {args.skempi.name} ({n_complexes} complexes) joined per chain\n"
                f"# against {args.sifts.name}. {len(unresolved)} (pdb, chain) pairs\n"
                f"# had no SIFTS mapping and are excluded.\n")
        w = csv.writer(f)
        w.writerow(["uniprot"])
        w.writerows([a] for a in sorted(accessions))
    print(f"\nwrote {args.out} ({len(accessions)} accessions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
