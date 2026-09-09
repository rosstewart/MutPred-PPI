#!/usr/bin/env python
"""Rebuild datasets/annotations/plddt_cache.pkl from AlphaFold DB monomer models.

Consumer: src/analysis/plddt_stratification.py, which reads

    {ACCESSION: {"plddt": float32[L], "cbeta": float32[L, 3]}}

keyed by a bare UniProt accession (uppercase, isoform suffix stripped) and looked
up per protein of a complex ID.  Only "plddt" is used there, but "cbeta" is part
of the on-disk object and is reproduced so the rebuilt file is a drop-in.

Source: per-protein AlphaFold DB PDB models, NOT the AF3 complexes in
datasets/af3_structures*/.  Those are pair models whose chains are trimmed to the
assayed constructs, so their per-residue arrays do not line up with this cache
(106 of 1,161 accessions differ in length, 101 are absent entirely).  pLDDT is
the CA B-factor; cbeta is the CB coordinate, falling back to CA for glycine.

Only fragment F1 is read.  For proteins longer than 1,400 residues AlphaFold DB
splits the model into F1..Fn, so the cached array covers the first fragment only
-- reproduced deliberately, since that is what the published cache contains.

Usage (reproduces the published 1,161-entry cache bitwise):
  python src/analysis/build_plddt_cache.py \
      --extra-accession O43615 --extra-accession P10451 \
      --output /tmp/plddt_cache.pkl \
      --compare-to datasets/annotations/plddt_cache.pkl

The two --extra-accession values are legacy: they are in the published cache but
in neither column of the current benchmark table, so they must have entered from
an earlier revision of it.  The builder is incremental in spirit -- rerunning
against a fresh accession list will not resurrect them on its own.
"""
from __future__ import annotations

import argparse
import gzip
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

from paths import ANNOTATIONS_DIR

DEFAULT_AFDB_DIR = Path("/data/dbs/alphafold_db")
# Provenance of the published key set: the interactor column of the 2026 mutppi
# benchmark table.  Out of tree, hence overridable.
DEFAULT_INTERACTOR_CSV = Path("/home/rcstewart/ppi_lossgain/2026/mutppi/benchmark/training_data.csv")
DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "plddt_cache.pkl"
# v4 before v6: the published cache was built when v4 was the newest release, and
# the two releases give different coordinates for the same accession.
DEFAULT_VERSIONS = "4,6,3,2"


def _bare(accession: str) -> str:
    return accession.split("-")[0].strip().upper()


def accessions_from_csv(path: Path, column: str) -> set[str]:
    import pandas as pd
    df = pd.read_csv(path, usecols=[column])
    return {_bare(a) for a in df[column].dropna().astype(str)}


def accessions_from_vt_ids(path: Path) -> set[str]:
    """Protein tokens the consumer will look up, split exactly as it splits them."""
    with open(path, "rb") as f:
        vt_ids = pickle.load(f)
    out: set[str] = set()
    for vt_id in vt_ids:
        parts = str(vt_id).split(" ")[0].split("-")
        if len(parts) < 2:
            continue
        out.add(_bare(parts[0]))
        out.add(_bare("-".join(parts[1:])))
    return out


def accessions_from_cache(path: Path) -> set[str]:
    with open(path, "rb") as f:
        return set(pickle.load(f))


def find_model(afdb_dir: Path, accession: str, versions: list[int]) -> Path | None:
    for v in versions:
        p = afdb_dir / f"AF-{accession}-F1-model_v{v}.pdb.gz"
        if p.exists():
            return p
    return None


def parse_model(path: Path) -> dict | None:
    """Per-residue pLDDT and CB coordinates from a gzipped AlphaFold DB PDB."""
    plddt_map: dict[int, float] = {}
    ca_map: dict[int, np.ndarray] = {}
    cb_map: dict[int, np.ndarray] = {}

    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        with opener(path, "rt") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                atom_name = line[12:16].strip()
                if atom_name not in ("CA", "CB"):
                    continue
                try:
                    res_seq = int(line[22:26])
                    coord = np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                        dtype=np.float32,
                    )
                    b_factor = float(line[60:66])
                except ValueError:
                    continue
                if atom_name == "CA":
                    plddt_map[res_seq] = b_factor
                    ca_map[res_seq] = coord
                else:
                    cb_map[res_seq] = coord
    except OSError:
        return None

    if not plddt_map:
        return None

    res_nums = sorted(plddt_map)
    plddt = np.zeros(len(res_nums), dtype=np.float32)
    cbeta = np.zeros((len(res_nums), 3), dtype=np.float32)
    for i, rn in enumerate(res_nums):
        plddt[i] = plddt_map[rn]
        cbeta[i] = cb_map.get(rn, ca_map[rn])
    return {"plddt": plddt, "cbeta": cbeta}


def build(accessions: set[str], afdb_dir: Path, versions: list[int]) -> tuple[dict, list[str]]:
    cache: dict[str, dict] = {}
    not_found: list[str] = []
    ordered = sorted(accessions)
    for i, ac in enumerate(ordered, 1):
        model = find_model(afdb_dir, ac, versions)
        if model is None:
            not_found.append(ac)
            continue
        entry = parse_model(model)
        if entry is None:
            not_found.append(ac)
            continue
        cache[ac] = entry
        if i % 250 == 0:
            print(f"  {i}/{len(ordered)} accessions processed, {len(cache)} parsed", flush=True)
    return cache, not_found


def compare(built: dict, reference_path: Path) -> None:
    with open(reference_path, "rb") as f:
        ref = pickle.load(f)

    built_keys, ref_keys = set(built), set(ref)
    shared = built_keys & ref_keys
    print(f"\nComparison against {reference_path}")
    print(f"  reference entries : {len(ref)}")
    print(f"  built entries     : {len(built)}")
    print(f"  shared keys       : {len(shared)}")
    print(f"  missing (in reference, not built) : {len(ref_keys - built_keys)}")
    print(f"  new (built, not in reference)     : {len(built_keys - ref_keys)}")
    for label, extra in (("missing", ref_keys - built_keys), ("new", built_keys - ref_keys)):
        if extra:
            print(f"    {label} examples: {sorted(extra)[:5]}")

    len_mismatch = [k for k in shared if len(built[k]["plddt"]) != len(ref[k]["plddt"])]
    print(f"  length mismatches : {len(len_mismatch)} {sorted(len_mismatch)[:5]}")

    comparable = sorted(shared - set(len_mismatch))
    for field in ("plddt", "cbeta"):
        identical = 0
        max_diff = 0.0
        for k in comparable:
            a = np.asarray(built[k][field], dtype=np.float64)
            b = np.asarray(ref[k][field], dtype=np.float64)
            if np.array_equal(a, b):
                identical += 1
            max_diff = max(max_diff, float(np.max(np.abs(a - b))) if a.size else 0.0)
        print(f"  {field}: {identical}/{len(comparable)} arrays bitwise identical, max abs diff {max_diff:.6g}")

    if comparable:
        a = np.concatenate([np.asarray(built[k]["plddt"], dtype=np.float64) for k in comparable])
        b = np.concatenate([np.asarray(ref[k]["plddt"], dtype=np.float64) for k in comparable])
        print(f"  pLDDT pearson r over {a.size} residues: {np.corrcoef(a, b)[0, 1]:.6f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--afdb-dir", type=Path, default=DEFAULT_AFDB_DIR, help="AlphaFold DB PDB directory")
    p.add_argument("--model-versions", default=DEFAULT_VERSIONS,
                   help=f"AFDB model versions in preference order (default: {DEFAULT_VERSIONS})")
    p.add_argument("--interactor-csv", type=Path, action="append", default=None,
                   help=f"CSV of accessions (repeatable; default: {DEFAULT_INTERACTOR_CSV})")
    p.add_argument("--csv-column", default="interactor", help="column of --interactor-csv to read")
    p.add_argument("--vt-ids", type=Path, action="append", default=None,
                   help="vt_ids pickle; adds every protein token the consumer looks up")
    p.add_argument("--accession-file", type=Path, action="append", default=None,
                   help="plain text file, one accession per line")
    p.add_argument("--keys-from-cache", type=Path, default=None,
                   help="reuse the key set of an existing cache (verification/refresh only)")
    p.add_argument("--extra-accession", action="append", default=None, help="single accession (repeatable)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"output pickle (default: {DEFAULT_OUTPUT})")
    p.add_argument("--compare-to", type=Path, default=ANNOTATIONS_DIR / "plddt_cache.pkl",
                   help="existing cache to diff against ('none' to skip)")
    p.add_argument("--force", action="store_true", help="allow overwriting a file under datasets/annotations")
    args = p.parse_args()

    out = args.output.resolve()
    if not args.force and ANNOTATIONS_DIR.resolve() in out.parents and out.exists():
        p.error(f"refusing to overwrite {out}; write to scratch and diff first, or pass --force")

    versions = [int(v) for v in args.model_versions.split(",") if v.strip()]
    if not args.afdb_dir.is_dir():
        p.error(f"AlphaFold DB directory not found: {args.afdb_dir}")

    accessions: set[str] = set()
    explicit = any((args.interactor_csv, args.vt_ids, args.accession_file, args.keys_from_cache))
    for csv_path in (args.interactor_csv or ([] if explicit else [DEFAULT_INTERACTOR_CSV])):
        if not Path(csv_path).exists():
            p.error(f"accession CSV not found: {csv_path} (pass --interactor-csv/--accession-file)")
        got = accessions_from_csv(Path(csv_path), args.csv_column)
        print(f"  {csv_path} [{args.csv_column}]: {len(got)} accessions")
        accessions |= got
    for vt_path in (args.vt_ids or []):
        got = accessions_from_vt_ids(Path(vt_path))
        print(f"  {vt_path}: {len(got)} protein tokens")
        accessions |= got
    for list_path in (args.accession_file or []):
        got = {_bare(x) for x in Path(list_path).read_text().split() if x.strip()}
        print(f"  {list_path}: {len(got)} accessions")
        accessions |= got
    if args.keys_from_cache:
        got = accessions_from_cache(Path(args.keys_from_cache))
        print(f"  {args.keys_from_cache}: {len(got)} keys")
        accessions |= got
    accessions |= {_bare(a) for a in (args.extra_accession or [])}

    if not accessions:
        p.error("no accessions requested")
    print(f"Requested accessions: {len(accessions)}")

    cache, not_found = build(accessions, args.afdb_dir, versions)
    print(f"Built {len(cache)} entries; {len(not_found)} accessions had no AlphaFold DB F1 model")
    if not_found:
        print(f"  examples: {not_found[:5]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    print(f"Saved -> {out}")

    if str(args.compare_to).lower() != "none" and Path(args.compare_to).exists():
        compare(cache, Path(args.compare_to))
    return 0


if __name__ == "__main__":
    sys.exit(main())
