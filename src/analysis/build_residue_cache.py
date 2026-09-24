#!/usr/bin/env python
"""Per-residue solvent accessibility and pLDDT for the benchmark complexes.

Supplies the two quantities Supplementary Fig S5A needs and the repository did
not previously compute anywhere. The existing interface rule
(`interface_analysis.is_interface`) is binary -- a residue either makes a
cross-chain contact or it does not -- so its "non-interface" comparator holds
the buried core together with the free surface, and a variant in the core is
not the control a reader has in mind when comparing against "non-interface".

Two SASA passes per complex give both quantities:

  * ``rsa_unbound`` = SASA of the residue in the ISOLATED chain, divided by the
    Tien et al. 2013 theoretical maximum for that residue type. This separates
    surface from buried, and it is a property of the protein rather than of the
    complex, so a residue does not become "buried" merely by being covered by
    its partner.
  * ``dsasa`` = SASA(isolated chain) - SASA(complex). A residue that loses
    accessible surface on binding is the structural definition of an interface
    residue, and is more defensible than "has a contact edge" within 4.5 A.

``plddt`` comes from the mmCIF ``_atom_site.B_iso_or_equiv`` field, averaged
over the residue's atoms. It is read through gemmi, which locates the field BY
NAME in the loop header; mmCIF column order varies between AF3 output variants,
so fixed PDB column offsets must never be used on these files -- doing so once
returned -1.0 for 983 CIFs (see build_plddt_cache.py).

`plddt` is the mean over the residue's OWN atoms, so the per-complex mean of
this cache is residue-weighted. The manifest's `mean_plddt`, which
`build_plddt_cache.py` copies, is atom-weighted, so the two differ by up to
about 4 pLDDT on a single complex (mean 0.45) without either being wrong. A
per-residue cache has to weight per residue; do not "fix" the gap by switching
either one.

RSA HAS NO MEANING WITHOUT ITS REFERENCE. Absolute SASA is what every engine
returns; the per-residue-type maximum is the choice that decides where the
surface/buried line falls. Tien et al. 2013 theoretical maxima are used here
and must be stated in Methods alongside any threshold.

Output format (pickle)::

    {"P35557__A6NLX3": {
        "P35557": {"seq": "MLD...", "rsa": float32[n], "dsasa": float32[n],
                   "plddt": float32[n]},
        "A6NLX3": {...}},
     ...}

KEY = THE PAIR, in the manifest's own chain order, `__`-delimited, matching
`build_plddt_cache.py`. AF3 chain order is unrelated to the (interactor,
partner) order a caller will have, so use `lookup()` rather than a bare
`cache[key]`. Within a pair the chains are keyed by accession, and each carries
the sequence read from the structure so a caller can confirm it is reading the
chain it meant to.

Usage:
  python src/analysis/build_residue_cache.py --output /tmp/residue_cache.pkl
  python src/analysis/build_residue_cache.py --output /tmp/residue_cache.pkl \
      --workers 16
"""
from __future__ import annotations

import argparse
import copy
import gzip
import pickle
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from paths import ANNOTATIONS_DIR, DATASETS_DIR, cv_reference_dir

PAIR_SEP = "__"

#: Theoretical maximum accessible surface area per residue type, A^2.
#: Tien MZ, Meyer AG, Sydykova DK, Spielman SJ, Wilke CO. Maximum allowed
#: solvent accessibilites of residues in proteins. PLoS ONE 8(11): e80635
#: (2013), Table 1, "Theoretical" column. Chosen over the Miller and Rose
#: empirical sets because it is the current standard and is a closed-form
#: reference rather than a sample statistic.
MAX_ASA_TIEN = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLU": 223.0, "GLN": 225.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

DEFAULT_MANIFEST = DATASETS_DIR / "af3_structures_canonical" / "manifest.csv"
DEFAULT_ROWS = cv_reference_dir() / "sahni_fragoza_train_rows.csv.gz"
DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "residue_cache.pkl"


def pair_key(a: str, b: str) -> str:
    return f"{a}{PAIR_SEP}{b}"


def lookup(cache: dict, a: str, b: str):
    """The pair's per-chain record in either chain orientation, else None."""
    return cache.get(pair_key(a, b)) or cache.get(pair_key(b, a))


def _residue_arrays(chain) -> tuple[str, np.ndarray, np.ndarray]:
    """(one-letter sequence, per-residue SASA, per-residue mean pLDDT).

    Non-standard residues keep their position -- dropping them would shift every
    downstream index against the 1-based mutation position -- and are given a
    NaN RSA, which excludes them from any threshold without excluding their
    neighbours.
    """
    seq, sasa, plddt = [], [], []
    for res in chain:
        name = res.get_resname()
        seq.append(THREE_TO_ONE.get(name, "X"))
        sasa.append(float(res.sasa))
        bfacs = [a.get_bfactor() for a in res]
        plddt.append(float(np.mean(bfacs)) if bfacs else np.nan)
    return "".join(seq), np.array(sasa, dtype=np.float32), np.array(plddt, dtype=np.float32)


def process_structure(path: Path) -> dict | None:
    """Per-chain RSA, dSASA and pLDDT arrays for one complex, or None on failure.

    Runs in a worker process, so it imports Biopython locally and returns plain
    arrays rather than Biopython objects.
    """
    from Bio.PDB import MMCIFParser
    from Bio.PDB.SASA import ShrakeRupley

    warnings.filterwarnings("ignore")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        model = MMCIFParser(QUIET=True).get_structure("x", fh)[0]

    sr = ShrakeRupley()
    # Bound state first: `compute` writes `.sasa` onto the residues in place, so
    # the unbound pass has to run on copies or it would overwrite these.
    sr.compute(model, level="R")
    bound = {ch.id: np.array([float(r.sasa) for r in ch], dtype=np.float32)
             for ch in model}

    out = {}
    for ch in model:
        isolated = copy.deepcopy(ch)
        sr.compute(isolated, level="R")
        seq, unbound_sasa, plddt = _residue_arrays(isolated)
        maxasa = np.array([MAX_ASA_TIEN.get(r.get_resname(), np.nan) for r in isolated],
                          dtype=np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            rsa = unbound_sasa / maxasa
        out[ch.id] = {"seq": seq, "rsa": rsa.astype(np.float32),
                      "dsasa": (unbound_sasa - bound[ch.id]).astype(np.float32),
                      "plddt": plddt}
    return out


def _worker(job):
    """(key, chain_a_id, acc_a, chain_b_id, acc_b, path) -> (key, record|None, err)."""
    key, ch_a, acc_a, ch_b, acc_b, path = job
    try:
        per_chain = process_structure(Path(path))
    except Exception as exc:                      # noqa: BLE001 -- reported, not raised
        return key, None, f"{type(exc).__name__}: {exc}"
    record = {}
    for chain_id, acc in ((ch_a, acc_a), (ch_b, acc_b)):
        if chain_id not in per_chain:
            return key, None, f"chain {chain_id} absent from the structure"
        record[acc] = per_chain[chain_id]
    return key, record, None


def target_pairs(rows_path: Path) -> set[tuple[str, str]]:
    """The (interactor, partner) accession pairs the S5 analysis iterates."""
    df = pd.read_csv(rows_path, usecols=["interactor", "partner"])
    return {(str(i), str(p)) for i, p in zip(df["interactor"], df["partner"])}


def manifest_index(manifest: Path) -> dict[tuple[str, str], tuple]:
    """(acc_a, acc_b) in both orientations -> (key, chain_a_id, acc_a, chain_b_id, acc_b, path).

    Both orientations are indexed because the manifest's chain order is AF3's,
    which has no relation to the (interactor, partner) order of the row tables.
    First manifest row wins a collision, matching `build_plddt_cache.py`.
    """
    df = pd.read_csv(manifest, usecols=["filename", "chain_a_accession",
                                        "chain_b_accession", "chain_a_id", "chain_b_id"])
    root = manifest.parent
    index: dict[tuple[str, str], tuple] = {}
    for fn, a, b, ca, cb in df.itertuples(index=False):
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        entry = (pair_key(a, b), ca, a, cb, b, str(root / fn))
        index.setdefault((a, b), entry)
        index.setdefault((b, a), entry)
    return index


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                   help=f"AF3 structure manifest (default: {DEFAULT_MANIFEST})")
    p.add_argument("--rows", type=Path, default=DEFAULT_ROWS,
                   help=f"row table naming the pairs to cover (default: {DEFAULT_ROWS})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"output pickle (default: {DEFAULT_OUTPUT})")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel processes (default: 8)")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N complexes (smoke test)")
    p.add_argument("--force", action="store_true",
                   help="allow overwriting a file under datasets/annotations")
    args = p.parse_args()

    out = args.output.resolve()
    if not args.force and ANNOTATIONS_DIR.resolve() in out.parents and out.exists():
        p.error(f"refusing to overwrite {out}; write to scratch and diff first, or pass --force")
    for required in (args.manifest, args.rows):
        if not required.exists():
            p.error(f"not found: {required}")

    pairs = target_pairs(args.rows)
    index = manifest_index(args.manifest)
    jobs, missing = {}, 0
    for pair in sorted(pairs):
        entry = index.get(pair)
        if entry is None:
            missing += 1
            continue
        jobs.setdefault(entry[0], entry)
    jobs = sorted(jobs.values())
    if args.limit:
        jobs = jobs[:args.limit]

    print(f"Pairs in {args.rows.name}: {len(pairs)}")
    print(f"  resolved to {len(jobs)} distinct complexes; {missing} pairs absent "
          f"from the manifest")

    cache, errors = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (key, record, err) in enumerate(pool.map(_worker, jobs, chunksize=4), 1):
            if record is None:
                errors.append((key, err))
            else:
                cache[key] = record
            if n % 200 == 0:
                print(f"  {n}/{len(jobs)} complexes, {len(errors)} failed", flush=True)

    print(f"Built {len(cache)} complexes; {len(errors)} failed")
    for key, err in errors[:10]:
        print(f"  FAILED {key}: {err}")

    rsa = np.concatenate([c["rsa"] for rec in cache.values() for c in rec.values()])
    finite = rsa[np.isfinite(rsa)]
    print(f"  residues: {len(rsa):,} ({len(rsa) - len(finite):,} non-standard, RSA NaN)")
    print(f"  RSA < 0.25 (buried): {100.0 * float((finite < 0.25).mean()):.1f}%")
    print(f"  RSA median {float(np.median(finite)):.3f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
