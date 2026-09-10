#!/usr/bin/env python
"""Build the pLDDT cache from OUR OWN AF3 complex structures.

Output format (pickle)::

    {"A6NLX3__P35557": 74.63, ...}   # dict[str, float]

KEY = THE PAIR, NOT THE ACCESSION.  The single consumer,
src/analysis/plddt_stratification.py, needs exactly one scalar per complex: it
bins the complex-level mean pLDDT at 70/85 and never indexes a per-residue or
per-chain value.  The AF3 manifests are already one row per predicted complex
with a `mean_plddt` column, so a pair key is a 1:1 re-keying of the source with
no aggregation invented on our side.  Keying by accession would instead force
the consumer to average two monomers and would silently drop the interface
context that the complex model is the whole point of.

Two further consequences of the pair key, both wanted:

  * It is isoform-aware.  `O14787-2__Q13207` is its own structure, distinct from
    `O14787__Q13207`.  The old accession-keyed AlphaFold DB cache had to fall
    back to the parent accession's monomer for 89 isoform-bearing pairs.
  * The key round-trips to a file on disk: key + ".cif.gz" is the `filename`
    column of the manifest it came from.

ORIENTATION.  The key preserves the manifest's own chain order
(`chain_a_accession__chain_b_accession`) so that round-trip holds.  AF3 chain
order is unrelated to the (interactor, partner) order a caller will have, so
lookups must try both orientations -- use `lookup()` below rather than a bare
`cache[key]`.

DELIMITER.  `__` per the repo-wide convention: `_` breaks on RefSeq ids and a
single `-` breaks on UniProt isoform suffixes.

The AlphaFold DB monomer reading path and the `cbeta` coordinate arrays that
used to live in this file are both gone.  `cbeta` was dead -- nothing outside
this builder ever read it.

Provenance of `mean_plddt`: the manifest builders take it from the mmCIF
`_atom_site.B_iso_or_equiv` field located BY NAME in the loop header.  mmCIF
column order varies between AF3 output variants, so fixed PDB column offsets
must never be used on these files; doing so once returned -1.0 for 983 CIFs.
This builder therefore copies the manifest value rather than re-parsing, and
rejects any row whose pLDDT is outside (0, 100].

Usage:
  python src/analysis/build_plddt_cache.py --output /tmp/plddt_pair_cache.pkl
  python src/analysis/build_plddt_cache.py --output /tmp/plddt_pair_cache.pkl \
      --compare-legacy datasets/annotations/plddt_cache.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from paths import ANNOTATIONS_DIR, DATASETS_DIR

PAIR_SEP = "__"

# Precedence order: the benchmark complexes first, the variant-database
# complexes second.  The two manifests overlap on a few dozen pairs (some in
# opposite chain orders) and the benchmark models are the ones the published
# analysis is about, so they win any collision.
DEFAULT_MANIFESTS = [
    DATASETS_DIR / "af3_structures_canonical" / "manifest.csv",
    DATASETS_DIR / "af3_structures_variant_dbs_canonical" / "manifest.csv",
]
DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "plddt_pair_cache.pkl"

# pLDDT is a percentage; the -1.0 sentinel from the historical mmCIF column-offset
# bug and any other out-of-range value must not reach the 70/85 bins.
PLDDT_MIN, PLDDT_MAX = 0.0, 100.0


def pair_key(a: str, b: str) -> str:
    """Cache key for a protein pair, in the given chain order."""
    return f"{a}{PAIR_SEP}{b}"


def lookup(cache: dict, a: str, b: str):
    """Mean pLDDT for the pair (a, b) in either chain orientation, else None."""
    val = cache.get(pair_key(a, b))
    if val is None:
        val = cache.get(pair_key(b, a))
    return val


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["chain_a_accession", "chain_b_accession", "mean_plddt"])
    df["mean_plddt"] = pd.to_numeric(df["mean_plddt"], errors="coerce")
    return df


def build(manifests: list[Path]) -> tuple[dict, dict]:
    """Merge manifests into a pair-keyed cache.  Earlier manifests win collisions."""
    cache: dict[str, float] = {}
    stats = {"rows": 0, "bad_plddt": 0, "collisions_same_order": 0,
             "collisions_flipped_order": 0, "max_collision_spread": 0.0}

    for path in manifests:
        df = load_manifest(path)
        kept = 0
        for a, b, val in zip(df["chain_a_accession"], df["chain_b_accession"], df["mean_plddt"]):
            stats["rows"] += 1
            if pd.isna(a) or pd.isna(b) or pd.isna(val) or not (PLDDT_MIN < val <= PLDDT_MAX):
                stats["bad_plddt"] += 1
                continue
            val = float(val)
            existing = lookup(cache, str(a), str(b))
            if existing is not None:
                key = "collisions_same_order" if pair_key(str(a), str(b)) in cache \
                    else "collisions_flipped_order"
                stats[key] += 1
                stats["max_collision_spread"] = max(
                    stats["max_collision_spread"], abs(existing - val))
                continue  # first manifest wins
            cache[pair_key(str(a), str(b))] = val
            kept += 1
        print(f"  {path}: {len(df)} rows, {kept} new pairs", flush=True)

    return cache, stats


# ── verification ───────────────────────────────────────────────────────────────

def compare_legacy(cache: dict, legacy_path: Path) -> None:
    """Measure the new cache against the accession-keyed AlphaFold DB cache.

    Reports, over exactly the complex_ids the stratification analysis iterates:
    coverage each way, how many low/medium/high bin assignments change, and the
    correlation of the per-complex mean where both resolve.  The bin-change
    count is the number that moves the figure.
    """
    # Imported lazily: plddt_stratification imports this module at top level.
    from plddt_stratification import bin_plddt, complex_id_pairs, complex_mean_plddt
    from paths import cv_reference_dir

    with open(legacy_path, "rb") as f:
        legacy = pickle.load(f)

    import pandas as pd
    rows_path = cv_reference_dir() / "sahni_fragoza_train_rows.csv.gz"
    canonical_rows = pd.read_csv(rows_path)
    complex_ids = sorted({f"{r['interactor']}-{r['partner']}"
                          for _, r in canonical_rows.iterrows()})
    pairs = complex_id_pairs()

    def legacy_mean(cid: str):
        prots = pairs.get(cid)
        if prots is None:
            parts = cid.split("-")
            if len(parts) < 2:
                return None
            prots = (parts[0], "-".join(parts[1:]))
        arrays = []
        for prot in prots:
            entry = legacy.get(prot)
            if entry is None:
                entry = legacy.get(prot.split("-")[0])
            if entry is None:
                return None
            arrays.append(entry["plddt"] if isinstance(entry, dict) else entry)
        return float(np.mean(np.concatenate(arrays)))

    old_vals, new_vals = {}, {}
    for cid in complex_ids:
        o = legacy_mean(cid)
        if o is not None:
            old_vals[cid] = o
        n = complex_mean_plddt(cid, cache, pairs)
        if n is not None:
            new_vals[cid] = n

    both = sorted(set(old_vals) & set(new_vals))
    changed = [c for c in both if bin_plddt(old_vals[c]) != bin_plddt(new_vals[c])]

    print(f"\nComparison against legacy {legacy_path}")
    print(f"  complex_ids in the analysis      : {len(complex_ids)}")
    print(f"  covered by legacy (AFDB monomer) : {len(old_vals)}")
    print(f"  covered by new (AF3 complex)     : {len(new_vals)}")
    print(f"  gained (new only)                : {len(set(new_vals) - set(old_vals))}")
    print(f"  lost (legacy only)               : {len(set(old_vals) - set(new_vals))}")
    print(f"  resolved by both                 : {len(both)}")
    print(f"  BIN ASSIGNMENTS CHANGED          : {len(changed)} "
          f"({100.0 * len(changed) / max(len(both), 1):.1f}% of shared)")

    transitions: dict[tuple[str, str], int] = {}
    for c in changed:
        t = (bin_plddt(old_vals[c]), bin_plddt(new_vals[c]))
        transitions[t] = transitions.get(t, 0) + 1
    for (o, n), k in sorted(transitions.items(), key=lambda kv: -kv[1]):
        print(f"    {o:>6} -> {n:<6} {k}")

    for label, vals in (("legacy", old_vals), ("new", new_vals)):
        counts = {b: sum(1 for c in both if bin_plddt(vals[c]) == b)
                  for b in ("low", "medium", "high")}
        print(f"  {label} bin counts over shared: {counts}")

    if len(both) > 1:
        a = np.array([old_vals[c] for c in both])
        b = np.array([new_vals[c] for c in both])
        print(f"  per-complex mean pearson r : {np.corrcoef(a, b)[0, 1]:.4f}")
        print(f"  mean shift (new - legacy)  : {float(np.mean(b - a)):+.2f} "
              f"(legacy {a.mean():.2f}, new {b.mean():.2f})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, action="append", default=None,
                   help="AF3 manifest.csv (repeatable; earlier wins collisions). "
                        f"Default: {[str(m) for m in DEFAULT_MANIFESTS]}")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"output pickle (default: {DEFAULT_OUTPUT})")
    p.add_argument("--compare-legacy", type=Path, default=None,
                   help="accession-keyed AlphaFold DB cache to measure the rebuild against")
    p.add_argument("--force", action="store_true",
                   help="allow overwriting a file under datasets/annotations")
    args = p.parse_args()

    out = args.output.resolve()
    if not args.force and ANNOTATIONS_DIR.resolve() in out.parents and out.exists():
        p.error(f"refusing to overwrite {out}; write to scratch and diff first, or pass --force")

    manifests = args.manifest or DEFAULT_MANIFESTS
    missing = [m for m in manifests if not Path(m).exists()]
    if missing:
        p.error(f"manifest not found: {missing} (pass --manifest)")

    print(f"Building from {len(manifests)} manifest(s)")
    cache, stats = build([Path(m) for m in manifests])
    print(f"Built {len(cache)} pairs from {stats['rows']} manifest rows")
    print(f"  rejected pLDDT (missing or outside {PLDDT_MIN}-{PLDDT_MAX}): {stats['bad_plddt']}")
    print(f"  duplicate pairs dropped: {stats['collisions_same_order']} same chain order, "
          f"{stats['collisions_flipped_order']} flipped; "
          f"max disagreement {stats['max_collision_spread']:.2f} pLDDT")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    print(f"Saved -> {out}")

    if args.compare_legacy and Path(args.compare_legacy).exists():
        compare_legacy(cache, Path(args.compare_legacy))
    return 0


if __name__ == "__main__":
    sys.exit(main())
