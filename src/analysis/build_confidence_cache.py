#!/usr/bin/env python
"""Rebuild datasets/annotations/confidence_scores.pkl from AF3 summary confidences.

Consumer: src/analysis/roc_plots.py, which reads

    {complex_key: {'iptm': float, 'ptm': float}}

where complex_key is the AF3 job name (lowercase, `_`-joined protein IDs, e.g.
'a0a024qyx0_o43889', 'np_000014_sgta', 'o00308-1_b7z4b8').  roc_plots.py splits
that key with ids.split_wt_id(), so the key must keep the job-name spelling --
it is remapped to UniProt downstream, not here.

Source: the AF3 `*_summary_confidences.json` files for the three-dataset
training run.  Each holds top-level `iptm` and `ptm` (AF3 writes them rounded to
2 decimals; the cache stores them verbatim).

Usage:
  python src/analysis/build_confidence_cache.py \
      --output /tmp/confidence_scores.pkl \
      --compare-to datasets/annotations/confidence_scores.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import tempfile
from pathlib import Path

from paths import ANNOTATIONS_DIR, DATA_ROOT

DEFAULT_JSON_DIRS = [DATA_ROOT / "three_datasets_af3_models" / "old_train_confidences" / "json"]
DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "confidence_scores.pkl"

_SUMMARY_RE = re.compile(r"^(.+?)_summary_confidences\.json$")


def collect_json_files(dirs: list[Path], recursive: bool) -> list[Path]:
    pattern = "**/*_summary_confidences.json" if recursive else "*_summary_confidences.json"
    files: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"confidence directory not found: {d}")
        found = sorted(d.glob(pattern))
        print(f"  {d}: {len(found)} summary_confidences JSON files")
        files.extend(found)
    return files


def build(files: list[Path]) -> dict:
    cache: dict[str, dict] = {}
    collisions = 0
    unreadable = 0
    for path in files:
        m = _SUMMARY_RE.match(path.name)
        if m is None:
            continue
        key = m.group(1)
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: unreadable {path}: {exc}")
            unreadable += 1
            continue
        entry = {
            "iptm": float(data["iptm"]) if data.get("iptm") is not None else None,
            "ptm": float(data["ptm"]) if data.get("ptm") is not None else None,
        }
        # Earlier directories win, so --json-dir order is the precedence order.
        if key in cache:
            if cache[key] != entry:
                collisions += 1
            continue
        cache[key] = entry
    if collisions:
        print(f"  {collisions} keys seen in more than one directory with differing values (first kept)")
    if unreadable:
        print(f"  {unreadable} files could not be parsed")
    return cache


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

    for field in ("iptm", "ptm"):
        diffs = [
            abs(built[k][field] - ref[k][field])
            for k in shared
            if built[k][field] is not None and ref[k][field] is not None
        ]
        n_exact = sum(1 for d in diffs if d == 0.0)
        max_diff = max(diffs) if diffs else 0.0
        print(f"  {field}: {n_exact}/{len(diffs)} exactly equal, max abs diff {max_diff:.6g}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json-dir", type=Path, action="append", default=None,
                   help="directory of *_summary_confidences.json (repeatable; first wins on key clash)")
    p.add_argument("--recursive", action="store_true", help="search --json-dir trees recursively")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"output pickle (default: {DEFAULT_OUTPUT})")
    p.add_argument("--compare-to", type=Path, default=ANNOTATIONS_DIR / "confidence_scores.pkl",
                   help="existing cache to diff against ('none' to skip)")
    p.add_argument("--force", action="store_true", help="allow overwriting a file under datasets/annotations")
    args = p.parse_args()

    out = args.output.resolve()
    if not args.force and ANNOTATIONS_DIR.resolve() in out.parents and out.exists():
        p.error(f"refusing to overwrite {out}; write to scratch and diff first, or pass --force")

    dirs = args.json_dir or DEFAULT_JSON_DIRS
    print("Collecting AF3 summary confidences...")
    files = collect_json_files([Path(d) for d in dirs], args.recursive)
    cache = build(files)
    print(f"Built {len(cache)} entries from {len(files)} files")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    print(f"Saved -> {out}")

    if str(args.compare_to).lower() != "none" and Path(args.compare_to).exists():
        compare(cache, Path(args.compare_to))
    return 0


if __name__ == "__main__":
    sys.exit(main())
