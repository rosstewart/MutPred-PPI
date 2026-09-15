#!/usr/bin/env python3
"""Assemble zenodo/, ready to upload.

Zenodo takes individual files, so this stages two kinds of entry:

  * single files become SYMLINKS, costing no disk. Several of them live outside
    the repository, under $MUTPRED_DATA_ROOT/<db>/, which symlinks handle.
  * directories become .tar.gz archives built in place, because a directory
    cannot be uploaded.

Alongside them it writes MANIFEST.sha256 and a deposit-facing README.txt, both
generated from the same specification below, so the listing cannot drift from
the contents.

Licence-restricted material (COSMIC, HGMD) is excluded by construction: it is
not in the specification. Unpublished VarChAMP material is excluded by an
explicit filter, `_redistributable`, because parts of the specification are
globs and a glob excludes nothing -- `TRAINING_EVAL_DIR.glob("*_rows.csv.gz")`
swept in the VarChAMP measurement tables, and `cv_reference/` is a whole
directory. Structures are
the in-house half of the canonical tree, selected on the manifest's `provenance`
column; the full manifest is deposited so the ProtVar half is identifiable.

    python src/build_zenodo_deposit.py --dry-run
    python src/build_zenodo_deposit.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from paths import (ANNOTATIONS_DIR, DATA_ROOT, DATASETS_DIR, REPO_ROOT,
                   RESULTS_DIR, TRAINING_EVAL_DIR, WEIGHTS_DIR,
                   contact_graph_store)

OUT = REPO_ROOT / "zenodo"

#: Repositories whose predictions may be redistributed. COSMIC and HGMD are
#: licence-restricted and deliberately absent.
OPEN_DBS = ("clinvar", "gnomad", "neurodev", "asd")

#: Substring marking a path as derived from the unpublished VarChAMP/IGVF
#: measurements. Every canonical name that carries them has it: the
#: `varchamp_all` and `sahni_fragoza_varchamp_all` row/split tables, their
#: cv_reference fold arrays, and the GCV result pickles, which store per-fold
#: `labels` arrays and so carry the measurements themselves.
_UNPUBLISHED = "varchamp"


def _redistributable(path: Path) -> bool:
    """False for anything derived from the unpublished VarChAMP measurements.

    Model checkpoints are exempt: depositing a model TRAINED on VarChAMP is
    intended and documented, and weights are not the measurements.
    """
    if WEIGHTS_DIR in path.parents or path.parent == WEIGHTS_DIR:
        return True
    return _UNPUBLISHED not in path.name.lower()


def _filtered(paths):
    """Drop unpublished entries, and prune them out of directory entries.

    A directory is kept as a directory only when every file under it is
    redistributable; otherwise its redistributable files are listed
    individually, so `cv_reference/` cannot smuggle its VarChAMP arrays in
    behind a single directory entry.
    """
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            bad = [q for q in p.rglob("*") if q.is_file() and not _redistributable(q)]
            if not bad:
                out.append(p)
                continue
            print(f"  [filter] {p.name}/: {len(bad)} unpublished file(s) excluded")
            out.extend(q for q in sorted(p.rglob("*"))
                       if q.is_file() and _redistributable(q))
        elif _redistributable(p):
            out.append(p)
        else:
            print(f"  [filter] excluded (unpublished): {p.name}")
    return out


def _prediction_tsv(db: str) -> Path:
    """One repository's interaction predictions, in either supported layout."""
    collected = RESULTS_DIR / "variant_dbs_all_data" / f"{db}_mutpred_ppi_predictions.tsv"
    return collected if collected.exists() else DATA_ROOT / db / "mutpred_ppi_predictions.tsv"


def bundles() -> list[dict]:
    """The deposit, as a handful of archives that unpack to the right places.

    Each archive is rooted at the repository, so unpacking is `tar xzf X -C .`
    with no path juggling and no chance of landing a file in the wrong tier.
    Keeping the count small matters: a Zenodo record with two dozen loose
    entries gives a reader no idea which ones they need.
    """
    vdb = RESULTS_DIR / "variant_dbs_all_data"
    stab = RESULTS_DIR / "variant_dbs_stability"

    datasets = (
        sorted(TRAINING_EVAL_DIR.glob("*_rows.csv.gz"))
        + sorted(TRAINING_EVAL_DIR.glob("*_splits.csv.gz"))
        + [TRAINING_EVAL_DIR / "sequences.csv.gz", TRAINING_EVAL_DIR / "aliases.csv",
           DATASETS_DIR / "cv_reference",
           ANNOTATIONS_DIR,
           DATASETS_DIR / "megascale_rows.csv.gz",
           DATASETS_DIR / "mega_splits.pkl",
           DATASETS_DIR / "mutpred2_inputs",
           DATASETS_DIR / "af3_structures_canonical" / "manifest.csv",
           contact_graph_store()]
        + [DATASETS_DIR / "variant_dbs" / f"{db}_rows.csv.gz" for db in OPEN_DBS]
        # The WT+variant sequences. Iterating a row table needs them -- the
        # tables carry accessions and a pair key, not sequence -- so without
        # these no variant-repository inference can run from the deposit at all.
        # 63 MB gzipped for the four open repositories; COSMIC and HGMD are
        # licence-restricted and excluded with the rest of their material.
        + [DATASETS_DIR / "variant_dbs" / f"{db}_interaction_loss_wt_and_vt.fasta.gz"
           for db in OPEN_DBS]
        + [DATASETS_DIR / "variant_dbs" / "aliases.csv"]
    )

    # Prediction TSVs are staged into results/variant_dbs_all_data/ under the
    # collected name, which is the layout every consumer looks for first, so a
    # user never has to set MUTPRED_DATA_ROOT just to read them.
    results = (
        sorted((RESULTS_DIR / "gcv").glob("*_detailed_results.pkl"))
        + sorted((RESULTS_DIR / "gcv").glob("*_aucs.npy"))
        + [vdb / db for db in OPEN_DBS]
        + [RESULTS_DIR / "master_variant_db_predictions_unrestricted.csv.gz",
           RESULTS_DIR / "robustness", RESULTS_DIR / "protein_class",
           RESULTS_DIR / "biclass_gcv", RESULTS_DIR / "stability_interaction"]
        + [stab / f"{db}_stability_predictions.tsv" for db in OPEN_DBS]
    )
    renamed = {_prediction_tsv(db): vdb / f"{db}_mutpred_ppi_predictions.tsv"
               for db in OPEN_DBS}
    results += list(renamed)

    datasets = _filtered(datasets)
    results = _filtered(results)
    renamed = {k: v for k, v in renamed.items() if _redistributable(Path(k))}

    return [
        {"name": "mutpred-ppi-datasets.tar.gz", "srcs": datasets, "rename": {},
         "why": "row tables, fold splits, annotations and the contact-graph store",
         "unpack": "tar xzf mutpred-ppi-datasets.tar.gz -C ."},
        {"name": "mutpred-ppi-results.tar.gz", "srcs": results, "rename": renamed,
         "why": "cross-validation results and variant-repository predictions",
         "unpack": "tar xzf mutpred-ppi-results.tar.gz -C ."},
        {"name": "mutpred-ppi-weights.tar.gz", "srcs": [WEIGHTS_DIR], "rename": {},
         "why": "every model checkpoint and scaler",
         "unpack": "tar xzf mutpred-ppi-weights.tar.gz -C ."},
        {"name": "mutpred-ppi-structures.tar.gz", "srcs": None, "rename": {},
         "why": "the AlphaFold 3 structures folded for this study (optional)",
         "unpack": "tar xzf mutpred-ppi-structures.tar.gz -C ."},
    ]


def in_house_structures() -> tuple[Path, list[str]]:
    """(structure dir, filenames) for the in-house half of the canonical tree."""
    d = DATASETS_DIR / "af3_structures_canonical"
    manifest = d / "manifest.csv"
    if not manifest.exists():
        return d, []
    with open(manifest, newline="") as fh:
        reader = csv.DictReader(fh)
        if "provenance" not in (reader.fieldnames or []):
            raise SystemExit(
                f"{manifest} has no `provenance` column, so the in-house structures "
                f"cannot be separated from the ProtVar ones. Re-run "
                f"canonicalize_structures.py, which now records it.")
        return d, [r["filename"] for r in reader if r["provenance"] == "af3_in_house"]


def _arcname(src: Path, rename: dict) -> str:
    """Path inside the archive, relative to the repository root."""
    target = rename.get(src, src)
    try:
        return str(target.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        # Sources outside the repo (the per-database prediction TSVs) are given
        # an explicit in-repo destination by `rename`; anything else is a bug.
        raise SystemExit(f"{src} is outside the repository and has no mapped "
                         f"destination; add one to `rename`.")


def _size(p: Path) -> int:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size if p.exists() else 0


def _fmt(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be deposited, with sizes, and build nothing")
    ap.add_argument("--force", action="store_true", help="overwrite an existing zenodo/")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    specs = bundles()
    struct_dir, struct_files = in_house_structures()

    missing, total = [], 0
    print(f"{'archive':<34} {'size':>10}  contents")
    for b in specs:
        if b["srcs"] is None:
            n = sum(_size(struct_dir / f) for f in struct_files)
            print(f"  {b['name']:<32} {_fmt(n):>10}  "
                  f"{len(struct_files):,} in-house structures")
        else:
            present = [x for x in b["srcs"] if x.exists()]
            missing += [f"{b['name']}  <- {x}" for x in b["srcs"] if not x.exists()]
            n = sum(_size(x) for x in present)
            print(f"  {b['name']:<32} {_fmt(n):>10}  {len(present)} path(s)")
        total += n
    print(f"\n  {'TOTAL (uncompressed)':<32} {_fmt(total):>10}")

    if missing:
        print(f"\n{len(missing)} source(s) absent:")
        for m in missing:
            print(f"  {m}")
        print("\nLicence-restricted and unpublished material is excluded by design; "
              "anything else absent means a step has not been run.")

    if args.dry_run:
        return 0

    if out.exists():
        if not args.force:
            print(f"\n{out} exists. Pass --force to rebuild it.", file=sys.stderr)
            return 1
        shutil.rmtree(out)
    out.mkdir(parents=True)

    written = []
    for b in specs:
        dest = out / b["name"]
        if b["srcs"] is None:
            if not struct_files:
                continue
            with tarfile.open(dest, "w:gz") as tar:
                for i, f in enumerate(struct_files, 1):
                    src = struct_dir / f
                    if src.exists():
                        tar.add(src, arcname=f"datasets/af3_structures_canonical/{f}")
                    if i % 5000 == 0:
                        print(f"        {i:,}/{len(struct_files):,}", flush=True)
        else:
            present = [x for x in b["srcs"] if x.exists()]
            if not present:
                continue
            with tarfile.open(dest, "w:gz") as tar:
                for src in present:
                    tar.add(src, arcname=_arcname(src, b["rename"]))
        written.append(dest)
        print(f"[tar] {b['name']}  ({_fmt(dest.stat().st_size)})")

    print("\nhashing...")
    with open(out / "MANIFEST.sha256", "w") as fh:
        for path in sorted(written):
            fh.write(f"{_sha256(path)}  {path.name}\n")
    with open(out / "README.txt", "w") as fh:
        fh.write(_readme(specs, struct_files))

    print(f"\n{out} ready: {len(written)} archives + MANIFEST.sha256 + README.txt")
    print("Verify with:  cd zenodo && sha256sum -c --ignore-missing MANIFEST.sha256")
    return 0


def _readme(specs, struct_files) -> str:
    lines = [
        "MutPred-PPI data deposit",
        "========================",
        "",
        "Code:  https://github.com/rosstewart/mutpred-ppi",
        "Paper: https://doi.org/10.64898/2025.12.20.695738",
        "",
        "Install",
        "-------",
        "Every archive is rooted at the repository, so unpack them from the",
        "repository root and each file lands where the code expects it.",
        "",
        "  git clone https://github.com/rosstewart/mutpred-ppi.git",
        "  cd mutpred-ppi",
        "",
    ]
    for b in specs:
        lines.append(f"  {b['unpack']}")
    lines += [
        "",
        "  sha256sum -c --ignore-missing MANIFEST.sha256",
        "",
        "Then:  python -c \"from paths import describe; describe()\"",
        "",
        "Contents",
        "--------",
    ]
    for b in specs:
        lines.append(f"  {b['name']}")
        lines.append(f"      {b['why']}")
    lines += [
        "",
        "The first three archives are needed to reproduce the figures. The",
        "structures archive is optional: it is only needed to rebuild the",
        "contact-graph store, which is already inside the datasets archive.",
        "",
        "Not included",
        "------------",
        "  COSMIC and HGMD derived data require a licence. Regenerate with",
        "  src/data_processing/variant_databases/map_cosmic.py and map_hgmd.py.",
        "",
        "  VarChAMP measurements were unpublished at the time of writing and",
        "  will be cross-linked from data.igvf.org on release.",
        "",
        f"  {76023:,} of the {100739:,} structures in manifest.csv come from ProtVar",
        "  (EMBL-EBI) and are not ours to redistribute; the `provenance` column",
        f"  identifies them. The {len(struct_files):,} in-house ones are included.",
        "",
        "  Protein language model caches (about 1.2 TB) are regenerated on",
        "  demand; see docs/DATA.md.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
