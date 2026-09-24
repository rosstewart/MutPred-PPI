#!/usr/bin/env python3
"""Domain counts per interactor from TED, with InterPro as a cross-check.

Supplementary Fig S5C splits interactors into single- and multi-domain. The
rule it used was `len(set(InterPro accessions)) == 1`, read from
`pfam_domains_cache.pkl`, an orphan artifact nothing in the repository builds.
That count mixes every InterPro entry TYPE -- Family, Domain, Conserved_site,
Homologous_superfamily -- so a genuinely single-domain protein routinely
accrues three to five accessions and `n == 1` almost never fires. Only 125 of
968 interactors (12.9%) came out single-domain.

Two replacements are fetched here so the choice can be made on evidence:

  * TED (https://ted.cathdb.info), a structure-based domain parse of the
    AlphaFold models. Per domain it gives `chopping` (residue ranges),
    `nres_domain`, `consensus_level` and `cath_label`, so the domain count is
    unambiguous and the ranges additionally support a domain-vs-linker test for
    the mutated residue.
  * InterPro restricted to `?type=domain`, which drops the family and
    superfamily entries the old rule counted.

Both are cached on disk and the fetch is resumable, so this is a real producer
rather than another undocumented pickle.

TED keys on AlphaFold DB accessions, which have no isoform entries, so
accessions are reduced with `bare_accession()` before lookup -- the same
fallback `protein_class_stratification.build_row_groups` already applies.

Output:
  datasets/annotations/ted_domains_cache.json   raw responses, keyed by accession
  results/protein_class/domain_annotations.csv  one row per interactor:
      uniprot_id, n_ipr_all, n_ipr_domain, n_ted, ted_choppings,
      ted_consensus_levels

Usage:
    conda run -n ppi python src/analysis/fetch_ted_domains.py
    conda run -n ppi python src/analysis/fetch_ted_domains.py --source ted
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from paths import ANNOTATIONS_DIR, PROTEIN_CLASS_DIR, cv_reference_dir  # noqa: E402
from utils.identifiers import bare_accession  # noqa: E402

TED_URL = "https://ted.cathdb.info/api/v1/uniprot/summary/{acc}"
INTERPRO_URL = ("https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/"
                "uniprot/{acc}/?type=domain&page_size=100")

TED_CACHE = ANNOTATIONS_DIR / "ted_domains_cache.json"
INTERPRO_CACHE = ANNOTATIONS_DIR / "interpro_domains_cache.json"
PFAM_CACHE = ANNOTATIONS_DIR / "pfam_domains_cache.pkl"
OUT_CSV = PROTEIN_CLASS_DIR / "domain_annotations.csv"

DEFAULT_ROWS = cv_reference_dir() / "sahni_fragoza_train_rows.csv.gz"


def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f)
    tmp.replace(path)


def fetch_one(url: str, retries: int = 3, timeout: int = 60):
    """Parsed JSON, or None when the resource could not be reached.

    404 and 204 are both ANSWERS -- the protein has no entry, and InterPro
    returns 204 with an empty body for a protein with no entry of the requested
    type -- so both are cached as an empty result and a rerun does not ask
    again. Treating 204 as an error instead left 234 of 968 accessions
    permanently unresolved, because the retry hit the same empty body every
    time. Anything else is retried, then gives up and returns None WITHOUT
    caching, so the next run retries it.
    """
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code in (204, 404):
                return {}
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                print(f"  WARNING: {url}: {exc}", flush=True)
                return None
            time.sleep(2 ** attempt)
    return None


def fetch_all(accessions: list[str], url_template: str, cache_path: Path,
              label: str, pause: float = 0.1) -> dict:
    cache = load_cache(cache_path)
    todo = [a for a in accessions if a not in cache]
    print(f"{label}: {len(cache)} cached, {len(todo)} to fetch", flush=True)
    for n, acc in enumerate(todo, 1):
        result = fetch_one(url_template.format(acc=acc))
        if result is not None:
            cache[acc] = result
        if n % 50 == 0:
            print(f"  {n}/{len(todo)}", flush=True)
            save_cache(cache, cache_path)
        time.sleep(pause)
    save_cache(cache, cache_path)
    return cache


def ted_domains(entry: dict) -> list[dict]:
    return entry.get("data", []) if entry else []


def interpro_domains(entry: dict) -> list[str]:
    if not entry:
        return []
    return [r["metadata"]["accession"] for r in entry.get("results", [])]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=Path, default=DEFAULT_ROWS,
                   help=f"row table naming the interactors (default: {DEFAULT_ROWS})")
    p.add_argument("--source", choices=["both", "ted", "interpro"], default="both",
                   help="which resource to fetch (default: both)")
    p.add_argument("--output", type=Path, default=OUT_CSV,
                   help=f"comparison table (default: {OUT_CSV})")
    args = p.parse_args()

    if not args.rows.exists():
        p.error(f"not found: {args.rows}")

    rows = pd.read_csv(args.rows, usecols=["interactor"])
    accessions = sorted({bare_accession(str(a)) for a in rows["interactor"]})
    print(f"{len(accessions)} distinct interactors (isoform suffixes reduced)")

    ted = fetch_all(accessions, TED_URL, TED_CACHE, "TED") \
        if args.source in ("both", "ted") else load_cache(TED_CACHE)
    ipr = fetch_all(accessions, INTERPRO_URL, INTERPRO_CACHE, "InterPro (type=domain)") \
        if args.source in ("both", "interpro") else load_cache(INTERPRO_CACHE)

    # The rule currently on the figure, for the third column of the comparison.
    ipr_all: dict[str, int] = {}
    if PFAM_CACHE.exists():
        with open(PFAM_CACHE, "rb") as f:
            ipr_all = {k: len(set(v)) for k, v in pickle.load(f)["hits"].items()}

    records = []
    for acc in accessions:
        doms = ted_domains(ted.get(acc, {}))
        ipr_hits = interpro_domains(ipr.get(acc, {}))
        records.append({
            "uniprot_id": acc,
            "n_ipr_all": ipr_all.get(acc),
            "n_ipr_domain": len(ipr_hits) if acc in ipr else None,
            "n_ted": len(doms) if acc in ted else None,
            "ted_choppings": ";".join(d.get("chopping", "") for d in doms),
            "ted_consensus_levels": ";".join(d.get("consensus_level", "") for d in doms),
        })

    df = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved -> {args.output}")

    for col in ("n_ipr_all", "n_ipr_domain", "n_ted"):
        vals = df[col].dropna()
        if vals.empty:
            print(f"  {col}: no data")
            continue
        single = int((vals == 1).sum())
        print(f"  {col}: {len(vals)} proteins, {single} single-domain "
              f"({100.0 * single / len(vals):.1f}%), median {vals.median():.0f}, "
              f"max {int(vals.max())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
