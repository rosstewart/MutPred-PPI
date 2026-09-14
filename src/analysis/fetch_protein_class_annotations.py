#!/usr/bin/env python3
"""Fetch GO functional class annotations for variant DB interactors from UniProt REST API.

Groups proteins into broad functional classes:
  kinase, transcription_factor, receptor_signaling, metabolic_enzyme,
  structural_cytoskeletal, ubiquitin_proteasome, dna_rna_binding, other

Saves results to:
  results/protein_class/protein_class_annotations.csv
    columns: uniprot_id, go_terms (semicolon-separated), protein_class

Usage:
    conda run -n ppi python src/analysis/fetch_protein_class_annotations.py
"""
from __future__ import annotations

import csv
import json
import time

import pandas as pd
import requests

# --- repo-relative path resolution (see src/paths.py) ---
from paths import DATASETS_DIR, REPO_ROOT  # noqa: E402
from utils.identifiers import bare_accession  # noqa: E402


_PUB = REPO_ROOT
# The canonical variant-database row tables, not the prediction TSVs. The TSVs
# carry a welded `complex_id`, and recovering the interactor from it meant
# `cid.split("_")[0]` -- a guess that is wrong for any RefSeq-style accession.
# These tables have `interactor` as its own column, so there is nothing to split.
_DB_DIR = DATASETS_DIR / "variant_dbs"
_OUT = _PUB / "results" / "protein_class" / "protein_class_annotations.csv"

# GO slim classifications: GO term ID → broad class
# These cover the most common functional classes in interactomes
GO_CLASS_RULES: list[tuple[str, list[str]]] = [
    # kinases: protein kinase activity
    ("kinase", ["GO:0016301", "GO:0004672", "GO:0004674", "GO:0004713",
                "GO:0004675", "GO:0004712", "GO:0016310"]),
    # transcription factors: DNA-binding TF activity
    ("transcription_factor", ["GO:0003700", "GO:0000981", "GO:0003677",
                               "GO:0006355", "GO:0001228"]),
    # ubiquitin/proteasome: ubiquitin ligase/proteasome/deubiquitinase
    ("ubiquitin_proteasome", ["GO:0004842", "GO:0006511", "GO:0016567",
                               "GO:0008234", "GO:0061630"]),
    # receptors/signaling: receptor activity, GTPase, second messenger
    ("receptor_signaling", ["GO:0004872", "GO:0007165", "GO:0003924",
                             "GO:0007186", "GO:0007187", "GO:0004871",
                             "GO:0007264", "GO:0005096"]),
    # DNA/RNA binding/processing: helicase, nuclease, RNA binding
    ("dna_rna_binding", ["GO:0003723", "GO:0003684", "GO:0004386",
                          "GO:0006259", "GO:0006396", "GO:0016779"]),
    # structural/cytoskeletal: actin, tubulin, intermediate filament, collagen
    ("structural_cytoskeletal", ["GO:0005200", "GO:0005198", "GO:0045095",
                                  "GO:0005856", "GO:0030054"]),
    # metabolic enzyme: general catalytic activity (checked last — most specific wins)
    ("metabolic_enzyme", ["GO:0003824", "GO:0016787", "GO:0016740",
                           "GO:0016491", "GO:0016853"]),
]


def classify_go_terms(go_terms: list[str]) -> str:
    """Return the most specific protein class for a list of GO term IDs."""
    go_set = set(go_terms)
    for class_name, class_go_ids in GO_CLASS_RULES:
        if go_set & set(class_go_ids):
            return class_name
    return "other"


def fetch_go_terms_batch(uniprot_ids: list[str], retries: int = 3) -> dict[str, list[str]]:
    """Fetch GO term IDs for a batch of UniProt accessions via REST API.

    Returns {uniprot_id: [go_term_id, ...]}.
    """
    # UniProt batch REST endpoint
    query = " OR ".join(f"accession:{uid}" for uid in uniprot_ids)
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "fields": "accession,go",
        "format": "json",
        "size": len(uniprot_ids),
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            result: dict[str, list[str]] = {}
            for entry in data.get("results", []):
                acc = entry.get("primaryAccession", "")
                go_terms = []
                for ref in entry.get("uniProtKBCrossReferences", []):
                    if ref.get("database") == "GO":
                        go_terms.append(ref["id"])
                result[acc] = go_terms
            return result
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [WARN] batch fetch failed: {e}")
    return {}


def collect_all_interactors() -> set[str]:
    """Every mutated protein across the variant-database row tables.

    `bare_accession` is applied because the UniProt search endpoint has no entry
    for an isoform accession -- `accession:O14787-2` returns nothing, while the
    GO annotation wanted here is a property of the parent entry anyway. This is
    the one place the isoform suffix is deliberately dropped.
    """
    tables = sorted(_DB_DIR.glob("*_rows.csv.gz"))
    if not tables:
        raise FileNotFoundError(
            f"no *_rows.csv.gz under {_DB_DIR}. The variant-database row tables "
            f"ship in the Zenodo datasets/ bundle; see docs/DATA_PREPARATION.md.")

    interactors: set[str] = set()
    for table in tables:
        df = pd.read_csv(table, usecols=["interactor"])
        interactors.update(bare_accession(a) for a in df["interactor"])
        print(f"  {table.name}: {len(df)} rows", flush=True)
    return interactors


def main() -> None:
    print("Collecting interactors from all variant DBs...", flush=True)
    all_interactors = sorted(collect_all_interactors())
    print(f"  {len(all_interactors)} unique interactors", flush=True)

    # Load already-fetched results to allow resuming
    done: dict[str, tuple[list[str], str]] = {}
    if _OUT.exists():
        df_done = pd.read_csv(_OUT)
        for _, row in df_done.iterrows():
            go_terms = row["go_terms"].split(";") if pd.notna(row["go_terms"]) and row["go_terms"] else []
            done[row["uniprot_id"]] = (go_terms, row["protein_class"])
        print(f"  {len(done)} already fetched", flush=True)

    to_fetch = [uid for uid in all_interactors if uid not in done]
    print(f"  {len(to_fetch)} to fetch", flush=True)

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if _OUT.exists() else "w"
    with open(_OUT, mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if mode == "w":
            writer.writerow(["uniprot_id", "go_terms", "protein_class"])

        batch_size = 90  # UniProt API caps at 100 OR conditions per query
        n_fetched = 0
        for i in range(0, len(to_fetch), batch_size):
            batch = to_fetch[i:i + batch_size]
            print(f"  Fetching batch {i // batch_size + 1}/{(len(to_fetch) - 1) // batch_size + 1} "
                  f"({len(batch)} IDs)...", end=" ", flush=True)
            go_map = fetch_go_terms_batch(batch)
            time.sleep(0.5)  # rate limiting

            for uid in batch:
                go_terms = go_map.get(uid, [])
                protein_class = classify_go_terms(go_terms)
                writer.writerow([uid, ";".join(go_terms), protein_class])
                n_fetched += 1
            csvfile.flush()
            print(f"OK ({n_fetched} total fetched)", flush=True)

    print(f"\nWrote annotations to {_OUT}", flush=True)

    # Summary
    df = pd.read_csv(_OUT)
    print("\nProtein class distribution:")
    print(df["protein_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
