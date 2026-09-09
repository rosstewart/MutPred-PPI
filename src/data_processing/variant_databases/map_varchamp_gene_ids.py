#!/usr/bin/env python
"""Build gene_symbol -> UniProt ID mappings for VarChAMP VC1p/CAVA cohorts.

Ports logic that previously only existed in uncontrolled legacy notebooks
(`make_af3_json_input_varchamp1p_evaluation.ipynb`, `make_af3_json_input_cava_evaluation.ipynb`,
outside this repo) into a version-controlled, reusable script. Uses the same
HGNC REST API approach as `map_cosmic.py::fetch_hgnc_info()`, plus one extra
disambiguation rule for genes with multiple UniProt IDs: prefer whichever has
a precomputed AlphaFold structure available.

Unlike the original notebooks (which read raw VarChAMP scoring CSVs that no
longer exist on disk), this script derives the required gene-symbol list
directly from the AF3 contact-graph filenames actually used by the pipeline
(`{gene}_{orf}_{gene}_{orf}.mat`) — more robust, since it only requires
mapping genes the pipeline actually consumes, not a full raw-data superset.

Usage:
    conda run -n ppi python src/data_processing/variant_databases/map_varchamp_gene_ids.py \\
        --graph-dir $MUTPRED_DATA_ROOT/varchamp1p/af3_graphs \\
        --output $MUTPRED_DATA_ROOT/home/varchamp1p/gene_symbol_to_uniprot.pkl

    conda run -n ppi python src/data_processing/variant_databases/map_varchamp_gene_ids.py \\
        --graph-dir $MUTPRED_DATA_ROOT/cava/af3_graphs \\
        --output $MUTPRED_DATA_ROOT/home/cava/gene_symbol_to_uniprot.pkl
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
from pathlib import Path

import pandas as pd
import requests

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import EXTERNAL_DIR  # noqa: E402

# AFDB v4 human monomers, resolved via external/ (run scripts/link_external.sh).
_AF_STRUCT_DIR = str(EXTERNAL_DIR / "alphafold_v4_human")


def genes_from_graph_dir(graph_dir: str) -> list[str]:
    """Parse gene symbols out of {gene}_{orf}_{gene}_{orf}.mat filenames."""
    genes: set[str] = set()
    for path in glob.glob(f"{graph_dir}/*.mat"):
        complex_id = os.path.splitext(os.path.basename(path))[0]
        parts = complex_id.split("_")
        if len(parts) != 4:
            continue  # unexpected format, skip
        gene_a, _orf_a, gene_b, _orf_b = parts
        genes.add(gene_a)
        genes.add(gene_b)
    return sorted(genes)


def fetch_hgnc_info(gene_list: list[str]) -> pd.DataFrame:
    """Fetch gene->UniProt mapping via the HGNC REST API (same as map_cosmic.py)."""
    base_url = "https://rest.genenames.org/fetch/symbol/"
    headers = {"Accept": "application/json"}
    rows = []
    for gene in gene_list:
        resp = requests.get(base_url + gene, headers=headers)
        if resp.status_code != 200:
            print(f"  Failed to fetch {gene} (HTTP {resp.status_code})", flush=True)
            continue
        docs = resp.json()["response"]["docs"]
        if not docs:
            continue
        info = docs[0]
        rows.append({
            "Gene Symbol": info.get("symbol"),
            "UniProt ID": ", ".join(info.get("uniprot_ids", [])),
        })
    return pd.DataFrame(rows)


def build_gene_symbol_to_uniprot(df_hgnc: pd.DataFrame) -> dict[str, str]:
    """Resolve multi-UniProt genes by preferring the one with an AF structure."""
    mapping: dict[str, str] = {}
    for _, row in df_hgnc.iterrows():
        gene = row["Gene Symbol"]
        uid = row["UniProt ID"]
        if not uid:
            continue
        if ", " in uid:
            found = None
            for candidate in uid.split(", "):
                if os.path.exists(f"{_AF_STRUCT_DIR}/AF-{candidate}-F1-model_v4.pdb.gz"):
                    found = candidate
                    break
            if found is None:
                print(f"  WARNING: {gene} has multiple UniProt IDs ({uid}) and none "
                      "has a precomputed AF structure — taking the first", flush=True)
                found = uid.split(", ")[0]
            uid = found
        mapping[gene] = uid
    return mapping


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir", required=True,
                    help="Directory of {gene}_{orf}_{gene}_{orf}.mat contact graphs")
    p.add_argument("--output", required=True, help="Output gene_symbol_to_uniprot.pkl path")
    args = p.parse_args()

    print(f"Scanning {args.graph_dir} for gene symbols...", flush=True)
    genes = genes_from_graph_dir(args.graph_dir)
    print(f"  {len(genes)} unique gene symbols", flush=True)

    print("Fetching HGNC info...", flush=True)
    df_hgnc = fetch_hgnc_info(genes)
    print(f"  {len(df_hgnc)}/{len(genes)} genes resolved via HGNC", flush=True)

    mapping = build_gene_symbol_to_uniprot(df_hgnc)
    print(f"  {len(mapping)} gene -> UniProt mappings built", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(mapping, f)
    print(f"Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
