#!/usr/bin/env python3
"""Build mutually-exclusive autosomal-recessive-only (AR) and autosomal-dominant-only
(AD) disease-gene UniProt sets from the ClinGen Gene-Disease Validity curations.

A gene is AR-only if its full mode-of-inheritance (MOI) set across all curated
diseases is exactly {AR} (no other MOI, including XL — X-linked inheritance is
effectively dominant in males but recessive in females, so it must not be folded
into either autosomal bucket). AD-only is defined analogously. Genes with both an
AR and an AD curation (for different diseases) are excluded from both sets.

Input:
    /data/ross/ppi_lossgain/interaction_loss/ClinGen_MOI.csv
    Curated per: Chen Y, Fayer S, Jain S, Benazouz M, Sverchkov Y, Stone J, Sharma H,
    Bergquist T, Stewart R, Mooney SD, Craven M, Radivojac P, Starita LM, Fowler DM,
    Pejaver V. Gene- and domain-aware calibration increases the clinical utility of
    variant effect predictors. bioRxiv [Preprint]. 2026 Mar 31:2026.02.17.706269.
    doi: 10.64898/2026.02.17.706269. PMID: 41756877; PMCID: PMC12934735.

Output:
    /data/ross/ppi_lossgain/interaction_loss/clingen_ar_ad_uniprot_sets.pkl
    {"AR": set[uniprot_str], "AD": set[uniprot_str]}

Usage:
    conda run -n ppi python src/analysis/build_ar_ad_gene_sets.py
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA_ROOT  # noqa: E402


_BASE = DATA_ROOT
CLINGEN_CSV       = _BASE / "ClinGen_MOI.csv"
GENE_TO_UNIPROT    = _BASE / "cosmic_mutations" / "gene_symbol_to_uniprot.pkl"
UNIPROT_TO_GENE_TSV = _BASE / "gnomad" / "gnomad_uniprot_to_gene.tsv"
OUT_PKL            = _BASE / "clingen_ar_ad_uniprot_sets.pkl"


def load_gene_moi(csv_path: Path) -> dict[str, set[str]]:
    df = pd.read_csv(csv_path, skiprows=4)
    df = df[df["MOI"].isin(["AR", "AD", "XL", "SD", "MT", "UD"])]
    gene_moi: dict[str, set[str]] = defaultdict(set)
    for gene, moi in zip(df["GENE SYMBOL"], df["MOI"]):
        gene_moi[gene].add(moi)
    return gene_moi


def load_gene_to_uniprot() -> dict[str, str]:
    with open(GENE_TO_UNIPROT, "rb") as f:
        primary: dict[str, str] = pickle.load(f)

    fallback: dict[str, str] = {}
    df = pd.read_csv(UNIPROT_TO_GENE_TSV, sep="\t")
    for uniprot, gene_names in zip(df["Entry"], df["Gene Names"]):
        if pd.isna(gene_names):
            continue
        primary_symbol = str(gene_names).split()[0]
        fallback.setdefault(primary_symbol, uniprot)

    combined = dict(fallback)
    combined.update(primary)  # pkl takes precedence — unambiguous single-value mapping
    return combined


def map_genes_to_uniprot(genes: set[str], gene_to_uniprot: dict[str, str]) -> tuple[set[str], list[str]]:
    mapped, unmapped = set(), []
    for g in genes:
        u = gene_to_uniprot.get(g)
        if u:
            mapped.add(u)
        else:
            unmapped.append(g)
    return mapped, unmapped


def main() -> None:
    print("Loading ClinGen MOI curations...", flush=True)
    gene_moi = load_gene_moi(CLINGEN_CSV)
    print(f"  {len(gene_moi):,} unique genes", flush=True)

    ar_only_genes = {g for g, mois in gene_moi.items() if mois == {"AR"}}
    ad_only_genes = {g for g, mois in gene_moi.items() if mois == {"AD"}}
    both_genes    = {g for g, mois in gene_moi.items() if "AR" in mois and "AD" in mois}
    print(f"  AR-only genes: {len(ar_only_genes):,}", flush=True)
    print(f"  AD-only genes: {len(ad_only_genes):,}", flush=True)
    print(f"  AR-and-AD genes (excluded from both): {len(both_genes):,}", flush=True)

    print("Loading gene symbol -> UniProt mapping...", flush=True)
    gene_to_uniprot = load_gene_to_uniprot()
    print(f"  {len(gene_to_uniprot):,} gene symbols mapped", flush=True)

    ar_uniprots, ar_unmapped = map_genes_to_uniprot(ar_only_genes, gene_to_uniprot)
    ad_uniprots, ad_unmapped = map_genes_to_uniprot(ad_only_genes, gene_to_uniprot)

    print(f"\nAR-only: {len(ar_uniprots):,}/{len(ar_only_genes):,} genes mapped to UniProt "
          f"({len(ar_unmapped)} unmapped)", flush=True)
    if ar_unmapped:
        print(f"  Unmapped AR genes: {sorted(ar_unmapped)}", flush=True)
    print(f"AD-only: {len(ad_uniprots):,}/{len(ad_only_genes):,} genes mapped to UniProt "
          f"({len(ad_unmapped)} unmapped)", flush=True)
    if ad_unmapped:
        print(f"  Unmapped AD genes: {sorted(ad_unmapped)}", flush=True)

    overlap = ar_uniprots & ad_uniprots
    if overlap:
        print(f"  WARNING: {len(overlap)} UniProt IDs map to both AR-only and AD-only "
              f"genes (isoform/paralog collision in gene->uniprot mapping): {sorted(overlap)}",
              flush=True)

    with open(OUT_PKL, "wb") as f:
        pickle.dump({"AR": ar_uniprots, "AD": ad_uniprots}, f)
    print(f"\nSaved -> {OUT_PKL}", flush=True)


if __name__ == "__main__":
    main()
