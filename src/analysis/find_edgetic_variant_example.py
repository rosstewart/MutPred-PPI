#!/usr/bin/env python3
"""Search COSMIC (recurrence ≥32), HGMD, and ClinVar Pathogenic for variants
MutPred-PPI classifies as Edgetic (mixed partner disruption, ≥3 tested partners),
to manually vet against the literature for a validated replacement for the
BRCA1 p.C61G manuscript example (which re-analysis showed is actually Quasi-null:
all 9 tested partners score 0.957-0.982 in brca1_extra/results/MutPred-PPI_preds.tsv).

NOT part of the manuscript — internal search tool only. Candidates require manual
literature/UniProt cross-check before use.

Usage:
    conda run -n ppi python src/analysis/find_edgetic_variant_example.py

Output:
    results_revisions/edgetic_example_search/candidates.tsv
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA_ROOT, REPO_ROOT  # noqa: E402


_PUB = REPO_ROOT
_BASE = DATA_ROOT
_HOME = _BASE / "home"
_DB   = _PUB / "results_revisions" / "variant_dbs"
_OUT  = _PUB / "results_revisions" / "edgetic_example_search"

sys.path.insert(0, str(_PUB / "src" / "analysis"))
from classify_variant_dbs import load_predictions, group_by_variant, classify_edgotype



COSMIC_MIN_RECURRENCE = 32
MIN_PARTNERS = 3

GENE_MAP_TSV = _BASE / "gnomad" / "gnomad_uniprot_to_gene.tsv"


def load_gene_map() -> dict[str, str]:
    df = pd.read_csv(GENE_MAP_TSV, sep="\t")
    gene_map = {}
    for uniprot, gene_names in zip(df["Entry"], df["Gene Names"]):
        if pd.isna(gene_names):
            continue
        gene_map[uniprot] = str(gene_names).split()[0]
    return gene_map


def find_edgetic_candidates(
    grouped: dict[tuple[str, str], dict[str, float]],
    allowed: dict[tuple[str, str], set[str]] | None,
    source_db: str,
    gene_map: dict[str, str],
) -> list[dict]:
    rows = []
    for (uniprot, variant), partner_scores in grouped.items():
        if allowed is not None:
            allowed_partners = allowed.get((uniprot, variant))
            if not allowed_partners:
                continue
            partner_scores = {p: s for p, s in partner_scores.items() if p in allowed_partners}
        if len(partner_scores) < MIN_PARTNERS:
            continue

        scores = list(partner_scores.values())
        edgotype = classify_edgotype(scores, threshold=0.5)
        if edgotype != "Edgetic":
            continue

        disrupted = {p: s for p, s in partner_scores.items() if s > 0.5}
        preserved = {p: s for p, s in partner_scores.items() if s <= 0.5}
        # separation = min disrupted score - max preserved score. Large positive values
        # indicate a clean bimodal split (convincing signature); values near 0 indicate
        # scores clustered right at the 0.5 threshold (noisy, unconvincing).
        separation = min(disrupted.values()) - max(preserved.values())
        rows.append({
            "source_db": source_db,
            "uniprot": uniprot,
            "gene_symbol": gene_map.get(uniprot, ""),
            "variant": variant,
            "n_partners": len(partner_scores),
            "n_disrupted": len(disrupted),
            "separation": separation,
            "disrupted_partners": ";".join(f"{p}:{s:.3f}" for p, s in disrupted.items()),
            "preserved_partners": ";".join(f"{p}:{s:.3f}" for p, s in preserved.items()),
        })
    return rows


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)

    print("Loading gene symbol mapping...", flush=True)
    gene_map = load_gene_map()
    print(f"  {len(gene_map):,} UniProt IDs mapped", flush=True)

    all_rows: list[dict] = []

    # --- ClinVar Pathogenic ---
    print("\nLoading ClinVar predictions...", flush=True)
    clinvar_pairs = load_predictions(str(_DB / "clinvar_mutpred_ppi_predictions.tsv"))
    clinvar_grouped = group_by_variant(clinvar_pairs)
    with open(_HOME / "clinvar" / "pathogenic_dirbind_variant_subset.pkl", "rb") as f:
        pathogenic_subset = pickle.load(f)
    allowed: dict[tuple[str, str], set[str]] = {}
    for u, v, p in pathogenic_subset:
        allowed.setdefault((u, v), set()).add(p)
    rows = find_edgetic_candidates(clinvar_grouped, allowed, "ClinVar Pathogenic", gene_map)
    print(f"  {len(rows):,} Edgetic candidates (≥{MIN_PARTNERS} partners)", flush=True)
    all_rows += rows

    # --- HGMD ---
    print("\nLoading HGMD predictions...", flush=True)
    hgmd_pairs = load_predictions(str(_DB / "hgmd_mutpred_ppi_predictions.tsv"))
    hgmd_grouped = group_by_variant(hgmd_pairs)
    with open(_HOME / "hgmd" / "variant_subset.pkl", "rb") as f:
        hgmd_subset = pickle.load(f)
    allowed = {}
    for u, v, p in hgmd_subset:
        allowed.setdefault((u, v), set()).add(p)
    rows = find_edgetic_candidates(hgmd_grouped, allowed, "HGMD", gene_map)
    print(f"  {len(rows):,} Edgetic candidates (≥{MIN_PARTNERS} partners)", flush=True)
    all_rows += rows

    # --- COSMIC recurrence >= 32 (gene-role-independent; all tested partners retained) ---
    print(f"\nLoading COSMIC predictions (recurrence >= {COSMIC_MIN_RECURRENCE})...", flush=True)
    cosmic_pairs = load_predictions(str(_DB / "cosmic_mutpred_ppi_predictions.tsv"))
    cosmic_grouped = group_by_variant(cosmic_pairs)
    with open(_BASE / "cosmic" / "vt_to_tumor_site.pkl", "rb") as f:
        vt_to_sites = pickle.load(f)
    cosmic_high_rec: set[tuple[str, str]] = set()
    for key, sites in vt_to_sites.items():
        if len(sites) >= COSMIC_MIN_RECURRENCE:
            parts = key.split(" ", 1)
            if len(parts) == 2:
                u, v1b = parts
                try:
                    var0 = f"{v1b[0]}{int(v1b[1:-1]) - 1}{v1b[-1]}"
                except ValueError:
                    continue
                cosmic_high_rec.add((u, var0))
    allowed = {(u, v): set(cosmic_grouped.get((u, v), {}).keys()) for (u, v) in cosmic_high_rec}
    rows = find_edgetic_candidates(cosmic_grouped, allowed, "COSMIC (recurrence>=32)", gene_map)
    print(f"  {len(rows):,} Edgetic candidates (≥{MIN_PARTNERS} partners)", flush=True)
    all_rows += rows

    if not all_rows:
        print("\nNo Edgetic candidates found.", flush=True)
        return

    df = pd.DataFrame(all_rows).sort_values("n_partners", ascending=False).reset_index(drop=True)
    out_tsv = _OUT / "candidates.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nSaved {len(df):,} candidates -> {out_tsv}", flush=True)

    print("\nTop 20 candidates (most tested partners, for a convincing multi-partner signature):",
          flush=True)
    for _, r in df.head(20).iterrows():
        print(f"  [{r['source_db']}] {r['gene_symbol'] or r['uniprot']} ({r['uniprot']}) "
              f"{r['variant']}: {r['n_disrupted']}/{r['n_partners']} partners disrupted "
              f"(separation={r['separation']:.3f})", flush=True)

    print("\nTop 20 CLEANEST candidates (largest score gap between disrupted/preserved "
          f"partners, ≥{MIN_PARTNERS} partners, min 4 partners for a more convincing split):",
          flush=True)
    clean = df[df["n_partners"] >= 4].sort_values("separation", ascending=False)
    for _, r in clean.head(20).iterrows():
        print(f"  [{r['source_db']}] {r['gene_symbol'] or r['uniprot']} ({r['uniprot']}) "
              f"{r['variant']}: {r['n_disrupted']}/{r['n_partners']} partners disrupted "
              f"(separation={r['separation']:.3f})", flush=True)


if __name__ == "__main__":
    main()
