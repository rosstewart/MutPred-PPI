#!/usr/bin/env python
"""Map the Fu et al. ASD missense variants to the BioGRID direct-binding network.

Migrates `map_asd_ndd_variants.ipynb`, which lived outside version control.
This is the `asd` inference database (previously spelled `tulika_autism`):
de novo missense variants from the Fu et al. autism study, in two tables (SebatLab and LIT) that are simply
concatenated. It is a DIFFERENT dataset from `map_neurodev.py`, which handles the
NeuroDev case/control set; the two are not interchangeable.

Reads
    {--variant-dir}/Fu_variants_SebatLab.tsv
    {--variant-dir}/Fu_variants_LIT.tsv      (columns: SYMBOL, Protein_position,
                                              Amino_acids as 'REF/ALT', ...)
    HGNC REST API (or --gene-symbol-to-uniprot cache) for gene -> UniProt
    {--biogrid-dir}/biogrid_dirbind_uniprot_to_interactors.pkl
    {--biogrid-dir}/uniprot_dirbind_to_seq.pkl

Writes
    all_complexes.pkl  {(mutated_uniprot, partner_uniprot)}, one orientation only
    id_to_seq.pkl      {'<uniprot>': seq} plus {'<uniprot> <variant>': mutant seq}
    all_variants.pkl   {(uniprot, variant, partner)} triplets
    gene_symbol_to_uniprot.pkl  (the resolved HGNC mapping, for reruns)

TRANSCRIPTION, NOT REDESIGN
---------------------------
Filters and conventions are verbatim; only I/O, paths and the CLI changed.
Points worth stating because they look like bugs and are not:

  * Positions are 1-BASED. `Protein_position` is asserted non-zero for exactly
    that reason, the variant string keeps the 1-based number, and the only -1 is
    the `variant[1:-1] - 1` needed to index a Python string. No other re-basing.
  * Sequences come from BioGRID's `uniprot_dirbind_to_seq.pkl`, NOT from
    `asd_uniprots.fasta`. The notebook parsed that FASTA into a
    `seq_dict` and then never used it; it is therefore not read here. A variant
    whose UniProt entry has no dirbind BioGRID sequence is dropped, as is one
    whose wild-type residue disagrees with that sequence at the given position.
  * A gene with several UniProt ids is resolved by asking UniProt which of them
    is reviewed (Swiss-Prot) AND human -- see build_gene_symbol_to_uniprot. The
    notebook instead picked whichever had an AlphaFold v4 file on disk, which is
    a proxy for review status and a poor one; that path survives as an offline
    fallback and via --no-uniprot-api.
  * `clean_complexes` keeps one orientation of each pair -- interactions are
    unordered -- and the triplet expansion then re-adds the partner in whichever
    slot the mutated protein does not occupy, so no edge is lost.

Usage:
    python map_asd_ndd.py \\
        --variant-dir $MUTPRED_DATA_ROOT/asd \\
        --biogrid-dir $MUTPRED_DATA_ROOT/biogrid \\
        --af-pdb-dir $MUTPRED_DATA_ROOT/alphafold_v4_human \\
        --af-pdb-pattern 'AF-{uniprot_id}-F1-model_v4.pdb.gz' \\
        --output-dir ./scratch_asd
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

from paths import EXTERNAL_DIR  # noqa: E402
from data_processing.variant_databases.biogrid_common import (  # noqa: E402
    build_variant_triplets, clean_complexes, get_complexes_in_biogrid,
    load_biogrid,
)
from utils import mutations  # noqa: E402


def read_fu_variants(variant_dir):
    """Concatenate the two Fu tables -> ({'GENE VARIANT'}, {gene}) sets."""
    import pandas as pd
    df1 = pd.read_csv(os.path.join(variant_dir, "Fu_variants_SebatLab.tsv"), sep="\t")
    df2 = pd.read_csv(os.path.join(variant_dir, "Fu_variants_LIT.tsv"), sep="\t")
    df = pd.concat([df1, df2], ignore_index=True)

    gene_ids, gene_vt_ids = set(), set()
    for _, row in df.iterrows():
        assert row["Protein_position"] != 0  # positions are 1-based, not 0-based
        ref, alt = row["Amino_acids"].split("/")
        gene_vt_ids.add(f"{row['SYMBOL']} {ref}{row['Protein_position']}{alt}")
        gene_ids.add(row["SYMBOL"])
    print(f"{len(df)} rows, {len(gene_vt_ids)} gene-variants, {len(gene_ids)} genes",
          flush=True)
    return gene_vt_ids, gene_ids


def fetch_hgnc_info(gene_list):
    """Gene symbol -> HGNC record, via the HGNC REST API."""
    import pandas as pd
    import requests
    base_url = "https://rest.genenames.org/fetch/symbol/"
    headers = {"Accept": "application/json"}
    data = []
    for gene in gene_list:
        response = requests.get(base_url + gene, headers=headers)
        if response.status_code == 200:
            result = response.json()["response"]["docs"]
            if result:
                info = result[0]  # first match if several
                data.append({
                    "Gene Symbol": info.get("symbol"),
                    "UniProt ID": ", ".join(info.get("uniprot_ids", [])),
                })
        else:
            print(f"Failed to fetch data for {gene}", flush=True)
    return pd.DataFrame(data)


def _reviewed_human_accessions(accessions):
    """Subset of `accessions` that UniProt lists as reviewed (Swiss-Prot), human.

    Asks the question directly instead of inferring it. Returns None if the API
    cannot be reached, so the caller can fall back rather than silently guess.
    """
    import requests

    try:
        query = " OR ".join(f"accession:{a}" for a in accessions)
        r = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={"query": f"({query}) AND reviewed:true AND organism_id:9606",
                    "fields": "accession", "format": "tsv", "size": 500},
            timeout=30)
        if r.status_code != 200:
            return None
        return {line.strip() for line in r.text.splitlines()[1:] if line.strip()}
    except Exception:
        return None


def build_gene_symbol_to_uniprot(df_hgnc, af_pdb_dir, af_pdb_pattern,
                                 use_uniprot_api: bool = True):
    """Gene symbol -> one UniProt accession, resolving HGNC's multi-id entries.

    HGNC sometimes lists several accessions for one symbol. Resolution order:

      1. **UniProt REST: reviewed (Swiss-Prot) AND human.** This asks the actual
         question -- which accession is the canonical human entry -- rather than
         inferring it.
      2. AlphaFold v4 model present on disk. This is what the original notebook
         did, and it is a poor proxy: AlphaFold DB coverage reflects that
         database's build, not UniProt review status, so a Swiss-Prot entry
         missing from the local mirror loses to a TrEMBL one that happens to be
         present. Kept only as an offline fallback, and to reproduce the
         published mapping via `--no-uniprot-api`.
      3. First accession HGNC listed.

    Blast radius here is one gene (NRXN1), so 1 and 2 agree in practice for this
    dataset; the ordering matters for anything that widens it.

    The notebook used an unzipped AlphaFold v4 mirror that no longer exists;
    the gzipped one works via `--af-pdb-dir` / `--af-pdb-pattern`.
    """
    gene_symbol_to_uniprot = {}
    for _, row in df_hgnc.iterrows():
        gene = row["Gene Symbol"]
        uniprot_id = row["UniProt ID"]
        if ", " in uniprot_id:
            print(gene, "multiple", uniprot_id, flush=True)
            candidates = uniprot_id.split(", ")
            found = False
            if use_uniprot_api:
                reviewed = _reviewed_human_accessions(candidates)
                if reviewed:
                    for candidate in candidates:
                        if candidate in reviewed:
                            uniprot_id, found = candidate, True
                            print(f"  reviewed human: {candidate}", flush=True)
                            break
            for potential_uniprot_id in ([] if found else candidates):
                path = os.path.join(
                    af_pdb_dir, af_pdb_pattern.format(uniprot_id=potential_uniprot_id))
                if os.path.exists(path):
                    uniprot_id = potential_uniprot_id
                    found = True
                    break
            if not found:
                uniprot_id = candidates[0]
        assert gene not in gene_symbol_to_uniprot
        gene_symbol_to_uniprot[gene] = uniprot_id
    return gene_symbol_to_uniprot


def apply_variants(gene_vt_ids, gene_symbol_to_uniprot, id_to_seq):
    """Insert '<uniprot> <variant>' -> mutant sequence for every valid variant."""
    for vt_id in gene_vt_ids:
        gene_id, variant = vt_id.split(" ")
        uniprot_id = gene_symbol_to_uniprot[gene_id]
        if "del" in variant or "ins" in variant or "Sec" in variant:
            continue
        wt_res = variant[0]
        mt_idx = mutations.index(variant)  # 1-based variant -> 0-based string index
        mt_res = variant[-1]
        if uniprot_id not in id_to_seq:
            continue
        if mt_idx >= len(id_to_seq[uniprot_id]) or id_to_seq[uniprot_id][mt_idx] != wt_res:
            continue  # bad sequence or variant
        vt_seq = list(id_to_seq[uniprot_id])
        vt_seq[mt_idx] = mt_res
        vt_seq = "".join(vt_seq)
        key = f"{uniprot_id} {variant}"
        if key in id_to_seq:
            # two HGNC genes can map to the same UniProt entry
            assert id_to_seq[key] == vt_seq
        id_to_seq[key] = vt_seq
    return id_to_seq


def sanity_check(all_complexes, id_to_seq, uniprot_to_interactors):
    """Every complex must be a reciprocal dirbind edge, every id a known protein."""
    for mutated_prot, partner_prot in all_complexes:
        assert not (mutated_prot not in uniprot_to_interactors
                    or partner_prot not in uniprot_to_interactors
                    or partner_prot not in uniprot_to_interactors[mutated_prot]
                    or mutated_prot not in uniprot_to_interactors[partner_prot])
    for vt_id in id_to_seq:
        assert vt_id.split(" ")[0] in uniprot_to_interactors


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant-dir", required=True,
                    help="holds Fu_variants_SebatLab.tsv and Fu_variants_LIT.tsv")
    ap.add_argument("--biogrid-dir", required=True,
                    help="holds the BioGRID dirbind pickles")
    ap.add_argument("--gene-symbol-to-uniprot", default=None,
                    help="cached mapping pickle; fetched from HGNC when absent")
    ap.add_argument("--af-pdb-dir",
                    default=str(EXTERNAL_DIR / "alphafold_v4_human"),
                    help="AlphaFold v4 human models, used only to break "
                         "multi-UniProt ties")
    ap.add_argument("--af-pdb-pattern", default="AF-{uniprot_id}-F1-model_v4.pdb")
    ap.add_argument("--no-uniprot-api", action="store_true",
                    help="skip the reviewed/human UniProt query and fall straight "
                         "back to the AlphaFold-file heuristic the notebook used")
    ap.add_argument("--output-dir", default="./scratch_asd")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gene_vt_ids, gene_ids = read_fu_variants(args.variant_dir)

    if args.gene_symbol_to_uniprot and os.path.exists(args.gene_symbol_to_uniprot):
        with open(args.gene_symbol_to_uniprot, "rb") as f:
            gene_symbol_to_uniprot = pickle.load(f)
    else:
        gene_symbol_to_uniprot = build_gene_symbol_to_uniprot(
            fetch_hgnc_info(list(gene_ids)), args.af_pdb_dir, args.af_pdb_pattern,
            use_uniprot_api=not args.no_uniprot_api)
        with open(os.path.join(args.output_dir, "gene_symbol_to_uniprot.pkl"), "wb") as f:
            pickle.dump(gene_symbol_to_uniprot, f)
    print(f"{len(gene_symbol_to_uniprot)} gene->UniProt mappings", flush=True)

    uniprot_to_interactors, uniprot_to_seq = load_biogrid(args.biogrid_dir)
    uniprot_wts = set(gene_symbol_to_uniprot.values())
    all_complexes = get_complexes_in_biogrid(
        uniprot_wts, uniprot_to_interactors, uniprot_to_seq)
    print(f"{len(all_complexes)} BioGRID dirbind complexes", flush=True)

    id_to_seq = {}
    for uniprot_id in {u for pair in all_complexes for u in pair}:
        assert uniprot_id in uniprot_to_seq
        id_to_seq[uniprot_id] = uniprot_to_seq[uniprot_id]
    id_to_seq = apply_variants(gene_vt_ids, gene_symbol_to_uniprot, id_to_seq)
    n_vt = sum(1 for k in id_to_seq if " " in k)
    print(f"id_to_seq: {len(id_to_seq)} entries ({n_vt} variants)", flush=True)

    sanity_check(all_complexes, id_to_seq, uniprot_to_interactors)
    complexes_cleaned = clean_complexes(all_complexes)
    print(f"complexes: {len(all_complexes)} -> {len(complexes_cleaned)} after "
          f"collapsing flipped duplicates", flush=True)

    all_variants = build_variant_triplets(id_to_seq, complexes_cleaned)
    unique_interactors = {t[2] for t in all_variants}
    unique_variants = {t[:2] for t in all_variants}
    print(f"{len(all_variants)} variant-interactor triplets "
          f"({len(unique_variants)} variants, {len(unique_interactors)} partners)",
          flush=True)

    with open(os.path.join(args.output_dir, "all_complexes.pkl"), "wb") as f:
        pickle.dump(complexes_cleaned, f)
    with open(os.path.join(args.output_dir, "id_to_seq.pkl"), "wb") as f:
        pickle.dump(id_to_seq, f)
    with open(os.path.join(args.output_dir, "all_variants.pkl"), "wb") as f:
        pickle.dump(all_variants, f)
    print(f"outputs written to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
