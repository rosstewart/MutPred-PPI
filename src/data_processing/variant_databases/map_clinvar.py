#!/usr/bin/env python
"""Map ClinVar missense variants to the BioGRID direct-binding PPI network.

Consolidates FOUR notebooks that had no counterpart in this repo -- ClinVar was
the only live inference target with no `map_*.py`:

    get_all_clinvar_benigns.ipynb     \\
    get_all_clinvar_pathogenics.ipynb  >  --stage variants   (they were
    get_all_clinvar_vus.ipynb         /                       byte-identical
                                                              apart from the
                                                              significance
                                                              substring)
    get_clinvar_interactors.ipynb     -> --stage interactors

The three extraction notebooks differed only in which significance substring
they matched and which files they named, so they collapse to one function taking
a tier. That is the whole consolidation -- nothing else is merged, because the
remaining logic is genuinely per-database.

WHY THIS MATTERS BEYOND TIDINESS
--------------------------------
ClinVar is the only variant database whose pairs are not 100% BioGRID
direct-binding: 455 of its 6,836 pairs are not dirbind edges, and the 105,984
published rows sitting on them carry no clinical-significance label, so they were
scored and then discarded by every downstream analysis. Those pairs entered
through the AF3 *folding* set rather than through this filter -- `write_fasta`
below draws partners solely from `biogrid_dirbind_uniprot_to_interactors.pkl`.
Having this code in the repo is what makes that checkable.

TRANSCRIPTION, NOT REDESIGN
---------------------------
Filters, cutoffs and conventions are carried over verbatim; only I/O, paths and
the CLI are rewritten. Two conventions worth stating because they look like bugs
and are not:

  * ClinVar variant notation here is **1-based** (`idx = int(variant[1:-1]) - 1`),
    and the emitted `{tier}_wts_and_mts.fasta` headers keep that 1-based string.
    The 0-based `clinvar_interaction_loss_wt_and_vt.fasta` consumed by
    `variant_db_inference.variant_rows` is produced by a *later* step, not here.
  * Partner sequences come from `uniprot_to_seq.pkl`, the non-dirbind sequence
    map, while partners themselves come from the dirbind interactor map. That
    asymmetry is deliberate in the original: it only widens which dirbind
    partners can be given a sequence, and adds no non-dirbind edges.

Outputs (--stage variants, per tier):
    clinvar_all_filtered_{tier}s.pkl   : the filtered variant_summary rows
    uniprot_ids_{tier}.txt             : UniProt ids, for a UniProt FASTA fetch
    {tier}_wts_and_mts.fasta           : WT + mutant sequences, 1-based headers

Outputs (--stage interactors):
    clinvar_{tier}_uniprot_vts.pkl     : {'uniprot variant'} per tier
    clinvar_{tier}_wt_vt_partners.fasta: WT + variants + partner sequences
    {tier}_dirbind_variant_subset.pkl  : (uniprot, variant, partner) triplets
    dirbind_with_vus_id_to_seq.pkl     : merged id -> sequence across tiers

Usage:
    python map_clinvar.py --stage variants \\
        --variant-summary variant_summary.txt \\
        --hgnc hgnc_complete_set.txt \\
        --uniprot-fasta-dir . --output-dir /data/ross/clinvar

    python map_clinvar.py --stage interactors \\
        --clinvar-dir /data/ross/clinvar --biogrid-dir biogrid \\
        --output-dir $MUTPRED_DATA_ROOT/clinvar
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
from data_processing.variant_databases.biogrid_common import (  # noqa: E402
    build_variant_triplets, clean_complexes,
)
from utils import mutations  # noqa: E402
from utils.sequences import read_fasta as read_fasta_shared  # noqa: E402

TIERS = ("benign", "pathogenic", "vus")

# The significance substrings each tier matches in `ClinicalSignificance`.
# Matched case-insensitively on BOTH spellings exactly as the notebooks did --
# ClinVar mixes 'Pathogenic' and 'pathogenic' within one field.
TIER_SUBSTRINGS = {
    "benign":     ("Benign", "benign"),
    "pathogenic": ("Pathogenic", "pathogenic"),
    "vus":        ("Uncertain significance", "uncertain significance"),
}

THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Glu": "E",
    "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Sec": "U", "Pyl": "O", "Asx": "B", "Glx": "Z",
    "Xaa": "X", "Ter": "*",
}


# ── stage 1: variant extraction (the three get_all_clinvar_* notebooks) ───────

def filter_variant_summary(df, tier: str):
    """ClinVar `variant_summary.txt` -> single-nucleotide missense rows of `tier`.

    Filters, in the notebooks' own order:
      1. Type == 'single nucleotide variant'
      2. ClinicalSignificance contains the tier substring
      3. ReviewStatus != 'no assertion criteria provided'  (i.e. >=1 review star;
         the stricter single-submitter cut is commented out in all three
         notebooks and is deliberately NOT applied here)
      4. Name carries a 'p.' protein change that is neither synonymous ('=')
         nor a stop ('Ter')
      5. one row per VariationID
    """
    df = df[df["Type"] == "single nucleotide variant"]
    hi, lo = TIER_SUBSTRINGS[tier]
    df = df[df["ClinicalSignificance"].apply(
        lambda cs: isinstance(cs, str) and (hi in cs or lo in cs))]
    df = df[df["ReviewStatus"] != "no assertion criteria provided"]

    keep, protein_variant = [], []
    for _, row in df.iterrows():
        name = row["Name"]
        if not isinstance(name, str) or "p." not in name:
            keep.append(False)
            protein_variant.append(None)
            continue
        var = name.split("p.")[-1].replace(")", "")
        if var.endswith("=") or var[-3:] == "Ter":
            keep.append(False)
            protein_variant.append(None)
        else:
            keep.append(True)
            protein_variant.append(var)
    df = df.assign(ProteinVariant=protein_variant)[keep]

    seen, dedup = {}, []
    for _, row in df.iterrows():
        vid = row["VariationID"]
        if vid not in seen:
            seen[vid] = row["ProteinVariant"]
            dedup.append(True)
        else:
            dedup.append(False)
    return df[dedup]


def load_hgnc_uniprot(hgnc_path) -> dict:
    """hgnc_id -> space-joined UniProt ids ('P12345 Q67890' when ambiguous)."""
    import pandas as pd
    df = pd.read_csv(hgnc_path, sep="\t", low_memory=False)
    out = {}
    for _, row in df.iterrows():
        uids = row["uniprot_ids"]
        if isinstance(uids, str):
            out[row["hgnc_id"]] = uids.replace("|", " ")
    return out


def three_letter_to_one(protein_variant: str) -> str:
    """'Arg175His' -> 'R175H'. Preserves the notebook's 'X'-prefix special case."""
    if protein_variant[0] == "X" and protein_variant[1] != "a":
        return "X" + protein_variant[3:-3] + THREE_TO_ONE[protein_variant[-3:]]
    return (THREE_TO_ONE[protein_variant[:3]] + protein_variant[3:-3]
            + THREE_TO_ONE[protein_variant[-3:]])


def _dedup_preserving_order(variants):
    """Drop repeated characters per string, then drop anything ending in '*'.

    Carried over verbatim. `dict.fromkeys` on a variant string collapses repeated
    CHARACTERS, which is what the original did; the trailing-'*' test is what
    removes stop-gain calls that survived the earlier filter.
    """
    out = []
    for s in variants:
        unique_chars = "".join(dict.fromkeys(s))
        if unique_chars.endswith("*"):
            continue
        out.append(unique_chars)
    return out


def _uniprot_header_key(header: str) -> str:
    """`sp|Q30154|DRB5_HUMAN ...` -> `Q30154`; falls back to the whole header
    if there is no second pipe field."""
    parts = header.split("|")
    return parts[1] if len(parts) > 1 else header


def parse_uniprot_fasta(path) -> dict:
    """UniProt FASTA -> {accession: sequence}, keyed on the '|'-delimited id."""
    return read_fasta_shared(path, _uniprot_header_key, on_duplicate="last")


def write_wts_and_mts(uniprot_variants: dict, seqs: dict, out_path) -> int:
    """Emit WT + mutant sequences, validating each variant against its sequence.

    Positions are 1-BASED in ClinVar notation, hence `int(variant[1:-1]) - 1`.
    Where an HGNC row named several UniProt ids, the first whose sequence carries
    the wild-type residue at that position wins -- the original's behaviour.
    """
    written, n = set(), 0
    with open(out_path, "w") as fh:
        for uniprot_id, variants in uniprot_variants.items():
            candidates = uniprot_id.split(" ")
            for variant in variants:
                wt_aa, mt_aa = variant[0], variant[-1]
                try:
                    idx = mutations.index(variant)
                except ValueError:
                    continue
                for acc in candidates:
                    seq = seqs.get(acc)
                    if seq is None or idx >= len(seq) or seq[idx] != wt_aa:
                        continue
                    mt = seq[:idx] + mt_aa + seq[idx + 1:]
                    if acc not in written:
                        fh.write(f">{acc}\n{seq}\n")
                        written.add(acc)
                    fh.write(f">{acc} {variant}\n{mt}\n")
                    n += 1
                    break
    return n


def run_variants(args) -> None:
    import pandas as pd
    print(f"reading {args.variant_summary} ...", flush=True)
    summary = pd.read_csv(args.variant_summary, sep="\t", low_memory=False)
    hgnc = load_hgnc_uniprot(args.hgnc)
    print(f"  {len(summary)} rows; {len(hgnc)} HGNC->UniProt entries", flush=True)

    for tier in args.tiers:
        df = filter_variant_summary(summary, tier)
        print(f"{tier}: {len(df)} filtered rows", flush=True)
        df.to_pickle(os.path.join(args.output_dir, f"clinvar_all_filtered_{tier}s.pkl"))

        uniprot_variants, missing = {}, 0
        for _, row in df.iterrows():
            key = hgnc.get(row["HGNC_ID"])
            if key is None:
                missing += 1
                continue
            uniprot_variants.setdefault(key, []).append(
                three_letter_to_one(row["ProteinVariant"]))
        uniprot_variants = {k: _dedup_preserving_order(v)
                            for k, v in uniprot_variants.items()}
        print(f"  {len(uniprot_variants)} proteins, {missing} rows without a "
              f"UniProt id", flush=True)

        with open(os.path.join(args.output_dir, f"uniprot_ids_{tier}.txt"), "w") as fh:
            fh.write(" ".join(uniprot_variants) + " ")

        fasta = os.path.join(args.uniprot_fasta_dir, f"uniprot_all_{tier}s.fasta")
        if not os.path.exists(fasta):
            print(f"  no {fasta}; fetch it from UniProt using uniprot_ids_{tier}.txt "
                  f"then re-run", flush=True)
            continue
        n = write_wts_and_mts(uniprot_variants, parse_uniprot_fasta(fasta),
                              os.path.join(args.output_dir, f"{tier}_wts_and_mts.fasta"))
        print(f"  wrote {n} validated variants", flush=True)


# ── stage 2: partner selection (get_clinvar_interactors) ─────────────────────

def write_fasta(tier: str, clinvar_dir, biogrid_dir, output_dir, fasta_suffix):
    """WT + variant + partner sequences for one tier, restricted to dirbind.

    A protein absent from `biogrid_dirbind_uniprot_to_interactors.pkl` is dropped
    outright -- this is the physical-direct-binding-evidence filter, and it is
    the only place partners are chosen.
    """
    with open(f"{biogrid_dir}/biogrid_dirbind_uniprot_to_interactors.pkl", "rb") as f:
        uniprot_to_interactors = pickle.load(f)
    # Intentionally the NON-dirbind sequence map; see the module docstring.
    with open(f"{biogrid_dir}/uniprot_to_seq.pkl", "rb") as f:
        uniprot_to_seq = pickle.load(f)

    wt_complexes, id_to_seq = set(), {}
    uniprot_id, variant = None, None
    with open(f"{clinvar_dir}/{tier}{fasta_suffix}") as fh:
        for line in fh:
            if line.startswith(">"):
                header = line[1:].strip()
                if " " in header:
                    parts = header.split(" ")
                    if len(parts) != 2:
                        uniprot_id, variant = None, None
                        continue
                    uniprot_id, variant = parts
                else:
                    uniprot_id, variant = header, None
                if uniprot_id not in uniprot_to_interactors:
                    uniprot_id, variant = None, None
            elif uniprot_id is not None:
                seq = line.strip()
                key = uniprot_id if variant is None else f"{uniprot_id} {variant}"
                id_to_seq[key] = seq
                for partner in uniprot_to_interactors[uniprot_id]:
                    # '|' in a partner id means BioGRID could not disambiguate it;
                    # such ids are absent from uniprot_to_seq and so skipped here.
                    if partner not in id_to_seq and partner in uniprot_to_seq:
                        id_to_seq[partner] = uniprot_to_seq[partner]
                    if variant is None and partner in uniprot_to_seq:
                        wt_complexes.add((uniprot_id, partner))

    out = os.path.join(output_dir, f"clinvar_{tier}_wt_vt_partners.fasta")
    with open(out, "w") as fh:
        for id_, seq in id_to_seq.items():
            fh.write(f">{id_}\n{seq}\n")
    n_var = sum(1 for k in id_to_seq if " " in k)
    print(f"{tier}: {len(id_to_seq)} sequences ({n_var} variants), "
          f"{len(wt_complexes)} complexes -> {os.path.basename(out)}", flush=True)
    return wt_complexes, id_to_seq


def drop_cross_tier_variants(per_tier: dict) -> int:
    """Remove any variant classified in more than one significance tier.

    A variant called both benign and pathogenic carries no usable label, so the
    original drops it from EVERY tier rather than picking one. Done here in a
    single pass over the union instead of the notebook's three near-identical
    passes, which is the same result.
    """
    counts: dict = {}
    for seqs in per_tier.values():
        for key in seqs:
            if " " in key:
                counts[key] = counts.get(key, 0) + 1
    conflicted = {k for k, v in counts.items() if v > 1}
    for seqs in per_tier.values():
        for key in conflicted:
            seqs.pop(key, None)
    return len(conflicted)


def run_interactors(args) -> None:
    per_tier_complexes, per_tier_seqs = {}, {}
    for tier in args.tiers:
        c, s = write_fasta(tier, args.clinvar_dir, args.biogrid_dir,
                           args.output_dir, args.fasta_suffix)
        per_tier_complexes[tier], per_tier_seqs[tier] = c, s
        vts = {k for k in s if " " in k}
        with open(os.path.join(args.output_dir,
                               f"clinvar_{tier}_uniprot_vts.pkl"), "wb") as fh:
            pickle.dump(vts, fh)

    merged = {}
    for tier in args.tiers:
        merged.update(per_tier_seqs[tier])
    with open(os.path.join(args.output_dir,
                           "dirbind_with_vus_id_to_seq.pkl"), "wb") as fh:
        pickle.dump(merged, fh)
    print(f"merged id_to_seq: {len(merged)}", flush=True)

    n_conflict = drop_cross_tier_variants(per_tier_seqs)
    print(f"dropped {n_conflict} variants classified in >1 tier", flush=True)

    all_complexes = set().union(*per_tier_complexes.values())
    non_duplicate = clean_complexes(all_complexes)
    print(f"complexes: {len(all_complexes)} -> {len(non_duplicate)} after "
          f"collapsing flipped duplicates", flush=True)

    for tier in args.tiers:
        triplets = build_variant_triplets(per_tier_seqs[tier], non_duplicate)
        out = os.path.join(args.output_dir, f"{tier}_dirbind_variant_subset.pkl")
        with open(out, "wb") as fh:
            pickle.dump(triplets, fh)
        print(f"{tier}: {len(triplets)} (uniprot, variant, partner) triplets", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("variants", "interactors"), required=True)
    ap.add_argument("--tiers", nargs="+", default=list(TIERS), choices=TIERS)
    ap.add_argument("--output-dir", required=True)
    # stage: variants
    ap.add_argument("--variant-summary", help="ClinVar variant_summary.txt")
    ap.add_argument("--hgnc", help="hgnc_complete_set.txt")
    ap.add_argument("--uniprot-fasta-dir", default=".",
                    help="holds uniprot_all_{tier}s.fasta")
    # stage: interactors
    ap.add_argument("--clinvar-dir", help="holds {tier}{--fasta-suffix}")
    ap.add_argument("--biogrid-dir", help="holds the BioGRID pickles")
    ap.add_argument("--fasta-suffix", default="_wts_and_mts_04_25.fasta",
                    help="suffix of the per-tier WT+MT FASTA; the published run "
                         "used the April 2025 snapshot, hence the default")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.stage == "variants":
        missing = [f for f in ("variant_summary", "hgnc") if not getattr(args, f)]
        if missing:
            ap.error(f"--stage variants requires --{' --'.join(missing)}")
        run_variants(args)
    else:
        missing = [f for f in ("clinvar_dir", "biogrid_dir") if not getattr(args, f)]
        if missing:
            ap.error("--stage interactors requires --"
                     + " --".join(m.replace("_", "-") for m in missing))
        run_interactors(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
