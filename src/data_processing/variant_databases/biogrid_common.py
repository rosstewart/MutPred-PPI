"""Helpers shared verbatim by the variant-database mappers.

Only functions that were **byte-identical** across `map_*.py` live here.  The
other repeated names (`build_complex_subset`, `apply_variants`, `fetch_hgnc_info`,
`load_refseq_to_uniprot`, `sanity_check`, ...) differ per database for real
reasons -- different ID namespaces, different variant notation, different
sanity thresholds -- and are deliberately left in place rather than merged
behind flags.

The point of centralising these four is that they encode conventions the whole
variant-DB pipeline depends on: which BioGRID pickles are authoritative, and
that an interaction is an unordered pair.  Those must not drift between
databases.
"""
from __future__ import annotations

import pickle


def load_biogrid(biogrid_dir):
    """Load the direct-binding BioGRID interactome and its sequence map."""
    with open(f"{biogrid_dir}/biogrid_dirbind_uniprot_to_interactors.pkl", "rb") as f:
        uniprot_to_interactors = pickle.load(f)
    with open(f"{biogrid_dir}/uniprot_dirbind_to_seq.pkl", "rb") as f:
        uniprot_to_seq = pickle.load(f)
    return uniprot_to_interactors, uniprot_to_seq


def clean_complexes(all_complexes):
    """Collapse (a, b) / (b, a) to one orientation — interactions are unordered."""
    cleaned = set()
    for a, b in all_complexes:
        if (b, a) not in cleaned:
            cleaned.add((a, b))
    return cleaned


def get_complexes_in_biogrid(uniprot_wts, uniprot_to_interactors, uniprot_to_seq):
    """All (wt, partner) pairs for which BioGRID has a partner with a sequence."""
    wt_complexes = set()
    for uid in uniprot_wts:
        if uid not in uniprot_to_interactors:
            continue
        for partner in uniprot_to_interactors[uid]:
            if partner in uniprot_to_seq:
                wt_complexes.add((uid, partner))
    return wt_complexes


def build_variant_triplets(id_to_seq, complexes):
    """Expand '<uniprot> <variant>' keys into (uniprot, variant, partner) triplets.

    Indexes the complexes by protein first. The original rescanned every complex
    for every variant, which is O(variants x complexes) -- fine for the smaller
    databases but effectively non-terminating for ClinVar's millions of variants
    against tens of thousands of complexes. The output is unchanged: a protein's
    partner set is exactly `{p for (uid, p)} | {u for (u, uid)}`, which is what
    the two `if` branches collected.
    """
    partners_of: dict = {}
    for u, p in complexes:
        partners_of.setdefault(u, set()).add(p)
        partners_of.setdefault(p, set()).add(u)

    variants = set()
    for key in id_to_seq:
        if " " not in key:
            continue
        uid, variant = key.split(" ", 1)
        for partner in partners_of.get(uid, ()):
            variants.add((uid, variant, partner))
    return variants
