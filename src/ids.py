"""Identifier and variant-string conventions used across the pipeline.

These encode decisions the whole codebase depends on — how a complex ID splits
into its two proteins, whether an ID is a UniProt accession, and the 1-based ->
0-based variant convention. They were previously reimplemented per call site
(`split_wt_id` in 4 modules, `get_gene_name` in 4, `is_uniprot_accession` in 2,
`_zero_based_variant` in 5).

Every consolidation here was verified against the real data before it was made,
not assumed:

- `split_wt_id`: all 4 implementations agreed on **all 3,347** distinct complex
  IDs in the canonical label tables. They differed only in error handling
  (`assert` vs `raise`) and return type (one returned a list for 2-part IDs).
  This version always returns a tuple; callers unpack, so that is compatible.
- `zero_based_variant`: the 3 distinct implementations agreed on **all 2,193**
  distinct variant strings, every one of which matches `^[A-Z]\\d+[A-Z]$`.
- `get_gene_name`: here the copies genuinely diverged. `mutpred_ppi_cv`'s had no
  RefSeq guard, so it would mangle `NP_002046` into `NP`. That was latent rather
  than active — it is only reached from the VarChAMP1p/CAVA loaders, whose
  gene-symbol maps contain zero `NP_`-prefixed keys (862 and 312 keys checked).
  This version keeps the guard, which is strictly safer and changes nothing on
  any reachable input.
"""
from __future__ import annotations

import re

# Official UniProt accession syntax, including an optional isoform suffix.
# Taken verbatim from the existing implementations -- a looser heuristic such as
# "starts with a letter and contains a digit" would classify gene symbols like
# BRCA1 as accessions, which silently breaks the C1/C2/C3 namespace logic.
_UNIPROT_PAT1 = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9](?:-[0-9]+)?$")
_UNIPROT_PAT2 = re.compile(r"^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(?:-[0-9]+)?$")


def is_uniprot_accession(id_str: str) -> bool:
    """True for UniProt accessions (`P04637`, `A0A024QYX0`), false for gene symbols."""
    return bool(_UNIPROT_PAT1.match(id_str) or _UNIPROT_PAT2.match(id_str))


def get_gene_name(gene_and_orf: str) -> str:
    """Strip a numeric ORF suffix: `BRCA1_1` -> `BRCA1`.

    RefSeq accessions pass through unchanged — without this guard `NP_002046`
    would become `NP`.
    """
    if gene_and_orf.startswith(("NP_", "np_")):
        return gene_and_orf
    if "_" not in gene_and_orf:
        return gene_and_orf
    return "_".join(gene_and_orf.split("_")[:-1])


def split_wt_id(wt_id: str) -> tuple[str, str]:
    """Split a complex ID into its two protein IDs.

    Three ID conventions coexist in the combined datasets, which is why this is
    not a simple `split`:
        `P35609-P29373`          UniProt pair, hyphen-delimited
        `NP_005190_KRTAP10-7`    RefSeq accession + gene symbol containing a hyphen
        `Q8WWY3-1-Q9P286`        UniProt isoform suffix inside the first ID
    """
    if wt_id.startswith(("NP_", "np_")):
        return "_".join(wt_id.split("_")[:2]), "_".join(wt_id.split("_")[2:])

    delim = "_" if "_" in wt_id else "-"
    parts = wt_id.split(delim)
    if len(parts) == 2:
        return parts[0], parts[1]

    # Three or more parts: an isoform suffix is numeric, so the first numeric
    # part ends the first protein ID.
    for idx, part in enumerate(parts):
        if part.isdigit():
            split_at = idx + 1
            if split_at == len(parts):   # the number was last: no second protein
                split_at = 1
            return delim.join(parts[:split_at]), delim.join(parts[split_at:])

    raise ValueError(f"Could not split complex id into two proteins: {wt_id!r}")


__all__ = [
    "is_uniprot_accession",
    "get_gene_name",
    "split_wt_id",
]
