"""Composite identifiers: two kinds, two delimiters, one place they are handled.

Splitting a composite was never the problem in this repo -- an AMBIGUOUS
delimiter was. Both delimiters below were chosen because they cannot occur inside
the things they join:

    interactor + partner   ->  `__`     `O14787-2__Q13207`
    protein    + variant   ->  ` `      `P25054 S305R`

Why not the alternatives:

    `_`   breaks on RefSeq ids            NP_002046_GFAP
    `-`   breaks on UniProt isoforms      O14787-2
    `:`   already means something else in the inference pipeline

With `__` and a space, `split` is exact rather than a heuristic, so both
directions are public and total. That is the whole point: composites are fine,
guessing where they divide is not.

`split_legacy_pair` exists only for artifacts written before this convention --
`.mat` stems, the old `cv_reference` ids, `complex_id` columns in published TSVs.
It is a heuristic by necessity and dies when those artifacts are archived.
"""
from __future__ import annotations

__all__ = [
    "PAIR_SEP", "VARIANT_SEP",
    "pair_id", "split_pair_id", "is_pair_id",
    "variant_id", "split_variant_id", "is_variant_id",
    "bare_accession", "split_legacy_pair",
]

PAIR_SEP = "__"
VARIANT_SEP = " "


# ── protein + protein ────────────────────────────────────────────────────────

def pair_id(interactor: str, partner: str) -> str:
    """`('A', 'B')` -> `'A__B'`. Order is preserved: this is not a sorted key.

    For the ORDER-INDEPENDENT content address of a pair, use
    `contact_graphs.pair_key`, which hashes the two sequences. This function
    names a pair for humans and filenames; that one identifies a graph.
    """
    for part in (interactor, partner):
        if PAIR_SEP in part:
            raise ValueError(f"{part!r} contains the pair separator {PAIR_SEP!r}")
    return f"{interactor}{PAIR_SEP}{partner}"


def split_pair_id(pair: str) -> tuple[str, str]:
    """`'A__B'` -> `('A', 'B')`. Exact, because neither side can contain `__`."""
    parts = str(pair).split(PAIR_SEP)
    if len(parts) != 2:
        raise ValueError(f"{pair!r} is not a `{PAIR_SEP}`-delimited pair id")
    return parts[0], parts[1]


def is_pair_id(text: str) -> bool:
    return str(text).count(PAIR_SEP) == 1


# ── protein + variant ────────────────────────────────────────────────────────

def variant_id(accession: str, mutation: str) -> str:
    """`('P25054', 'S305R')` -> `'P25054 S305R'`.

    Whitespace-delimited because neither an accession nor a mutation string can
    contain whitespace. This is already the on-disk convention for ProtT5 keys,
    FASTA headers and every annotation dict, so it is adopted rather than
    replaced.
    """
    for part in (accession, mutation):
        if any(c.isspace() for c in str(part)):
            raise ValueError(f"{part!r} contains whitespace")
    return f"{accession}{VARIANT_SEP}{mutation}"


def split_variant_id(text: str) -> tuple[str, str]:
    """`'P25054 S305R'` -> `('P25054', 'S305R')`."""
    parts = str(text).split()
    if len(parts) != 2:
        raise ValueError(f"{text!r} is not an `accession mutation` pair")
    return parts[0], parts[1]


def is_variant_id(text: str) -> bool:
    return len(str(text).split()) == 2


# ── accessions ───────────────────────────────────────────────────────────────

def bare_accession(accession: str) -> str:
    """`'O14787-2'` -> `'O14787'`. The parent of an isoform.

    Explicit and named because an inline `split("-")[0]` is how five analysis
    scripts silently merged isoforms into their parents. Call this only when the
    parent is genuinely what is wanted -- gene-level grouping, or a lookup in a
    per-accession database that has no isoform entries.
    """
    return str(accession).split("-")[0]


# ── legacy ───────────────────────────────────────────────────────────────────

def split_legacy_pair(wt_id: str) -> tuple[str, str]:
    """Best-effort split of a pre-`__` composite. Heuristic; do not use for new data.

    Three conventions coexist in artifacts written before `PAIR_SEP`:

        P35609-P29373          UniProt pair, hyphen-delimited
        NP_005190_KRTAP10-7    RefSeq accession + gene symbol containing a hyphen
        Q8WWY3-1-Q9P286        UniProt isoform suffix inside the first id

    Moved verbatim from `ids.split_wt_id`, which is the best existing
    implementation. It is retained ONLY to read archived artifacts and should be
    deleted with them.
    """
    if wt_id.startswith(("NP_", "np_")):
        return "_".join(wt_id.split("_")[:2]), "_".join(wt_id.split("_")[2:])

    delim = "_" if "_" in wt_id else "-"
    parts = wt_id.split(delim)
    if len(parts) == 2:
        return parts[0], parts[1]

    # Three or more parts: an isoform suffix is numeric, so the first numeric
    # part ends the first protein id.
    for idx, part in enumerate(parts):
        if part.isdigit():
            split_at = idx + 1
            break
    else:
        split_at = 1
    return delim.join(parts[:split_at]), delim.join(parts[split_at:])
