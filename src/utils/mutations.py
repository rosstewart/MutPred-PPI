"""Mutation strings: parse once, convert once, apply once.

Before this module the repo held **~30 inline `±1` conversions and 5 separate
regexes**, and they did not agree. That is not a style problem: the copies
disagreed about out-of-range positions, about which alphabet is legal, and about
what to do when the wild-type residue does not match the sequence. Consolidating
them therefore CHANGES BEHAVIOUR at some call sites, and each such change is
meant to be measured and reported rather than absorbed silently.

BASE CONVENTION
---------------
A mutation string is `{WT}{POSITION}{MT}`, e.g. `A123V`. Positions are **1-BASED**
everywhere a row is represented -- the canonical row tables, the triplet tables,
HGVS, ClinVar, every annotation file. 0-based exists only as a Python index, and
this module is the one place the two meet.

    to_zero_based / to_one_based   convert between the two STRING forms
    parse(...) -> (wt, pos1, mt)   pos1 is 1-based
    index(...) -> int              the 0-based array index, named so it is obvious

POLICY DECISIONS, and why
-------------------------
* **Alphabet: `[A-Za-z]`, normalised to upper case.** The strictest previous
  regex required `[A-Z]` and would reject a lowercase record outright;
  `ids.parse_mutation` accepted anything. Rejecting on case would drop real rows
  for a cosmetic reason, so accept and normalise.
* **Out-of-range positions raise.** `pos < 1` in a 1-based string is not a
  position, it is a bug upstream. Several previous copies computed `pos - 1 = -1`
  and silently indexed from the END of the sequence, which is how a variant could
  be applied to the wrong residue and never be noticed.
* **`apply()` returns None on a wild-type mismatch, and the caller must count
  it.** The four previous implementations returned the sequence unchanged, did no
  check at all, asserted, or silently skipped. `None` forces the caller to
  decide, and a counted skip is the house rule.
"""
from __future__ import annotations

import re

__all__ = [
    "MUTATION_RE", "parse", "index", "position", "to_one_based", "to_zero_based",
    "apply", "is_mutation",
]

# Accepts lowercase and normalises; see the policy note above.
MUTATION_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def is_mutation(text: str) -> bool:
    """True if `text` is a well-formed mutation string."""
    return bool(MUTATION_RE.match(str(text)))


def parse(mutation: str) -> tuple[str, int, str]:
    """`'A123V'` -> `('A', 123, 'V')`, position **1-BASED**, residues upper-cased."""
    m = MUTATION_RE.match(str(mutation))
    if not m:
        raise ValueError(f"unparseable mutation {mutation!r}")
    pos = int(m.group(2))
    if pos < 1:
        raise ValueError(f"{mutation!r} has a non-positive 1-based position")
    return m.group(1).upper(), pos, m.group(3).upper()


def index(mutation: str) -> int:
    """The 0-BASED array index for a 1-based mutation string.

    Named rather than written as `pos - 1` at the call site, so every place the
    conversion happens is greppable.
    """
    return parse(mutation)[1] - 1


def position(mutation: str) -> int:
    """The raw integer in the string, with NO base assumed.

    For the call sites that hold an already-0-based string (ProtT5 keys, subgraph
    H5 variant keys, the inference sidecars) and want it as an array index. Using
    `index()` there would subtract one from a number that is already an index.
    The name says "I am not converting", which is the point -- an inline
    `int(v[1:-1])` said nothing at all.
    """
    m = MUTATION_RE.match(str(mutation))
    if not m:
        raise ValueError(f"unparseable mutation {mutation!r}")
    return int(m.group(2))


def to_zero_based(mutation_1b: str) -> str:
    """`'A123V'` (1-based) -> `'A122V'` (0-based), for ProtT5 / H5 keys."""
    wt, pos, mt = parse(mutation_1b)
    return f"{wt}{pos - 1}{mt}"


def to_one_based(mutation_0b: str) -> str:
    """`'A122V'` (0-based) -> `'A123V'` (1-based), the canonical row form.

    Accepts position 0, which is legal in a 0-based string; `parse` is not used
    here because it would reject it.
    """
    m = MUTATION_RE.match(str(mutation_0b))
    if not m:
        raise ValueError(f"unparseable mutation {mutation_0b!r}")
    return f"{m.group(1).upper()}{int(m.group(2)) + 1}{m.group(3).upper()}"


def apply(sequence: str, mutation: str) -> str | None:
    """Sequence with `mutation` (1-BASED) applied, or None if it does not fit.

    None means one of: the position lies outside the sequence, or the wild-type
    residue disagrees with what is there. The caller must count the None -- a
    silently unmutated sequence is indistinguishable from a wild-type one, and
    that is exactly the failure this returns None to prevent.
    """
    wt, pos, mt = parse(mutation)
    i = pos - 1
    if i >= len(sequence) or sequence[i] != wt:
        return None
    return sequence[:i] + mt + sequence[i + 1:]
