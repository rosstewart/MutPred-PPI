"""One FASTA reader, with the header policy made an argument rather than a guess.

Seven hand-rolled parsing loops and ~8 BioPython call sites existed before this.
They were NOT interchangeable, which is why this module takes a `header_parser`
instead of picking a default:

    prott5_loader / precompute_prott5   mangle `/` and `.` to `_` for H5 keys
    map_cosmic                          takes `header.split("|")[1]`
    00_make_af3_json_input              takes `header.split("|")[0]`
    variant_rows                        drops records whose header is not an accession

Merging those behind one implicit default would silently change WHICH SEQUENCES
LOAD in four pipelines. So the caller states its policy, and the shared part is
only the parsing loop.

The readers stream and never materialise the file: the variant-DB FASTAs run to
hundreds of MB.
"""
from __future__ import annotations

import gzip
from collections.abc import Callable, Iterator
from pathlib import Path

__all__ = [
    "iter_fasta", "read_fasta", "accession_only", "first_token",
    "pipe_field", "whole_header", "h5_safe_key",
]


# -- header policies ---------------------------------------------------------

def first_token(header: str) -> str | None:
    """`'P25054 S305R'` -> `'P25054'`. The first whitespace-delimited token."""
    parts = header.split()
    return parts[0] if parts else None


def accession_only(header: str) -> str | None:
    """The header if it is a bare accession, else None (record dropped).

    For FASTAs that mix wild-type records (`>P25054`) with variant records
    (`>P25054 S305R`), where only the wild types name a protein.
    """
    parts = header.split()
    return parts[0] if len(parts) == 1 else None


def pipe_field(n: int) -> Callable[[str], str | None]:
    """`sp|P12345|NAME_HUMAN` -> field `n`. UniProt-style headers."""
    def parser(header: str) -> str | None:
        parts = header.split("|")
        return parts[n] if len(parts) > n else None
    return parser


def whole_header(header: str) -> str | None:
    """The header unchanged (already stripped of its leading `>` by the reader).

    For WT_VT-format FASTAs, where the WHOLE header -- not just its first
    token -- distinguishes a wild-type record (`P25054`) from a variant record
    (`P25054 S305R`). `first_token` on either yields the same string, which is
    the right choice for accession lookups but the wrong one for a key that
    must keep the two apart.
    """
    return header if header else None


def h5_safe_key(header: str, *, whole: bool) -> str | None:
    """`/` and `.` replaced by `_`, for HDF5 dataset names. `None` if empty.

    HDF5 reads `/` as a group separator, so an unmangled header would create an
    unintended hierarchy. Preserved exactly, because the ProtT5 stores depend on
    this mangling to find their keys again.

    `whole` is REQUIRED, with no default, on purpose: `whole=True` mangles the
    WHOLE header (`whole_header`); `whole=False` mangles only its first token
    (`first_token`). Picking a default here would repeat the exact bug this
    signature exists to prevent -- `first_token` collapses `>P25054` and
    `>P25054 S305R` onto the same key `"P25054"`, silently merging every
    variant embedding into its wild type. The WT_VT ProtT5 writers need
    `whole=True` to keep the two apart, matching how the existing
    `prott5_embeddings.h5` stores were actually keyed; a caller reading a
    format where the header IS the accession (no variant suffix) can use
    `whole=False` safely. There is no default that is correct for both, so
    each call site must say which it means.
    """
    token = whole_header(header) if whole else first_token(header)
    return None if not token else token.replace("/", "_").replace(".", "_")


# -- reading -----------------------------------------------------------------

def iter_fasta(path, header_parser: Callable[[str], str | None] = first_token
               ) -> Iterator[tuple[str, str]]:
    """Yield `(key, sequence)`; records whose parser returns None are skipped.

    Handles `.gz` transparently. Case is preserved -- at least one caller encodes
    chain membership in it.
    """
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    key, buf = None, []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if key is not None and buf:
                    yield key, "".join(buf)
                key, buf = header_parser(line[1:].strip()), []
            elif key is not None:
                buf.append(line.strip())
    if key is not None and buf:
        yield key, "".join(buf)


def read_fasta(path, header_parser: Callable[[str], str | None] = first_token,
               *, on_duplicate: str = "first"):
    """`{key: sequence}`, or `{key: [sequences]}` when `on_duplicate="all"`.

    `on_duplicate` is explicit because it has bitten this project: the variant-DB
    FASTAs repeat most accessions, and for 111 of them the repeats hold DIFFERENT
    sequences (an upstream gene->UniProt collision). Silently keeping whichever
    came first dropped 22,078 rows once.

        "first"  keep the first occurrence   (the historical behaviour)
        "last"   keep the last
        "raise"  refuse to guess
        "all"    values are lists of every distinct sequence seen
    """
    out: dict = {}
    for key, seq in iter_fasta(path, header_parser):
        if on_duplicate == "all":
            v = out.setdefault(key, [])
            if seq not in v:
                v.append(seq)
        elif key not in out:
            out[key] = seq
        elif on_duplicate == "last":
            out[key] = seq
        elif on_duplicate == "raise" and out[key] != seq:
            raise ValueError(f"{path}: {key!r} appears twice with different sequences")
    return out
