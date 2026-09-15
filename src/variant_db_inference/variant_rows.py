#!/usr/bin/env python
"""Row source for the variant databases: one triplet, one prediction.

A triplet `(interactor, partner, mutation)` fully determines which contact graph
to load AND its orientation. That is the whole design. Nothing here infers
identity or orientation from a filename, which is what the previous file-driven
loop did -- it globbed `{db}/af3_graphs/*.mat`, split the stem on `_`, and called
the first token the interactor. In the variant-DB trees 121 clinvar / 121 cosmic
/ 116 gnomad / 29 hgmd files store their chains in the opposite order to their
own name, so that loop scored the partner chain for the equal-length ones.

Both chains of a complex can legitimately carry variants. Those are simply two
rows, each loading the same stored graph in its own orientation; the old loop
needed a reciprocal `A_B.mat` / `B_A.mat` pair on disk to express that.

WHERE ROWS COME FROM, and why not from the triplet tables
---------------------------------------------------------
Rows are `{variants in the DB's wt_and_vt FASTA} x {pairs present in the contact
graph store}`. The materialised triplet tables
(`*_dirbind_variant_subset.pkl`, `all_variants.pkl`) are used for LABELS only:
measured against the scored H5 entries, ~7% of currently published clinvar rows
are absent from those tables, and 94% of those misses are a missing *pair*, not
a missing variant. Enumerating from the tables would silently drop them.

BASE CONVENTION -- the sharpest hazard in this pipeline
-------------------------------------------------------
The two halves of the data disagree, and nothing documented it:

    1-based : triplet tables, `id_to_seq.pkl` keys   (verified 3000/3000)
    0-based : FASTA headers, ProtT5 keys, subgraph H5 variant keys (3000/3000)

Conversion happens ONLY in `utils.mutations` (`to_one_based` / `to_zero_based`),
re-exported here for the callers that already import them. Never an inline +1.
"""
from __future__ import annotations

import pickle
import re
import sys
from collections import Counter
from pathlib import Path

_PUB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PUB / "src"))
from paths import DATA_ROOT  # noqa: E402
from contact_graphs import ContactGraphStore, sha  # noqa: E402
from utils.mutations import MUTATION_RE as _MUT_RE  # noqa: E402
from utils.mutations import to_one_based, to_zero_based  # noqa: E402
from utils.sequences import accession_only, read_fasta  # noqa: E402

_ROOT = DATA_ROOT
_TABLE_DIR = _PUB / "datasets" / "variant_dbs"
_ALIASES = _TABLE_DIR / "aliases.csv"

def _fasta(db: str) -> Path:
    """One database's WT+variant FASTA, preferring the deposited copy.

    Two locations, in order:

      1. `datasets/variant_dbs/<db>_interaction_loss_wt_and_vt.fasta.gz` -- the
         Zenodo copy, for the four repositories whose sequences may be
         redistributed. Sequences are needed to iterate the row tables at all,
         so without this a reader working from the deposit could not score a
         single variant, no matter which caches they had.
      2. `$MUTPRED_DATA_ROOT/<db>/<db>_interaction_loss_wt_and_vt.fasta` -- the
         uncompressed working copy this pipeline writes when it builds a
         repository from source. Licensed repositories (COSMIC, HGMD) only ever
         have this one.

    `utils.sequences.iter_fasta` picks its opener from the suffix, so the
    gzipped and plain forms are interchangeable here.
    """
    deposited = _TABLE_DIR / f"{db}_interaction_loss_wt_and_vt.fasta.gz"
    if deposited.exists():
        return deposited
    return _ROOT / db / f"{db}_interaction_loss_wt_and_vt.fasta"


# Per database: the WT+variant FASTA (0-based headers) and the sequence map.
DB_SOURCES = {
    "clinvar": {"fasta": _fasta("clinvar"),
                "id_to_seq": _ROOT / "clinvar" / "id_to_seq.pkl"},
    "cosmic":  {"fasta": _fasta("cosmic")},
    "gnomad":  {"fasta": _fasta("gnomad"),
                "id_to_seq": _ROOT / "gnomad" / "id_to_seq.pkl"},
    "hgmd":    {"fasta": _fasta("hgmd"),
                "id_to_seq": _ROOT / "hgmd" / "id_to_seq.pkl"},
    "neurodev": {"fasta": _fasta("neurodev"),
                 "id_to_seq": _ROOT / "neurodev" / "id_to_seq.pkl"},
    "asd":      {"fasta": _fasta("asd"),
                 "id_to_seq": _ROOT / "asd" / "id_to_seq.pkl"},
}

# Historical names, still used by callers, docs and other modules.
#
# `neurodev` is the NeuroDev case/control cohort and is the canonical name; it
# was briefly spelled `autism`, which is wrong twice over -- the cohort is
# neurodevelopmental disorder broadly, and the name collided with the separate
# Fu et al. de novo autism set. That second dataset is `asd`, previously spelled
# `tulika_autism` / `fu_autism` after the person who assembled it. The two are
# different cohorts: `neurodev` carries case/control labels across four
# disorders, `asd` is de novo ASD cases only and is unlabelled.
#
# Resolve rather than reject: a caller passing a real dataset under an older
# spelling is not an error.
DB_ALIASES = {
    "autism": "neurodev",
    "tulika_autism": "asd",
    "fu_autism": "asd",
}


def resolve_db(db: str) -> str:
    """Canonical database name for any of its historical spellings."""
    return DB_ALIASES.get(db, db)


_ACC_RE = re.compile(r"^[A-Za-z0-9]+(-\d+)?$")


# ── inputs ───────────────────────────────────────────────────────────────────

def load_wt_sequences(db: str) -> dict[str, list[str]]:
    """accession -> [candidate wild-type sequences], from the FASTA.

    The FASTA is the authoritative source because it is exactly what ProtT5
    embedded: a protein absent from it has no embedding, so a row naming it could
    not have been scored. It is also the only source cosmic has -- there is no
    `cosmic/id_to_seq.pkl`.

    Wild-type entries are the bare-accession headers (">P25054"); ">P25054 S305R"
    is a mutant, derivable from WT + mutation and deliberately not collected.
    `id_to_seq.pkl`, where it exists, is merged in only to fill gaps.

    WHY A LIST AND NOT ONE SEQUENCE
    -------------------------------
    Most accessions repeat their wild-type record many times over (clinvar: 4,244
    of 4,554), and the repeats are normally byte-identical. For 111 accessions
    across the four databases they are NOT: the records hold genuinely different
    proteins, because an upstream gene-symbol/RefSeq -> UniProt mapping collision
    filed two proteins under one accession. Q16236 and O60675 each claim both the
    same 156-aa and 605-aa sequences; P08779/P13647/P12035/P35908 form a keratin
    cluster sharing sequences pairwise.

    This used to collapse with `setdefault`, i.e. "keep whichever record the file
    happened to list first". That arbitrary pick is what dropped 22,078 published
    rows -- their mutation validates against the OTHER record. Returning every
    candidate lets `iter_rows` pin the right one per pair, from the graph.

    The real repair is upstream, in the mapping code; this is containment.
    """
    db = resolve_db(db)
    seqs = read_fasta(DB_SOURCES[db]["fasta"], accession_only, on_duplicate="all")

    pkl = DB_SOURCES[db].get("id_to_seq")
    if pkl and Path(pkl).exists():
        with open(pkl, "rb") as fh:
            raw = pickle.load(fh)
        for k, v in raw.items():
            # Only fills gaps. An accession the FASTA already names keeps exactly
            # the FASTA's candidates: id_to_seq.pkl disagrees with the FASTA for
            # 81 clinvar accessions, and the FASTA is what ProtT5 embedded.
            if _ACC_RE.match(str(k)) and str(k) not in seqs:
                seqs[str(k)] = [str(v)]
    return seqs


def load_fasta_variants(db: str) -> dict[str, set[str]]:
    """accession -> {mutation, 1-BASED}. FASTA headers are 0-based on disk."""
    db = resolve_db(db)
    out: dict[str, set[str]] = {}
    for header_key, _seq in _iter_variant_headers(db):
        acc, mut0 = header_key
        out.setdefault(acc, set()).add(to_one_based(mut0))
    return out


def _iter_variant_headers(db: str):
    """((accession, mutation_0b), sequence) for the FASTA's VARIANT records."""
    from utils.sequences import iter_fasta

    def variant_header(header: str):
        parts = header.split()
        if len(parts) != 2 or not _MUT_RE.match(parts[1]):
            return None
        return (parts[0], parts[1])

    return iter_fasta(DB_SOURCES[db]["fasta"], variant_header)


def db_pair_keys(db: str, aliases_csv: Path | str) -> set[str]:
    """The store keys whose graph came from THIS database's own `af3_graphs`.

    The store merges all six variant databases into one file, but the databases
    are separate experiments: clinvar holds 6,836 distinct pairs of the 19,978
    total. Crossing a clinvar interactor against every pair in the merged store
    enumerated 3.2x the published rows, so the DB partition is required, not
    cosmetic.
    """
    db = resolve_db(db)
    import csv
    want = f"/{db}/"
    keys = set()
    with open(aliases_csv) as fh:
        for r in csv.DictReader(fh):
            if want in r["source"]:
                keys.add(r["pair_key"])
    return keys


def pair_graph_keys(db: str, aliases_csv: Path | str) -> dict[frozenset, str]:
    """{accession_i, accession_p} -> the store key of THAT pair's graph.

    Needed to pin which sequence an accession contributed, when the FASTA offers
    more than one (see `load_wt_sequences`). Searching the store for any candidate
    combination that happens to have a graph is not good enough: the store holds
    graphs for other pairs built from the other candidate, so the search finds two
    hits and gives up. The alias row ties one accession PAIR to one graph, which
    resolves it.
    """
    db = resolve_db(db)
    import csv
    want = f"/{db}/"
    out: dict[frozenset, str] = {}
    with open(aliases_csv) as fh:
        for r in csv.DictReader(fh):
            if want not in r["source"]:
                continue
            a, _, b = r["complex_id"].partition("_")
            if b:
                out.setdefault(frozenset((a, b)), r["pair_key"])
    return out


def store_pairs(db: str, aliases_csv: Path | str) -> dict[str, set[str]]:
    """accession -> {partner accessions} that this database has a graph for.

    Taken from the alias table's `complex_id` (the old `.mat` stem), NOT by
    reverse-mapping the store's sequence hashes back to accessions. That reverse
    map is not a function: sequence -> accession is one-to-MANY, because
    identical-sequence proteins are the same content. P62081, P62082 and P62083
    are one ribosomal sequence under three accessions, so a naive
    `{sha(s): a for a, s in seqs.items()}` silently keeps only the last, and every
    published row naming one of the others is dropped. That cost 127k clinvar
    rows before it was caught.

    Using the stem here is NOT filename-derived orientation. It supplies only the
    set of accession pairs the database knows about; which of the two is the
    interactor comes from the row, and the graph itself is still fetched by
    sequence.
    """
    db = resolve_db(db)
    import csv
    want = f"/{db}/"
    pairs: dict[str, set[str]] = {}
    with open(aliases_csv) as fh:
        for r in csv.DictReader(fh):
            if want not in r["source"]:
                continue
            a, _, b = r["complex_id"].partition("_")
            if not b:
                continue
            pairs.setdefault(a, set()).add(b)
            pairs.setdefault(b, set()).add(a)
    return pairs


# ── the row source ───────────────────────────────────────────────────────────

def iter_rows(db: str, store: ContactGraphStore, *, seqs=None, variants=None,
              pairs=None, stats=None):
    """Yield dicts: interactor, partner, mutation (1-based), and both sequences.

    The mutation is asserted against the interactor's own sequence, so a row can
    never carry a position that disagrees with the protein it names.

    PINNING THE SEQUENCE, WHEN AN ACCESSION HAS MORE THAN ONE
    ---------------------------------------------------------
    `load_wt_sequences` returns every candidate for an accession, because for 111
    accessions the FASTA holds two different proteins under one name (see there).
    The candidate is chosen HERE, per pair, from the stored graph: the graph was
    built from two specific chain sequences, so whichever candidate combination
    has a graph is the one this pair actually used.

    When the graph cannot pin a single interactor sequence the row is DROPPED and
    counted in `stats`. It deliberately does not fall back to "whichever candidate
    validates the mutation" -- the candidates are usually different proteins, so
    that would silently file a variant under the wrong gene, which is worse than
    dropping it and much harder to notice.

    Pass a `collections.Counter` as `stats` to receive the drop reasons.
    """
    db = resolve_db(db)
    seqs = seqs if seqs is not None else load_wt_sequences(db)
    variants = variants if variants is not None else load_fasta_variants(db)
    pairs = pairs if pairs is not None else store_pairs(db, _ALIASES)
    if stats is None:
        stats = Counter()

    # Only built when an accession is actually ambiguous -- the overwhelming
    # majority of pairs have one candidate per side and never touch this.
    _graph_key: dict[frozenset, str] | None = None
    _key_shas: dict[str, set[str]] | None = None

    def _pin(acc_i: str, acc_p: str, cand_i: list[str], cand_p: list[str]):
        """(interactor_seq, partner_seq) pinned by that pair's graph, or None."""
        if len(cand_i) == 1 and len(cand_p) == 1:
            return cand_i[0], cand_p[0]          # nothing to disambiguate
        nonlocal _graph_key, _key_shas
        if _graph_key is None:
            _graph_key = pair_graph_keys(db, _ALIASES)
            _key_shas = {m["key"]: {m["seq_a_sha"], m["seq_b_sha"]}
                         for m in store.meta()}
        key = _graph_key.get(frozenset((acc_i, acc_p)))
        if key is None:
            return None
        shas = _key_shas.get(key)
        if not shas:
            return None
        hit_i = [s for s in cand_i if sha(s) in shas]
        hit_p = [s for s in cand_p if sha(s) in shas]
        # A homodimer of two versions puts both candidates in the same graph, so
        # require a unique interactor; the partner may stay ambiguous because the
        # mutation is only ever checked against the interactor.
        if len(hit_i) != 1 or not hit_p:
            return None
        return hit_i[0], hit_p[0]

    for interactor, muts in variants.items():
        partners = pairs.get(interactor)
        if not partners:
            stats["interactor has no pair in this DB"] += 1
            continue
        cand_i = seqs.get(interactor)
        if not cand_i:
            stats["interactor sequence unknown"] += 1
            continue
        for partner in sorted(partners):
            cand_p = seqs.get(partner)
            if not cand_p:
                stats["partner sequence unknown"] += 1
                continue
            pinned = _pin(interactor, partner, cand_i, cand_p)
            if pinned is None:
                stats["ambiguous: no single graph-pinned sequence"] += 1
                continue
            iseq, pseq = pinned
            for mut in sorted(muts):
                m = _MUT_RE.match(mut)
                wt_aa, pos1 = m.group(1), int(m.group(2))
                if pos1 < 1 or pos1 > len(iseq) or iseq[pos1 - 1] != wt_aa:
                    stats["mutation disagrees with the pinned sequence"] += 1
                    continue
                yield {"interactor": interactor, "partner": partner,
                       "mutation": mut, "interactor_sequence": iseq,
                       "partner_sequence": pseq}


# ── the emitted tables, read back ────────────────────────────────────────────

def table_path(db: str) -> Path:
    """`datasets/variant_dbs/{db}_rows.csv.gz`, written by build_variant_db_tables."""
    return _TABLE_DIR / f"{resolve_db(db)}_rows.csv.gz"


def iter_table_rows(db: str, path=None, *, seqs=None, stats=None):
    """Yield table rows (all columns, mutation 1-BASED) with both sequences added.

    The table stores `pair_key`, not the sequences: inlining a 1,000-residue
    string across gnomad's 10.5M rows is what the content address exists to
    avoid. The sequence is recovered by hashing this database's FASTA candidates,
    which is safe in the direction accession -> sequence is not -- sha ->
    sequence is injective, so a `pair_key` half names exactly one chain.

    Consumers get the same keys `iter_rows` yields, so either can be the row
    source; this one additionally carries the annotations and the BioGRID
    direct-binding filter the table was built with.
    """
    import csv
    import gzip

    seqs = seqs if seqs is not None else load_wt_sequences(db)
    if stats is None:
        stats = Counter()
    # Hashed once per candidate, not once per row.
    cand_shas = {a: [(sha(s), s) for s in ss] for a, ss in seqs.items()}
    by_sha = {h: s for v in cand_shas.values() for h, s in v}

    p = Path(path or table_path(db))
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            halves = r["pair_key"].split("_")
            hit = [s for h, s in cand_shas.get(r["interactor"], []) if h in halves]
            if len(hit) != 1:
                # The 111 conflated accessions again: two different proteins under
                # one name, and this row's key cannot say which chain is which.
                stats["pair_key does not pin one interactor sequence"] += 1
                continue
            iseq = hit[0]
            h_i = sha(iseq)
            pseq = by_sha.get(halves[0] if halves[1] == h_i else halves[1])
            if pseq is None:
                stats["partner sequence unknown"] += 1
                continue
            r["interactor_sequence"], r["partner_sequence"] = iseq, pseq
            yield r
