#!/usr/bin/env python
"""Rename every AF3 structure into the one UniProt namespace, by content.

Both structure trees currently use the naming AF3 happened to emit:

    fold_a0a024qyx0_o43889_model_0.pdb.gz     training / evaluation
    fold_o00499_o00401_model_0.cif.gz         variant databases

which is lower-cased, `fold_`-prefixed, `_model_0`-suffixed, and joins the pair
with `_`. That is unparseable in the presence of isoform accessions, loses the
UniProt case, and -- worst -- is not always true: in the variant-DB trees 121
clinvar / 121 cosmic / 116 gnomad / 29 hgmd files store the chains in the
opposite order to the one their own filename claims, so any consumer that reads
the interactor off the filename scores the wrong chain.

The new name is derived from the structure's CONTENTS, never from its old name:
chain sequences are read out of the file and mapped back to accessions through
the canonical tables and the variant-DB FASTAs.

    {ACC_LO}__{ACC_HI}.{ext}          e.g.  O14787-2__Q13207.cif.gz

`__` separates the pair because no accession in the namespace contains an
underscore, so the name stays unambiguous even for isoforms and for RefSeq-style
ids. Accessions are sorted, so one structure gets one name and the filename
encodes no orientation -- orientation is a property of a row, resolved from
sequences at load time, and must never be inferred from a filename again.

Training/eval and variant-DB structures go to separate output directories but
share this one naming scheme and this one implementation.

Usage:
    python src/data_processing/canonicalize_structures.py \\
        --structures datasets/af3_structures --out datasets/af3_structures_canonical
    python src/data_processing/canonicalize_structures.py \\
        --structures datasets/af3_structures_variant_dbs/clinvar \\
        --out datasets/af3_structures_variant_dbs_canonical/clinvar

Promotion is by HARDLINK. The usual chain builds into a temporary directory and
then promotes with `cp -alf`, which means the build directory and the live tree
become the same inodes. That build directory is therefore an ALIAS, not a
snapshot: deleting it frees nothing, and it would not preserve anything if the
live tree were damaged. Do not leave it behind under a name that reads as a
backup.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import io
import os
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from Bio.PDB import MMCIFParser, PDBParser
from joblib import Parallel, delayed

from paths import DATA_ROOT  # noqa: E402
from contact_graphs import _is_polymer_residue, residue_to_one, sha
from utils.gcv_common import DATASET_CONFIGS, load_data
from utils.sequences import iter_fasta

_ROOT = DATA_ROOT
_VDB_FASTAS = [
    _ROOT / "clinvar" / "clinvar_interaction_loss_wt_and_vt.fasta",
    _ROOT / "clinvar" / "clinvar_benign_wt_vt_partners.fasta",
    _ROOT / "gnomad" / "gnomad_interaction_loss_wt_and_vt.fasta",
    _ROOT / "gnomad" / "gnomad_8p_partner_wt_and_vt.fasta",
    _ROOT / "hgmd" / "hgmd_interaction_loss_wt_and_vt.fasta",
    _ROOT / "cosmic" / "cosmic_interaction_loss_wt_and_vt.fasta",
    _ROOT / "neurodev" / "neurodev_interaction_loss_wt_and_vt.fasta",
    _ROOT / "asd" / "asd_interaction_loss_wt_and_vt.fasta",
]




def _open(path: str):
    return (gzip.open(path, "rt") if path.endswith(".gz") else open(path))


# Sequences that appear in the canonical row tables. A structure is only usable
# for a row if its chains ARE those sequences, so this decides tie-breaks below.
_TABLE_SEQS: set[str] = set()


def build_seq_map() -> dict[str, str]:
    """sequence -> accession, from the canonical tables plus the variant-DB FASTAs."""
    m: dict[str, str] = {}
    for name in DATASET_CONFIGS:
        df = load_data(DATASET_CONFIGS[name])
        for a, s in zip(df["interactor"], df["interactor_sequence"]):
            m.setdefault(s, str(a))
        for b, s in zip(df["partner"], df["partner_sequence"]):
            m.setdefault(s, str(b))
    n_tab = len(m)
    _TABLE_SEQS.update(m)          # tables only, before the FASTA top-up
    for fa in _VDB_FASTAS:
        if not fa.exists():
            continue
        for acc, seq in iter_fasta(fa, _wt_accession_only):
            m.setdefault(seq, acc)
    print(f"  {n_tab} sequences from canonical tables, {len(m) - n_tab} more from FASTAs",
          flush=True)
    return m


_WT_ACC_RE = re.compile(r"[A-Za-z0-9]+(-\d+)?")


def _wt_accession_only(header: str) -> str | None:
    """Bare accession only, `None` otherwise -- so a variant header is dropped.

    Wild-type headers are a bare accession (`P25054`); variant headers append
    the substitution (`P25054 S305R`). Only the wild type names a protein --
    taking the first token for both would key ~300k mutant sequences to their
    parent accession. Stricter than `utils.sequences.accession_only`: the
    single token must also look like an accession, since this function's
    output becomes the accession half of a sequence->accession map used for
    name resolution, where a malformed single-token header must not be
    treated as real.
    """
    parts = header.split()
    if len(parts) == 1 and _WT_ACC_RE.fullmatch(parts[0]):
        return parts[0]
    return None


def chain_seqs(path: str):
    """[(chain_id, sequence)] using the canonical MAP residue policy.

    Previously filtered on `r.id[0] == " "` (excludes HETATM-flagged residues
    like MSE, but not UNK, which is a fourth distinct policy from the other two
    graph builders) and discarded the WHOLE FILE on any unmapped residue. Now
    shares `contact_graphs.residue_to_one` / `_is_polymer_residue`: a modified
    residue maps to its parent letter, anything unrecognised (including UNK)
    maps to "X", and nothing is ever dropped -- so accession lookup below can
    no longer fail silently over a single non-standard residue.
    """
    txt = _open(path).read()
    parser = MMCIFParser(QUIET=True) if ".cif" in path else PDBParser(QUIET=True)
    st = parser.get_structure("x", io.StringIO(txt))
    out = []
    for ch in st[0]:
        seq = "".join(residue_to_one(r.get_resname()) for r in ch if _is_polymer_residue(r))
        if seq:
            out.append((ch.id, seq))
    return out


def _ext(p: str) -> str:
    for e in (".cif.gz", ".pdb.gz", ".cif", ".pdb"):
        if p.endswith(e):
            return e
    return Path(p).suffix


def _old_accessions(stem: str):
    """The pair the OLD filename claims, for contradiction reporting only."""
    s = re.sub(r"^fold_", "", stem)
    s = re.sub(r"_model(_\d+)?$", "", s)
    parts = s.split("_")
    return (parts[0], "_".join(parts[1:])) if len(parts) >= 2 else (None, None)


def _mean_plddt(path: str) -> float:
    """Mean pLDDT (the B-factor column) for a model. -1.0 when unreadable.

    Format-aware, deliberately. PDB is fixed-column, mmCIF is whitespace-
    delimited with a named `_atom_site` loop whose column ORDER varies between
    AF3 output variants -- so the mmCIF branch locates `B_iso_or_equiv` by name.
    Reading mmCIF with the PDB column offsets (as this did originally) returned
    -1.0 for 983 of 3,832 CIFs, which then lost every tie-break to a PDB. Since
    the rule is "higher pLDDT wins, .cif breaks a tie", that silently inverted it.
    """
    tot = n = 0.0
    try:
        if ".cif" in path:
            b_idx = None
            in_loop = False
            cols = 0
            for line in _open(path):
                s = line.strip()
                if s.startswith("_atom_site."):
                    in_loop = True
                    if s == "_atom_site.B_iso_or_equiv":
                        b_idx = cols
                    cols += 1
                    continue
                if in_loop and s.startswith(("ATOM", "HETATM")):
                    if b_idx is None:
                        return -1.0
                    f = s.split()
                    if len(f) > b_idx:
                        try:
                            tot += float(f[b_idx]); n += 1
                        except ValueError:
                            pass
                elif in_loop and n and not s.startswith(("ATOM", "HETATM")):
                    break            # end of the atom_site loop
        else:
            for line in _open(path):
                if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
                    try:
                        tot += float(line[60:66]); n += 1
                    except ValueError:
                        pass
    except Exception:
        return -1.0
    return tot / n if n else -1.0


def write_as_cif_gz(src: str, dest: Path) -> str | None:
    """Write `src` as gzipped mmCIF at `dest`. Returns an error string, or None.

    One format for the whole directory, because a pair keeping both a `.cif` and
    a `.pdb` is how two methods ended up scoring different structures for the
    same input.

    The conversion is GATED: a silently altered structure would be worse than the
    duplication it replaces, so the written file is re-read and its atom count
    and coordinates compared against the source. Anything over 1e-3 A is refused.
    """
    import gzip
    import shutil
    import gemmi

    try:
        if src.endswith(".cif.gz"):
            shutil.copy2(src, dest)          # already the target format
            return None

        st = gemmi.read_structure(src)
        st.setup_entities()
        before = [(a.pos.x, a.pos.y, a.pos.z)
                  for m in st for ch in m for r in ch for a in r]

        # Hidden sibling, NOT a ".tmp" suffix: gemmi infers format from the
        # extension, so the temp file must still end in .cif.gz.
        tmp = dest.with_name("." + dest.name)
        with gzip.open(tmp, "wt") as fh:
            fh.write(st.make_mmcif_document().as_string())

        # gemmi reads .cif.gz directly, so verify by re-reading the written file
        # rather than round-tripping through a string.
        back = gemmi.read_structure(str(tmp))
        back.setup_entities()
        after = [(a.pos.x, a.pos.y, a.pos.z)
                 for m in back for ch in m for r in ch for a in r]

        if len(before) != len(after):
            tmp.unlink(missing_ok=True)
            return f"atom count changed {len(before)} -> {len(after)}"
        worst = max((max(abs(p - q) for p, q in zip(b, a))
                     for b, a in zip(before, after)), default=0.0)
        if worst > 1e-3:
            tmp.unlink(missing_ok=True)
            return f"max coordinate delta {worst:.4f} A exceeds 1e-3"
        tmp.replace(dest)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _one(f: str):
    try:
        return f, chain_seqs(f), _mean_plddt(f)
    except Exception:
        return f, [], -1.0


def _resolve_with_unk(seq: str, seq2acc: dict, by_len: dict) -> str | None:
    """Accession for `seq`, tolerating positions the structure could not model.

    AlphaFold3 writes a residue it cannot build as `UNK`, which maps to "X".
    Selenoproteins are the common case: TXNRD1/TXNRD2/GPX1 carry a Sec (`U`)
    that AF3 emits as UNK, so the structure reads `...AGCXG` where the canonical
    sequence reads `...AGCUG`. The lengths agree; a single character does not.
    An exact dict lookup therefore drops the whole complex, and with it every
    variant row that needed it -- five such structures, four of them load-bearing
    for neurodev rows, were lost this way.

    So: exact match first, and only on a miss treat "X" as a wildcard against
    canonical sequences of the SAME length. The match must be UNIQUE -- if two
    accessions differ only where this structure is unknown, we cannot tell them
    apart and must not guess.
    """
    hit = seq2acc.get(seq)
    if hit is not None:
        return hit
    if "X" not in seq:
        return None
    found = set()
    for cand, acc in by_len.get(len(seq), ()):
        if all(a == "X" or a == b for a, b in zip(seq, cand)):
            found.add(acc)
            if len(found) > 1:
                return None          # ambiguous -- refuse rather than guess
    return found.pop() if len(found) == 1 else None


# Structures reach the canonical tree from two places, and only one of them may
# be redistributed: our own AlphaFold3 predictions, versus the EMBL-EBI ProtVar
# download mirrored under external/protvar_pdb. The distinction decides what can
# be deposited, so it is recorded per structure rather than inferred later.
PROVENANCE_PROTVAR = "protvar"
PROVENANCE_IN_HOUSE = "af3_in_house"


def classify_provenance(prior: str | None, source: str) -> str:
    """Provenance of one structure: an earlier verdict wins over a fresh guess.

    Re-running over an already-canonical tree makes `source` self-referential,
    so the incoming path says nothing. `prior` is that structure's verdict from
    the manifest already in the output directory, when there is one.
    """
    if prior in (PROVENANCE_PROTVAR, PROVENANCE_IN_HOUSE):
        return prior
    return PROVENANCE_PROTVAR if "protvar" in source.lower() else PROVENANCE_IN_HOUSE


def read_prior_provenance(manifest: Path) -> dict[str, str]:
    """filename -> provenance from an existing manifest, if it has the column."""
    if not manifest.exists():
        return {}
    with open(manifest, newline="") as fh:
        reader = csv.DictReader(fh)
        if "provenance" not in (reader.fieldnames or []):
            return {}
        return {r["filename"]: r["provenance"] for r in reader}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structures", required=True, nargs="+",
                    help="one or more source dirs; all land in --out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("building sequence -> accession map ...", flush=True)
    seq2acc = build_seq_map()
    # Length-bucketed view of the same map, for the UNK-tolerant fallback.
    _by_len: dict[int, list] = {}
    for _s, _a in seq2acc.items():
        _by_len.setdefault(len(_s), []).append((_s, _a))

    _EXTS = ("*.cif", "*.pdb", "*.cif.gz", "*.pdb.gz")
    per_dir = {d: sorted(f for e in _EXTS for f in glob.glob(str(Path(d) / e)))
               for d in args.structures}

    # A source directory that contributes nothing is almost always a mistake --
    # the glob is NOT recursive, and AlphaFold3 output is typically one level
    # deeper than the directory you first reach for (`af3_out/models/`, not
    # `af3_out/`). Reporting the total only, as this used to, makes a merge that
    # ingested none of your new structures look like a success: it prints a
    # plausible count (everything already in the canonical tree) and exits 0.
    empty = [d for d, fs in per_dir.items() if not fs]
    if empty:
        hint = ""
        for d in empty:
            deeper = sorted(str(Path(sub).relative_to(d))
                            for sub in glob.glob(str(Path(d) / "*"))
                            if Path(sub).is_dir()
                            and any(glob.glob(str(Path(sub) / e)) for e in _EXTS))
            if deeper:
                hint += f"\n    {d} -- did you mean {d}/{deeper[0]}/ ?"
        raise SystemExit(
            f"ERROR: {len(empty)} source directory/ies contain no structures "
            f"(the search is not recursive):\n"
            + "\n".join(f"    {d}" for d in empty) + hint)

    files = sorted(f for fs in per_dir.values() for f in fs)
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} structures across {len(args.structures)} dir(s)", flush=True)
    for d, fs in per_dir.items():
        print(f"    {len(fs):6d}  {d}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prior_provenance = read_prior_provenance(out / "manifest.csv")
    if prior_provenance:
        print(f"carrying provenance forward for {len(prior_provenance):,} structures",
              flush=True)
    parsed = Parallel(n_jobs=args.n_jobs, verbose=1)(delayed(_one)(f) for f in files)

    # Group by canonical name first: 93% of collisions are genuinely different
    # AF3 predictions of the same complex, not duplicate files, so "first in
    # sorted order" would pick arbitrarily between real alternatives. Choose the
    # highest mean pLDDT, which is deterministic and defensible.
    cand: dict = {}
    n_ok = n_unres = n_bad = n_clash = n_convert_fail = 0
    convert_errs: list[str] = []
    for f, chains, plddt in parsed:
        if len(chains) != 2:
            n_bad += 1
            continue
        (ca, sa), (cb, sb) = chains
        aa, ab = (_resolve_with_unk(sa, seq2acc, _by_len),
                  _resolve_with_unk(sb, seq2acc, _by_len))
        if not aa or not ab:
            n_unres += 1
            continue
        # Key on the PAIR only, never the format. Keying on the extension too
        # let a pair keep both a .cif and a .pdb, and measurement showed those
        # are usually DIFFERENT AF3 runs rather than conversions -- so a
        # PDB-reading method and a CIF-reading method would silently score
        # different structures for the same input. One pair, one structure.
        key = f"{min(aa, ab)}__{max(aa, ab)}"
        cand.setdefault(key, []).append((plddt, f, aa, ab, ca, cb, sa, sb))

    rows = []
    n_tie_cif = 0
    n_rescued = 0
    for key, cs in cand.items():
        n_clash += len(cs) - 1
        # Higher mean pLDDT wins. On an EXACT tie prefer the .cif: it is AF3's
        # native output, so a .pdb of the same score is a conversion of it at
        # best and a different run at worst.
        # Prefer a structure whose chains are the sequences our tables actually
        # use, THEN by pLDDT, then .cif.
        #
        # Keying on the accession pair assumes one sequence per accession, which
        # ProtVar breaks: it models a different isoform for some proteins, so a
        # higher-pLDDT ProtVar model could displace our own AF3 structure with
        # one whose chain is a different length. The pair still had a structure,
        # but no longer one matching the rows -- six VarChAMP pairs silently
        # became unscoreable that way, reported as "no structure".
        def _usable(c):
            return int(c[6] in _TABLE_SEQS and c[7] in _TABLE_SEQS)

        best = max(cs, key=lambda c: (_usable(c), c[0], ".cif" in c[1]))
        n_rescued += (1 if (_usable(best) and
                            not _usable(max(cs, key=lambda c: (c[0], ".cif" in c[1]))))
                      else 0)
        if len(cs) > 1:
            top = max(c[0] for c in cs)
            tied = [c for c in cs if c[0] == top]
            if len(tied) > 1 and ".cif" in best[1]:
                n_tie_cif += 1
        plddt, f, aa, ab, ca, cb, sa, sb = best
        name = f"{key}.cif.gz"        # one format for the whole directory

        # Recorded for traceability only. Do NOT read a disagreement between
        # this and `chain_a_accession` as a chain-order lie: the 090826 mapping
        # legitimately renames accessions (TrEMBL -> Swiss-Prot), so most
        # differences are remapping, not misordering. Detecting genuine order
        # lies needs a per-accession sequence consensus, not a string compare.
        stem = Path(f).name
        for e in (".cif.gz", ".pdb.gz", ".cif", ".pdb"):
            stem = stem[:-len(e)] if stem.endswith(e) else stem
        o1, _o2 = _old_accessions(stem)

        dest = out / name
        if not dest.exists():
            err = write_as_cif_gz(f, dest)
            if err:
                dest.with_name("." + dest.name).unlink(missing_ok=True)
                n_convert_fail += 1
                if len(convert_errs) < 5:
                    convert_errs.append(f"{name}: {err}")
                continue
        # Provenance must survive re-canonicalisation. When this runs over an
        # already-canonical tree (the usual case for a rebuild), `source` becomes
        # a self-reference and the record of which structures came from the
        # third-party ProtVar download rather than from our own AlphaFold3 runs
        # is destroyed -- which is exactly what determines what may be deposited.
        # Carry the earlier value forward when there is one.
        rows.append({"filename": name,
                     "chain_a_accession": aa, "chain_b_accession": ab,
                     "chain_a_id": ca, "chain_b_id": cb,
                     "len_a": len(sa), "len_b": len(sb),
                     "seq_a_sha": sha(sa), "seq_b_sha": sha(sb),
                     "old_name_first_token": o1,
                     "mean_plddt": round(plddt, 2),
                     "source_format": ".cif" if ".cif" in f else ".pdb",
                     "n_candidates": len(cs),
                     "source": f,
                     "provenance": classify_provenance(prior_provenance.get(name), f)})
        n_ok += 1

    if rows:
        with open(out / "manifest.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\ncanonicalised {n_ok}   unresolved {n_unres}   unparseable/non-dimer "
          f"{n_bad}   clashes {n_clash}   conversion failures {n_convert_fail}")
    print(f"ties broken in favour of .cif: {n_tie_cif}")
    print(f"pairs where a lower-pLDDT structure was preferred because its\n  sequences match the canonical tables: {n_rescued}")
    for e in convert_errs:
        print(f"   convert FAILED {e}")
    print("(clashes = two structures resolving to one pair, expected where an\n accession was remapped and both the old- and new-named file exist)")
    print(f"manifest -> {out / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
