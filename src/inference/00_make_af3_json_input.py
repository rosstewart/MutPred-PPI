#!/usr/bin/env python3
"""AlphaFold3 input generation.

Emits one JSON per unordered protein pair, in either of the two AlphaFold3 input
dialects:

  --format local   (default)  the open-source AlphaFold3 executable
      {"name", "modelSeeds": [...],
       "sequences": [{"protein": {"sequence": ..., "id": "A"}}, ...]}
      `id` may be a bare string or a list; both are accepted by AF3. This is the
      format `run_af3_ross4.sh` consumes.

  --format server             the AlphaFold Server web UI
      {"name", "modelSeeds": [...],
       "sequences": [{"proteinChain": {"sequence": ..., "count": 1}}, ...]}
      Note the different chain key (`proteinChain`) and the use of `count`
      instead of `id`. A file in this dialect will NOT run under the local
      executable, and vice versa.

Two input modes:

  1. FASTA + triplet TSV (the original interface)
         00_make_af3_json_input.py <fasta> <triplets.tsv> <out_dir>
  2. A single CSV carrying sequences inline (the 090826 mapping layout)
         00_make_af3_json_input.py --csv rows.csv <out_dir>
     Required columns: interactor, partner, interactor_sequence, partner_sequence

Non-standard residues are substituted rather than rejected (see NONSTANDARD_AA);
AF3 accepts only the 20 standard letters inside a `sequence` string.

Author: Ross Stewart, September 2025
"""

import argparse
import csv
import json
import os
import sys

from utils.sequences import first_token, iter_fasta

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")

# AF3 rejects anything outside the 20 standard letters in a `sequence` string.
# Substitute the chemically closest standard residue instead of dropping the
# complex: U (selenocysteine) is cysteine with Se in place of S, and O
# (pyrrolysine) is a lysine derivative. Both are structurally near-identical to
# their replacement at the resolution AF3 models.
NONSTANDARD_AA = {"U": "C", "O": "K"}
# Genuinely unknown/ambiguous codes have no sensible substitute.
UNKNOWN_AA = set("BJXZ")


def normalize_sequence(uid, seq):
    """Upper-case, substitute known non-standard residues, reject the rest."""
    seq = seq.upper()
    subs = {}
    for src, dst in NONSTANDARD_AA.items():
        if src in seq:
            subs[src] = (dst, seq.count(src))
            seq = seq.replace(src, dst)
    bad = set(seq) - VALID_AAS
    if bad:
        raise ValueError(f"{uid}: unsupported residues {sorted(bad)}")
    if subs:
        detail = ", ".join(f"{s}->{d} x{n}" for s, (d, n) in subs.items())
        print(f"  {uid}: substituted {detail}")
    return seq


def pair_name(id_a, id_b):
    """`{A}__{B}`, with accessions left exactly as UniProt writes them.

    The previous scheme joined the pair with `-` and so had to rewrite isoform
    hyphens as underscores, turning `O14787-2` into `O14787_2` and making
    `o14787_2-q13207` impossible to split back into two accessions. `__` needs no
    such rewrite: no accession in the namespace contains an underscore, so the
    delimiter stays unambiguous even for isoforms and for RefSeq-style ids.
    """
    return f"{id_a}__{id_b}"


def _af3_seq_id(header: str) -> str | None:
    """First whitespace token, then its first `|`-field -- reproducing the
    previous `record.id.split("|")[0]` exactly (BioPython's `record.id` is
    the header up to the first whitespace)."""
    tok = first_token(header)
    return None if tok is None else tok.split("|")[0]


def parse_fasta(fasta_file):
    sequences = {}
    for seq_id, seq in iter_fasta(fasta_file, _af3_seq_id):
        sequences[seq_id] = normalize_sequence(seq_id, seq)
    return sequences


def parse_triplet_tsv(tsv_file):
    pairs = set()
    with open(tsv_file) as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 3:
                pairs.add(tuple(sorted([row[0], row[2]])))
    return pairs


def parse_rows_csv(csv_file):
    """Read a rows CSV with sequences inline. Returns (sequences, pairs)."""
    import pandas as pd
    need = ["interactor", "partner", "interactor_sequence", "partner_sequence"]
    df = pd.read_csv(csv_file, usecols=need)
    sequences, pairs = {}, set()
    for a, b, sa, sb in df.itertuples(index=False):
        # Keyed on the accession exactly as given -- do NOT canonicalise a
        # trailing "-1". The 090826 mapping keeps a suffix only where the
        # isoform sequence genuinely differs, and both `Q9BRI3-1` and bare
        # `Q9BRI3` occur; collapsing them pairs an accession with the wrong
        # sequence.
        sequences.setdefault(a, sa)
        sequences.setdefault(b, sb)
        pairs.add(tuple(sorted([a, b])))
    return sequences, pairs


def create_af3_json(id_a, seq_a, id_b, seq_b, seeds, fmt):
    name = pair_name(id_a, id_b)
    if fmt == "server":
        chains = [{"proteinChain": {"sequence": seq_a, "count": 1}},
                  {"proteinChain": {"sequence": seq_b, "count": 1}}]
    else:
        chains = [{"protein": {"sequence": seq_a, "id": "A"}},
                  {"protein": {"sequence": seq_b, "id": "B"}}]
    return {"name": name, "modelSeeds": list(range(1, seeds + 1)), "sequences": chains}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fasta_file", nargs="?", help="FASTA (omit when using --csv)")
    ap.add_argument("triplet_tsv", nargs="?", help="TSV id_a<TAB>variant<TAB>id_b (omit with --csv)")
    ap.add_argument("output_dir")
    ap.add_argument("--csv", help="Rows CSV with sequences inline (alternative to fasta+tsv)")
    ap.add_argument("--format", choices=("local", "server"), default="local",
                    help="AF3 input dialect (default: local executable)")
    ap.add_argument("--seeds", type=int, default=1, help="Model seeds 1-5 (default 1)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Leave already-written JSONs alone (resumable)")
    args = ap.parse_args()

    if not 1 <= args.seeds <= 5:
        sys.exit("Error: --seeds must be between 1 and 5")
    if args.csv:
        if args.fasta_file and not args.output_dir:
            args.output_dir = args.fasta_file
    elif not (args.fasta_file and args.triplet_tsv):
        sys.exit("Error: provide either --csv, or both fasta_file and triplet_tsv")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.csv:
        print(f"Reading rows CSV: {args.csv}")
        sequences, pairs = parse_rows_csv(args.csv)
        sequences = {k: normalize_sequence(k, v) for k, v in sequences.items()}
    else:
        print(f"Parsing FASTA: {args.fasta_file}")
        sequences = parse_fasta(args.fasta_file)
        print(f"Parsing triplets: {args.triplet_tsv}")
        pairs = parse_triplet_tsv(args.triplet_tsv)
    print(f"{len(sequences)} sequences, {len(pairs)} unique pairs, format={args.format}")

    written = skipped = 0
    failed = []
    for id_a, id_b in sorted(pairs):
        out = os.path.join(args.output_dir, f"{pair_name(id_a, id_b)}.json")
        if args.skip_existing and os.path.exists(out):
            skipped += 1
            continue
        try:
            data = create_af3_json(id_a, sequences[id_a], id_b, sequences[id_b],
                                   args.seeds, args.format)
        except KeyError as e:
            failed.append((id_a, id_b, f"missing sequence for {e}"))
            continue
        except ValueError as e:
            failed.append((id_a, id_b, str(e)))
            continue
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        written += 1

    print(f"\nwritten={written} skipped={skipped} failed={len(failed)}")
    for a, b, why in failed[:20]:
        print(f"  FAILED {a}-{b}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
