#!/usr/bin/env python
"""Shared MutPred2 FASTA export / CSV import, for both GCV and the VarChAMP blind test.

MutPred2 is external (no trainable model here) and partner-agnostic: one
score per (protein, mutation), independent of which partner a row happens to
test it against. Both consumers -- GCV comparison (Fig 3/S1/S3/S7's
`{dataset}_mutpred2_standalone_{preds,labels}.npy`, short-key-named to match
`method_names.py`/`roc_plots.py`) and the blind test
(`import_mutpred2_varchamp_scores.py`) -- need exactly the same two
operations: write the query FASTA MutPred2 actually expects, and parse its
output CSV back into a `(accession, mutation) -> score` lookup. One
implementation, not two independently hand-rolled ones.

**Input format** (confirmed against a real prior run,
`data/mutpred2_input.fasta` / `data/mutpred2_output.csv` -- both retained as
reference, not read by this module): one FASTA record per protein, ALL of its
tested 1-based substitutions space-separated in the header after the
accession, wild-type sequence on the next line:

    >A0A024QYX0 E80K L18P W196S
    MTTNAGPLHPYWPQ...

**Output format**: a 3-column CSV **with a header row** --
`accession,substitution,MutPred2 score` -- e.g. `A1A5D9,E58K,0.508232...`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import mutations
from utils.legacy_guard import reject_legacy


def write_fasta_for_dataset(rows: pd.DataFrame, sequences: dict, out_path: Path) -> int:
    """One FASTA record per unique interactor, listing every distinct 1-based
    mutation tested for it anywhere in `rows`. Returns the number of records
    written.

    `rows` must carry `interactor` and `mutation` (1-based, e.g. `"E80K"|),
    matching `utils.gcv_common.load_data`'s canonical column names. Partner
    and label are irrelevant here -- MutPred2 never sees them.
    """
    by_protein: dict[str, set] = {}
    for interactor, mutation in zip(rows["interactor"], rows["mutation"]):
        by_protein.setdefault(interactor, set()).add(mutation)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for accession in sorted(by_protein):
            seq = sequences.get(accession)
            if seq is None:
                continue
            muts = " ".join(sorted(by_protein[accession],
                                   key=mutations.position))
            fh.write(f">{accession} {muts}\n{seq}\n")
    return len(by_protein)


def parse_mutpred2_csv(csv_path: Path) -> dict[tuple[str, str], float]:
    """`{(accession, mutation): score}` from a MutPred2 output CSV.

    Real column names, confirmed against `data/mutpred2_output.csv`:
    `accession, substitution, MutPred2 score` (header present). Reads with
    pandas rather than hand-rolled line splitting -- the header row and any
    blank/comment lines are handled by the parser, not by relying on a
    `ValueError` from trying to float() the header (which is how the old
    `data/parse_mutpred2_output.py` silently skipped it -- that script really is
    archived now, at `archive/dead_scripts_20260910/`; until 2026-09-10 this
    docstring called it archived while it was still sitting in `data/`, live and
    able to overwrite the Fig 3 baseline arrays from a pre-090826 row ordering).
    """
    reject_legacy(csv_path, check_mtime=False)  # an external tool's fresh output, not repo data
    df = pd.read_csv(csv_path)
    cols = {c.strip().lower(): c for c in df.columns}
    acc_col = cols.get("accession") or cols.get("proteinid") or df.columns[0]
    sub_col = cols.get("substitution") or cols.get("mutation") or df.columns[1]
    score_col = next((c for c in df.columns if "score" in c.lower()), df.columns[2])
    return {
        (str(a).strip(), str(s).strip()): float(v)
        for a, s, v in zip(df[acc_col], df[sub_col], df[score_col])
    }
