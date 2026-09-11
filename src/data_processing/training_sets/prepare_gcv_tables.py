#!/usr/bin/env python
"""Build the GCV row + split tables from the 090826 mapping CSVs.

This is stage 2 of the data-preparation chain: the mapping notebook
(`notebooks/map_ppi_datasets_090826.py`) produces the mapped dataset CSVs in
`datasets/source_mapping/datasets/`, and this consumes them to produce the
train/eval layer in `datasets/training_eval/` that `utils.gcv_common` reads.
See `docs/DATA_PREPARATION.md` for the full ordered chain.

Renamed from `repro_test/build_canonical_tables.py` (2026-09-10): the old name
was ambiguous with the mapping notebook -- both sounded like they "build the
datasets" -- and it lived in a gitignored directory, which is why
`REPRODUCIBILITY.md` listed the mapping CSVs as unexplained inputs.

One representation, one place. For each live dataset:

    datasets/training_eval/<name>_rows.csv.gz
        row_index, interactor, partner, mutation, position, wt_aa, mut_aa,
        perturbed, dataset, dataset_tier, fragoza_source, source_row_id, cluster
    datasets/training_eval/<name>_splits.csv.gz
        seed, row_index, test_fold, test_class

No vt_id strings, no .pos/.neg, no FASTA, no pickles. Sequences stay in the
mapping CSV and are joined by accession at load time.

`row_index` is the CSV's natural order and is never renumbered -- it is the join
key between the two tables and everything downstream.

Clustering is cd-hit at 50% identity on the **full complex sequence**
(interactor + partner concatenated), which is the published convention and must
not change. Folds are GroupKFold(10, shuffle=True, random_state=seed) over those
clusters, and `test_class` is C1/C2/C3 by whether each protein of a test pair was
seen in that fold's training set -- both exactly as in mutpred_ppi_cv.

Usage:
    python src/data_processing/training_sets/prepare_gcv_tables.py \
        [--dataset NAME] [--n-seeds 30] [--out DIR]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from paths import MAPPING_DIR, TRAINING_EVAL_DIR, cdhit_binary

MAPPING = MAPPING_DIR / "datasets"
OUT = TRAINING_EVAL_DIR

DATASETS = {
    "sahni_fragoza_varchamp_all_mapped090826": MAPPING / "sahni_fragoza_varchamp_all_mapped090826.csv",
    "sahni_fragoza_mapped090826":              MAPPING / "sahni_fragoza_mapped090826.csv",
    "varchamp_all_mapped090826":               MAPPING / "varchamp_all_mapped090826.csv",
    "sahni_only_mapped090826":                 MAPPING / "single_source" / "sahni_only_mapped090826.csv",
    "fragoza_only_mapped090826":               MAPPING / "single_source" / "fragoza_only_mapped090826.csv",
}

_RE_MUT = re.compile(r"^([A-Z])(\d+)([A-Z])$")

# Written by src/data_processing/annotate_af3_coverage.py.
AF3_FAILED_COL = "af3_failed"
ROW_COLS = ["interactor", "partner", "mutation", "position", "wt_aa", "mut_aa",
            "perturbed", "dataset", "dataset_tier",
            "fragoza_source", "source_row_id", "cluster"]


def cluster_sequences(sequences, identity: float = 0.5) -> list:
    """cd-hit clusters, verbatim semantics from mutpred_ppi_cv.cluster_sequences."""
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".fasta") as f:
        fasta_path = f.name
        for i, seq in enumerate(sequences):
            f.write(f">seq{i}\n{seq}\n")
    out_path = fasta_path + "_clustered"
    r = subprocess.run([cdhit_binary(), "-i", fasta_path, "-o", out_path,
                        "-c", str(identity), "-n", "3"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"cd-hit failed ({r.returncode}):\n{r.stderr.decode()}")
    clusters = {}
    cid = -1
    for line in open(out_path + ".clstr"):
        if line.startswith(">Cluster"):
            cid = int(line.split()[1])
        else:
            clusters[int(line.split(">seq")[1].split("...")[0])] = cid
    return [clusters[i] for i in range(len(sequences))]


def build_rows(name: str, csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)

    # Rows with no AlphaFold3 structure are dropped BEFORE row_index is
    # assigned, so row_index stays 0..n-1 and remains the positional join key
    # for the splits table and every downstream cache. They are kept, flagged,
    # in the mapping CSV for provenance; nothing past this point sees them.
    #
    # A complex with no structure has no contact graph, so every structure-based
    # method scores it NaN. Carrying such rows only spreads that NaN through the
    # per-class AUCs.
    if AF3_FAILED_COL not in df.columns:
        raise ValueError(
            f"{csv} has no '{AF3_FAILED_COL}' column. Run\n"
            f"  python src/data_processing/annotate_af3_coverage.py\n"
            f"after canonicalising the structures -- see docs/DATA_PREPARATION.md.")
    n_failed = int(df[AF3_FAILED_COL].astype(bool).sum())
    if n_failed:
        df = df[~df[AF3_FAILED_COL].astype(bool)].copy()
        print(f"  dropped {n_failed} row(s) with no AF3 structure "
              f"({AF3_FAILED_COL}=True); {len(df)} remain", flush=True)
    df = df.drop(columns=[AF3_FAILED_COL]).reset_index(drop=True)
    m = df["mutation"].str.extract(_RE_MUT)
    if m.isna().any().any():
        raise ValueError(f"{name}: unparseable mutations")
    df["wt_aa"], df["position"], df["mut_aa"] = m[0], m[1].astype(int), m[2]

    # Guarantees, asserted rather than assumed.
    seq = df["interactor_sequence"]
    # A position past the end of the sequence is a subset of validation failure,
    # not a separate error: the mapping keeps such rows flagged rather than
    # dropping them. Carry them, but never index with them.
    # Guarantees, asserted rather than assumed. The mapping emits only validated
    # mutations, so a failure here means the upstream CSV regressed.
    bad_pos = (df["position"] < 1) | (df["position"] > seq.str.len())
    if bad_pos.any():
        raise ValueError(f"{name}: {int(bad_pos.sum())} positions outside the sequence")
    n_bad = sum(1 for s, p, w in zip(seq, df["position"], df["wt_aa"]) if s[p - 1] != w)
    if n_bad:
        raise ValueError(f"{name}: {n_bad} rows where interactor_sequence[position-1] != wt_aa")
    dup = int(df.duplicated(["interactor", "partner", "mutation"]).sum())
    if dup:
        raise ValueError(f"{name}: {dup} duplicate (interactor,partner,mutation)")
    if df["perturbed"].isna().any():
        raise ValueError(f"{name}: null labels")
    print(f"  rows={len(df)}  validated, deduplicated, no null labels", flush=True)
    if dup:
        raise ValueError(f"{name}: {dup} duplicate (interactor,partner,mutation)")

    print("  clustering on the full complex sequence (cd-hit 50%)...", flush=True)
    complex_seq = (df["interactor_sequence"] + df["partner_sequence"]).tolist()
    df["cluster"] = [str(c) for c in cluster_sequences(complex_seq)]
    print(f"  {df['cluster'].nunique()} clusters", flush=True)

    df = df.reset_index(drop=True)
    df.index.name = "row_index"
    return df[[c for c in ROW_COLS if c in df.columns]]


def build_splits(rows: pd.DataFrame, n_seeds: int, n_splits: int = 10) -> pd.DataFrame:
    pairs = list(zip(rows["interactor"], rows["partner"]))
    n = len(rows)
    out = []
    for seed in range(n_seeds):
        kf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(
                kf.split(range(n), groups=rows["cluster"].values)):
            seen = set()
            for i in train_idx:
                seen.add(pairs[i][0]); seen.add(pairs[i][1])
            for i in test_idx:
                a, b = pairs[i]
                cls = 1 if (a in seen and b in seen) else (2 if (a in seen or b in seen) else 3)
                out.append((seed, int(i), fold, cls))
    return pd.DataFrame(out, columns=["seed", "row_index", "test_fold", "test_class"])


def build_sequences() -> pd.DataFrame:
    """One accession -> sequence table for every protein in any live dataset.

    Kept out of the row tables deliberately: inlining both sequences would
    duplicate tens of MB per dataset, while the union of distinct proteins is a
    couple of thousand rows. Rows join to this by accession at load time.
    """
    seqs = {}
    for name, csv in DATASETS.items():
        d = pd.read_csv(csv, usecols=["interactor", "partner",
                                      "interactor_sequence", "partner_sequence"])
        for a, s in zip(d["interactor"], d["interactor_sequence"]):
            if seqs.setdefault(a, s) != s:
                raise ValueError(f"{a}: conflicting sequences across datasets")
        for b, s in zip(d["partner"], d["partner_sequence"]):
            if seqs.setdefault(b, s) != s:
                raise ValueError(f"{b}: conflicting sequences across datasets")
    out = pd.DataFrame({"accession": list(seqs), "sequence": list(seqs.values())})
    return out.sort_values("accession").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all", choices=["all", *DATASETS])
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--out", type=Path, default=OUT,
                    help="output directory (default: datasets/training_eval/). "
                         "Point at a scratch dir to verify a rebuild against the "
                         "live tables without overwriting them.")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir != OUT:
        print(f"[note] writing to {out_dir}, not the canonical {OUT}", flush=True)
    targets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    seqs = build_sequences()
    seqs.to_csv(out_dir / "sequences.csv.gz", index=False)
    print(f"sequences.csv.gz: {len(seqs)} proteins "
          f"({seqs.sequence.str.len().min()}-{seqs.sequence.str.len().max()} aa)", flush=True)

    for name, csv in targets.items():
        print(f"\n=== {name} ===", flush=True)
        rows = build_rows(name, csv)
        rows.to_csv(out_dir / f"{name}_rows.csv.gz")
        splits = build_splits(rows, args.n_seeds)
        splits.to_csv(out_dir / f"{name}_splits.csv.gz", index=False)
        d = splits[splits.seed == 0]["test_class"].value_counts().sort_index().to_dict()
        print(f"  wrote {name}_rows.csv.gz ({len(rows)} rows) and "
              f"{name}_splits.csv.gz ({len(splits)} rows)", flush=True)
        print(f"  seed-0 classes: {d}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
