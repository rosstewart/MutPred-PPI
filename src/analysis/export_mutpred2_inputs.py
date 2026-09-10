#!/usr/bin/env python
"""Export a MutPred2 query FASTA for one canonical dataset.

MutPred2 is run externally (no trainable model in this repo); this writes
the input file it expects and stops there -- see
the module notes below for why (it is
"run externally; ... only parses its output CSV. No deviation is possible.").

Usage:
    conda run -n ppi python src/analysis/export_mutpred2_inputs.py \\
        --dataset sahni_fragoza_mapped090826
    conda run -n ppi python src/analysis/export_mutpred2_inputs.py \\
        --dataset varchamp_all_mapped090826   # the blind-test target

Then run MutPred2 on the resulting FASTA off-machine, and feed its output CSV to:
    src/analysis/import_mutpred2_gcv_scores.py       (GCV datasets)
    src/analysis/import_mutpred2_varchamp_scores.py  (varchamp_all_mapped090826)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mutpred2_common import write_fasta_for_dataset
from paths import DATASETS_DIR  # noqa: E402
from utils.gcv_common import DATASET_CONFIGS, load_data, load_sequences  # noqa: E402

OUT_DIR = DATASETS_DIR / "mutpred2_inputs"


def run(dataset: str, out: Path | None = None) -> None:
    cfg = DATASET_CONFIGS[dataset]
    rows = load_data(cfg)
    sequences = load_sequences()

    out_path = out or (OUT_DIR / f"{dataset}_mutpred2_input.fasta")
    n = write_fasta_for_dataset(rows, sequences, out_path)
    n_muts = rows.groupby("interactor")["mutation"].nunique().sum()
    print(f"{dataset}: {len(rows)} rows -> {n} proteins, {n_muts} distinct "
          f"(protein, mutation) queries -> {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    p.add_argument("--out", default=None, help="Output FASTA path (default: "
                   "datasets/mutpred2_inputs/{dataset}_mutpred2_input.fasta)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.dataset, Path(args.out) if args.out else None)
