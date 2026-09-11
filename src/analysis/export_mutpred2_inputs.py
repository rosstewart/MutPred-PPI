#!/usr/bin/env python
"""Export a MutPred2 query FASTA for one canonical dataset, or their union.

MutPred2 is run externally (no trainable model in this repo); this writes
the input file it expects and stops there -- see
the module notes below for why (it is
"run externally; ... only parses its output CSV. No deviation is possible.").

`--dataset all` unions all five canonical datasets first (via
`gcv_common.union_rows_across_datasets`, the same helper `prepare_af3_inputs.py`
uses for its required-pair set): a row can be pooled out of
`sahni_fragoza_varchamp_all` during conflict resolution even though the same
(interactor, mutation) is present and unconflicted in a smaller dataset, so the
pooled table alone can under-count. `import_mutpred2_gcv_scores.py` and
`import_mutpred2_varchamp_scores.py` both key strictly on (interactor,
mutation), so one union run's output CSV feeds every dataset's import.

Usage:
    conda run -n ppi python src/analysis/export_mutpred2_inputs.py \\
        --dataset sahni_fragoza_mapped090826
    conda run -n ppi python src/analysis/export_mutpred2_inputs.py \\
        --dataset varchamp_all_mapped090826   # the blind-test target
    conda run -n ppi python src/analysis/export_mutpred2_inputs.py \\
        --dataset all                         # union of all five, one run

Then run MutPred2 on the resulting FASTA off-machine, and feed its output CSV to:
    src/analysis/import_mutpred2_gcv_scores.py       (GCV datasets)
    src/analysis/import_mutpred2_varchamp_scores.py  (varchamp_all_mapped090826)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analysis.mutpred2_common import write_fasta_for_dataset
from paths import DATASETS_DIR  # noqa: E402
from utils.gcv_common import dataset_arg, dataset_config, DATASET_CHOICES, DATASET_CONFIGS, load_data, load_sequences, union_rows_across_datasets  # noqa: E402

OUT_DIR = DATASETS_DIR / "mutpred2_inputs"


def run(dataset: str, out: Path | None = None) -> None:
    if dataset == "all":
        rows = union_rows_across_datasets()
    else:
        rows = load_data(dataset_config(dataset))
    sequences = load_sequences()

    out_path = out or (OUT_DIR / f"{dataset}_mutpred2_input.fasta")
    n = write_fasta_for_dataset(rows, sequences, out_path)
    n_muts = rows.groupby("interactor")["mutation"].nunique().sum()
    print(f"{dataset}: {len(rows)} rows -> {n} proteins, {n_muts} distinct "
          f"(protein, mutation) queries -> {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   type=lambda v: v if v == "all" else dataset_arg(v),
                   choices=list(DATASET_CONFIGS) + ["all"])
    p.add_argument("--out", default=None, help="Output FASTA path (default: "
                   "datasets/mutpred2_inputs/{dataset}_mutpred2_input.fasta)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.dataset, Path(args.out) if args.out else None)
