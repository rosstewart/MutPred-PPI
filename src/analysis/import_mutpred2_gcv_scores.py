#!/usr/bin/env python
"""Import MutPred2 scores as the partner-agnostic GCV baseline (Fig 3, S1, S3, S7).

MutPred2 has no partner information, so it is not retrained/re-split per
fold like the other comparators -- one score per (protein, mutation) is
looked up for every row of the dataset, positionally aligned to the
canonical row order, exactly like `saambe3d_cv.py`/`mutppi_cv.py` produce
their arrays. Written under the full `_mapped090826` dataset name, as
`{dataset}_mutpred2_standalone_{preds,labels}.npy` in `results/gcv/` -- the same
convention as every other fixed predictor.

Usage:
    conda run -n ppi python src/analysis/export_mutpred2_inputs.py --dataset sahni_fragoza_mapped090826
    # run MutPred2 off-machine on the resulting FASTA, then:
    conda run -n ppi python src/analysis/import_mutpred2_gcv_scores.py \\
        --dataset sahni_fragoza_mapped090826 --csv /path/to/mutpred2_output.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.legacy_guard import DATASET_SUFFIX
from analysis.mutpred2_common import parse_mutpred2_csv
from paths import GCV_RESULTS_DIR  # noqa: E402
from utils.gcv_common import dataset_arg, dataset_config, DATASET_CHOICES, DATASET_CONFIGS, load_data  # noqa: E402


def run(dataset: str, csv_path: Path, outdir: Path = GCV_RESULTS_DIR) -> None:
    # Checked before any I/O: an unstamped name means the caller is aiming at a
    # superseded mapping generation, and that should fail immediately rather
    # than after parsing a CSV.
    cfg = dataset_config(dataset)
    if not cfg.name.endswith(DATASET_SUFFIX):
        raise ValueError(
            f"{dataset!r} resolves to {cfg.name!r}, which does not carry "
            f"{DATASET_SUFFIX!r} -- refusing to write arrays that could be read "
            f"back as current.")
    rows = load_data(cfg)

    print(f"Loading MutPred2 output: {csv_path}")
    scores = parse_mutpred2_csv(csv_path)
    print(f"  {len(scores)} (accession, mutation) scores loaded")

    preds = np.array(
        [scores.get((row.interactor, row.mutation), np.nan) for row in rows.itertuples()],
        dtype=np.float32)
    labels = rows["perturbed"].to_numpy().astype(np.int8)

    n_found = int(np.sum(~np.isnan(preds)))
    print(f"  matched: {n_found}/{len(rows)} rows ({len(rows) - n_found} missing "
          f"from the MutPred2 CSV)")

    # Write under the FULL, suffixed dataset name, like every other fixed
    # predictor (`saambe3d_cv.py`, `mutppi_cv.py`). This used to write the short
    # display key to match what the figure loader read; that loader now insists
    # on `DATASET_SUFFIX`, so that arrays from a superseded mapping generation
    # cannot be picked up by accident.
    short_name = cfg.name
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    preds_path = outdir / f"{short_name}_mutpred2_standalone_preds.npy"
    labels_path = outdir / f"{short_name}_mutpred2_standalone_labels.npy"
    np.save(preds_path, preds)
    np.save(labels_path, labels)
    print(f"Saved -> {preds_path}\nSaved -> {labels_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=dataset_arg, choices=list(DATASET_CONFIGS))
    p.add_argument("--csv", required=True, help="MutPred2 output CSV file")
    p.add_argument("--outdir", default=str(GCV_RESULTS_DIR))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.dataset, Path(args.csv), Path(args.outdir))
