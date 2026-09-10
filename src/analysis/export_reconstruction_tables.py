#!/usr/bin/env python
"""Export reconstruction tables for manuscript figures from existing result files.

This script does NOT rerun any inference or training. It only reads already-computed
pkl/npy result files (from results/gcv, results/varchamp_seqcnf_newvar_eval,
and the three robustness-analysis modules) and reshapes them into clean, documented CSVs
under datasets/reconstruction_tables/, so that anyone can reconstruct every figure's
curves/values without access to the original models, splits, or GPU.

Usage:
    conda run -n ppi python src/analysis/export_reconstruction_tables.py --figure all
    conda run -n ppi python src/analysis/export_reconstruction_tables.py --figure gcv
    conda run -n ppi python src/analysis/export_reconstruction_tables.py --figure blind_test
    conda run -n ppi python src/analysis/export_reconstruction_tables.py --figure robustness

Outputs (datasets/reconstruction_tables/):
    gcv_{method}.csv                          one row per (gcv_seed, fold, test_class, sample)
    blind_test_{method}.csv                   one row per (test_class, sample) for VarChAMP/VCFP
    robustness_{interface,plddt,protein_class}_curves.csv
    robustness_{interface,plddt,protein_class}_summary.csv
    README.md                                 figure -> file -> column meaning -> regen command
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from method_names import (  # noqa: E402
    METHOD_DISPLAY_NAMES, extract_method_and_dataset)
from paths import (  # noqa: E402
    CV_DIR as _P_CV_DIR, GCV_RESULTS_DIR, REPO_ROOT, VARCHAMP_BLIND_TEST_DIR, cv_reference_dir)


# ── Paths ──────────────────────────────────────────────────────────────────────
_PUB = str(REPO_ROOT)
_ANALYSIS_DIR = os.path.join(_PUB, "src", "analysis")
GCV_DIR = str(GCV_RESULTS_DIR)
BLIND_TEST_DIR = str(VARCHAMP_BLIND_TEST_DIR)
CV_DIR = str(cv_reference_dir())
OUT_DIR = os.path.join(_PUB, "datasets", "reconstruction_tables")


N_SEEDS = 30
from gcv_curves import FPR_GRID, N_SEM_DIVISOR  # noqa: E402  (single definition)


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers (copied, not imported, from roc_plots.py — that module has heavy
# side-effecting top-level code (writes npy/pkl files on import) so we only reuse
# its pure logic here).
# ═════════════════════════════════════════════════════════════════════════════

# Method display-name table + dataset/method extraction, copied verbatim (pure,
# no side effects) from roc_plots.py so that the set of GCV files we export
# exactly matches what main_comparison() actually plots for Fig 3 / S1 / S-new.


# Canonical rows support: datasets whose fold_splits indices are row positions
# into a canonical rows CSV (built after the 090826 mapping).  These can be
# linked to a specific (interactor, partner, mutation) directly.  Datasets not
# listed here have no canonical rows CSV and return an empty index.
#
# NOTE: sahni_fragoza_varchamp2026 and other varchamp configs are deliberately
# NOT included.  Their cv_splits carry '.bak_before_conflict_removal' backups
# indicating the splits were edited after the GCV pkls were generated,
# producing <1% per-fold/class count matches (744/221070 rows) and therefore
# essentially no reliable alignment.  vt_id linkage for those datasets is left
# blank rather than emitting misleading near-empty joins.
CANONICAL_ROWS_SUPPORT = {
    'sahni': dict(
        rows_csv='sahni_only_train_rows.csv.gz',
        splits_prefix='sahni_only_train_',
        ptc_prefix='sahni_only_',
    ),
    'sahni_fragoza': dict(
        rows_csv='sahni_fragoza_train_rows.csv.gz',
        splits_prefix='sahni_fragoza_train_',
        ptc_prefix='swing_train_',
    ),
}


def _vt_id_index_for_dataset(dataset: str, detailed_results: dict):
    """Return {(seed, fold, cls): [(interactor, partner, mutation), ...]} aligned
    with the pkl's per-class preds/labels arrays, or {} if unsupported / no match.

    For datasets in CANONICAL_ROWS_SUPPORT, row identity comes from the canonical
    rows CSV (built after the 090826 mapping): `test_idx` from fold_splits are
    positional indices into that CSV, so `canonical_rows.iloc[ridx]` gives
    (interactor, partner, mutation) without any string splitting.

    If a fold/class's row count doesn't match the pkl's preds count (methods may
    drop invalid preds), that slice is left unlinked rather than mis-aligned.
    """
    cfg = CANONICAL_ROWS_SUPPORT.get(dataset)
    if cfg is None:
        return {}

    rows_path = f"{CV_DIR}/{cfg['rows_csv']}"
    if not os.path.exists(rows_path):
        return {}
    canonical_rows = pd.read_csv(rows_path)

    index = {}
    for seed in range(N_SEEDS):
        fold_splits_path = f"{CV_DIR}/{cfg['splits_prefix']}fold_splits_{seed}.pkl"
        ptc_path = f"{CV_DIR}/{cfg['ptc_prefix']}pair_test_classes_{seed}.npy"
        if not all(os.path.exists(p) for p in (fold_splits_path, ptc_path)):
            continue
        iteration = detailed_results['iterations'].get(seed)
        if iteration is None:
            continue

        with open(fold_splits_path, 'rb') as f:
            fold_splits = pickle.load(f)
        ptc = np.load(ptc_path)

        flat_cursor = 0
        for fold, train_idx, test_idx in sorted(fold_splits, key=lambda t: t[0]):
            fold_data = iteration['folds'].get(fold)
            if fold_data is None:
                flat_cursor += len(test_idx)
                continue
            n_test = len(test_idx)
            ptc_fold = ptc[flat_cursor:flat_cursor + n_test]

            # Each test_idx is a canonical row position; no vt_ids pkl needed.
            fold_rows = [canonical_rows.iloc[ridx] for ridx in test_idx]

            per_class: dict[int, list] = {1: [], 2: [], 3: []}
            for row, cls in zip(fold_rows, ptc_fold):
                per_class[int(cls)].append(
                    (str(row['interactor']), str(row['partner']), str(row['mutation'])))

            for cls in (1, 2, 3):
                n_preds = len(fold_data[f'class_{cls}']['preds'])
                if n_preds == len(per_class[cls]):
                    index[(seed, fold, cls)] = per_class[cls]
                # else: leave unlinked (count mismatch) — no entry in index

            flat_cursor += n_test

    return index


# ═════════════════════════════════════════════════════════════════════════════
# GCV export (Fig 3, S1)
# ═════════════════════════════════════════════════════════════════════════════

def export_gcv():
    os.makedirs(OUT_DIR, exist_ok=True)
    detailed_files = glob.glob(os.path.join(GCV_DIR, "*_detailed_results.pkl"))
    detailed_files = [f for f in detailed_files
                      if 'mutpred2' not in f.lower() and 'saambe' not in f.lower()]

    written = []
    for filepath in sorted(detailed_files):
        method, dataset = extract_method_and_dataset(filepath)
        if method is None or method not in METHOD_DISPLAY_NAMES:
            continue

        with open(filepath, 'rb') as f:
            detailed_results = pickle.load(f)

        vt_index = _vt_id_index_for_dataset(dataset, detailed_results)

        rows = []
        for seed, iter_data in detailed_results['iterations'].items():
            for fold, fold_data in iter_data['folds'].items():
                for cls in (1, 2, 3):
                    class_data = fold_data[f'class_{cls}']
                    preds = np.asarray(class_data['preds'])
                    labels = np.asarray(class_data['labels'])
                    # Each entry is (interactor, partner, mutation) from canonical rows.
                    row_ids = vt_index.get((seed, fold, cls))
                    for i in range(len(preds)):
                        if row_ids is not None:
                            interactor, partner, variant = row_ids[i]
                            vt_id = f"{interactor}-{partner} {variant}"
                        else:
                            interactor = partner = variant = vt_id = None
                        rows.append((
                            vt_id, interactor, partner, variant,
                            seed, fold, cls,
                            float(labels[i]), float(preds[i]),
                        ))

        df = pd.DataFrame(rows, columns=[
            'vt_id', 'interactor', 'partner', 'variant',
            'gcv_seed', 'fold', 'test_class', 'true_label', 'predicted_score',
        ])
        out_path = os.path.join(OUT_DIR, f'gcv_{method}.csv')
        df.to_csv(out_path, index=False)
        written.append((out_path, len(df)))
        n_linked = df['vt_id'].notna().sum()
        print(f"  Wrote {out_path} ({len(df)} rows, {n_linked} vt_id-linked, dataset={dataset})")

    return written


# ═════════════════════════════════════════════════════════════════════════════
# Blind test / VarChAMP-VCFP export (Fig 4, S2)
# ═════════════════════════════════════════════════════════════════════════════

def _extract_blind_method_name(filepath: str) -> str:
    fname = os.path.basename(filepath)
    return re.sub(r'_c[123]_(labels|preds|vt_ids)\.npy$', '', fname)


def _parse_blind_vt_id(vt_id: str):
    """vt_id formats observed on disk (see task notes):
      3 parts 'interactor partner variant'  (varchamp_full_pooled — main Fig4/S2 arrays)
      2 parts 'interactor_partner variant'  (varchamp_full / varchamp_pooled arrays)
      2 parts 'interactor variant'          (older single-protein-only arrays, no partner)
    """
    parts = vt_id.split(' ')
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        complex_part, variant = parts
        if '_' in complex_part:
            interactor, partner = complex_part.split('_', 1)
        else:
            interactor, partner = complex_part, ''
        return interactor, partner, variant
    return '', '', vt_id


def _slugify(method_name: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9]+', '_', method_name).strip('_')
    return slug


def export_blind_test():
    os.makedirs(OUT_DIR, exist_ok=True)
    label_files = glob.glob(os.path.join(BLIND_TEST_DIR, "*_c?_labels.npy"))
    methods = sorted({_extract_blind_method_name(f) for f in label_files})

    written = []
    for method in methods:
        rows = []
        any_data = False
        for c in (1, 2, 3):
            preds_f = os.path.join(BLIND_TEST_DIR, f"{method}_c{c}_preds.npy")
            labels_f = os.path.join(BLIND_TEST_DIR, f"{method}_c{c}_labels.npy")
            vtids_f = os.path.join(BLIND_TEST_DIR, f"{method}_c{c}_vt_ids.npy")
            if not (os.path.exists(preds_f) and os.path.exists(labels_f)):
                continue
            preds = np.load(preds_f, allow_pickle=True)
            labels = np.load(labels_f, allow_pickle=True)
            vt_ids = np.load(vtids_f, allow_pickle=True) if os.path.exists(vtids_f) else None
            any_data = True
            for i in range(len(preds)):
                if vt_ids is not None:
                    interactor, partner, variant = _parse_blind_vt_id(str(vt_ids[i]))
                    vt_id = str(vt_ids[i])
                else:
                    vt_id = interactor = partner = variant = None
                rows.append((vt_id, interactor, partner, variant, f'C{c}',
                             float(labels[i]), float(preds[i])))
        if not any_data:
            continue

        df = pd.DataFrame(rows, columns=[
            'vt_id', 'interactor', 'partner', 'variant',
            'test_class', 'true_label', 'predicted_score',
        ])
        out_path = os.path.join(OUT_DIR, f'blind_test_{_slugify(method)}.csv')
        df.to_csv(out_path, index=False)
        written.append((out_path, len(df)))
        print(f"  Wrote {out_path} ({len(df)} rows) [{method}]")

    return written


# ═════════════════════════════════════════════════════════════════════════════
# Robustness figures (interface / pLDDT / protein-class stratification)
# ═════════════════════════════════════════════════════════════════════════════

def _export_one_robustness(name: str, module, group_order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"  Computing curves for {name} (this re-derives from existing GCV pkl + caches, "
          f"no inference is rerun)...")
    fold_curves, group_data = module.compute_curves()

    # Authoritative summary rows (n_variants, n_fold_curves, mean/sem AUROC) — reuse
    # the module's own plot_on_axes() so the numbers exactly match the published
    # figure/TSV; the matplotlib figure itself is discarded (not saved).
    fig, axes = plt.subplots(1, 3)
    summary_rows = module.plot_on_axes(axes, fold_curves, group_data)
    plt.close(fig)

    summary_records = []
    for row in summary_rows:
        cls_str, group, mean_auc, sem, n_variants, n_fold = row.split('\t')
        summary_records.append({
            'class': cls_str, 'group': group,
            'auc': float(mean_auc) if mean_auc != 'nan' else np.nan,
            'sem': float(sem) if sem != 'nan' else np.nan,
            'n_variants': int(n_variants), 'n_fold_curves': int(n_fold),
        })
    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(OUT_DIR, f'robustness_{name}_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    # Curve points: mean TPR ± SEM over the shared FPR_GRID, computed with the
    # identical formula used by plot_on_axes() (std ddof=1, SEM = std/sqrt(10)).
    curve_rows = []
    for cls in (1, 2, 3):
        for group in group_order:
            curves = fold_curves[cls][group]
            if not curves:
                continue
            arr = np.array(curves)
            mean_tpr = np.mean(arr, axis=0)
            std_tpr = np.std(arr, axis=0, ddof=1)
            sem_tpr = std_tpr / np.sqrt(N_SEM_DIVISOR)
            for fpr, mtpr, stpr in zip(FPR_GRID, mean_tpr, sem_tpr):
                curve_rows.append({
                    'class': f'C{cls}', 'group': group,
                    'fpr': fpr, 'mean_tpr': mtpr, 'sem_tpr': stpr,
                })
    curves_df = pd.DataFrame(curve_rows)
    curves_path = os.path.join(OUT_DIR, f'robustness_{name}_curves.csv')
    curves_df.to_csv(curves_path, index=False)

    print(f"  Wrote {curves_path} ({len(curves_df)} rows)")
    print(f"  Wrote {summary_path} ({len(summary_df)} rows)")
    return [(curves_path, len(curves_df)), (summary_path, len(summary_df))]


def export_robustness():
    os.makedirs(OUT_DIR, exist_ok=True)
    import interface_analysis
    import plddt_stratification
    import protein_class_stratification

    written = []
    written += _export_one_robustness('interface', interface_analysis,
                                       ['interface', 'non_interface'])
    written += _export_one_robustness('plddt', plddt_stratification,
                                       ['low', 'medium', 'high'])
    written += _export_one_robustness('protein_class', protein_class_stratification,
                                       ['single', 'multi'])
    return written


# ═════════════════════════════════════════════════════════════════════════════
# README
# ═════════════════════════════════════════════════════════════════════════════

README_TEMPLATE = """# Reconstruction tables

These CSVs let anyone reconstruct every manuscript figure's curves/values directly from
already-computed predictions and labels, without retraining any model or rerunning inference.
Generate (or regenerate) all of them with:

```
conda run -n ppi python src/analysis/export_reconstruction_tables.py --figure all
```

| Figure(s) | File(s) | Columns | Regeneration command |
|---|---|---|---|
| Fig 3, S1 (GCV method comparison ROC curves) | `gcv_{method}.csv` (one per method/dataset combination plotted in `roc_plots.py`) | `vt_id, interactor, partner, variant, gcv_seed, fold, test_class, true_label, predicted_score`. `vt_id`/`interactor`/`partner`/`variant` are populated only where the underlying cv_splits bookkeeping files exist with per-seed variant orderings AND those orderings are internally consistent with the pkl's per-fold/class sample counts (sahni, sahni_fragoza, sahni_varchamp1p_cava, sahni_fragoza_varchamp1p_cava datasets); rows for datasets without usable per-seed vt_ids files (sahni_fragoza_varchamp2026, sahni_fragoza_varchamp_pooled/_full/_full_pooled) still contain preds/labels but leave those 4 columns blank. Recompute AUC per (gcv_seed, fold, test_class) with `sklearn.metrics.roc_auc_score`; average across seeds/folds per test_class to reproduce Fig 3/S1. | `conda run -n ppi python src/analysis/export_reconstruction_tables.py --figure gcv` |
| Fig 4, S2 (VarChAMP/VCFP blind test ROC curves) | `blind_test_{method}.csv` (one per method key found in `results/varchamp_seqcnf_newvar_eval/`) | `vt_id, interactor, partner, variant, test_class, true_label, predicted_score` (test_class in {C1,C2,C3}). Recompute AUC per test_class with `sklearn.metrics.roc_auc_score` to reproduce Fig 4/S2. | `conda run -n ppi python src/analysis/export_reconstruction_tables.py --figure blind_test` |
| Fig 5, S4, S-stability (variant-DB disease-enrichment / stability-interaction figures) | `../master_variant_db_predictions.csv.gz` (one row per interactor/variant/partner triplet, all-data-model predictions + disease-label flags) | See that file's own header row. | (already generated; not produced by this script) |
| Interface-residue robustness panel | `robustness_interface_curves.csv`, `robustness_interface_summary.csv` | curves: `class, group (interface/non_interface), fpr, mean_tpr, sem_tpr`; summary: `class, group, auc, sem, n_variants, n_fold_curves` | `conda run -n ppi python src/analysis/export_reconstruction_tables.py --figure robustness` |
| pLDDT-stratification robustness panel | `robustness_plddt_curves.csv`, `robustness_plddt_summary.csv` | curves: `class, group (low/medium/high), fpr, mean_tpr, sem_tpr`; summary: `class, group, auc, sem, n_variants, n_fold_curves` | (same as above) |
| Protein-class (single- vs multi-domain) robustness panel | `robustness_protein_class_curves.csv`, `robustness_protein_class_summary.csv` | curves: `class, group (single/multi), fpr, mean_tpr, sem_tpr`; summary: `class, group, auc, sem, n_variants, n_fold_curves` | (same as above) |

Notes:
- `test_class` for GCV/blind-test tables follows the manuscript's C1/C2/C3 convention:
  C1 = both interactors seen in training, C2 = one seen, C3 = neither seen.
- The robustness `_curves.csv` files store the mean TPR curve +/- SEM over the shared
  100-point FPR grid (`np.linspace(0, 1, 100)`), matching exactly what
  `interface_analysis.py` / `plddt_stratification.py` / `protein_class_stratification.py`
  plot; the `_summary.csv` files are the same summary rows those scripts already write to
  their own `.tsv` outputs, just reformatted as CSV with a `class`/`group` split.
- Nothing in `results/variant_dbs_all_data/` is read or touched by this script.
"""


def write_readme():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'README.md'), 'w') as f:
        f.write(README_TEMPLATE)
    print(f"  Wrote {os.path.join(OUT_DIR, 'README.md')}")


# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=['gcv', 'blind_test', 'robustness', 'all'],
                         default='all')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.figure in ('gcv', 'all'):
        print("== GCV (Fig 3, S1) ==")
        export_gcv()
    if args.figure in ('blind_test', 'all'):
        print("== Blind test / VCFP (Fig 4, S2) ==")
        export_blind_test()
    if args.figure in ('robustness', 'all'):
        print("== Robustness panels ==")
        export_robustness()

    write_readme()
    print("Done.")


if __name__ == '__main__':
    main()
