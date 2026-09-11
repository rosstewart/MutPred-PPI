"""Constants shared by every GCV curve/error-bar computation.

`N_SEM_DIVISOR` was copy-pasted into five modules, each carrying a comment saying
it "matches hardcoded value in roc_plots.py". It sets the denominator of the
standard error on every ROC band and AUC error bar in the paper, so five
independent copies of a number that must agree is a real hazard: change one and
the figures silently disagree with each other.

Same for `FPR_GRID` -- the interpolation grid must be identical across scripts or
mean curves are averaged at different x positions.

Importing this module changes no value; all five copies held the same numbers.
"""
from __future__ import annotations

import numpy as np

# Standard error denominator for per-fold AUC/TPR aggregation.
#
# This is the number of CV folds (10), not the number of curves being averaged
# (30 seeds x 10 folds = 300).  It is deliberately conservative: the 300 curves
# are not independent -- each seed re-partitions the same rows -- so dividing by
# sqrt(300) would understate the uncertainty considerably.  The published figures
# use sqrt(10).
N_SEM_DIVISOR = 10

# Common false-positive-rate grid that every ROC curve is interpolated onto
# before averaging.
FPR_GRID = np.linspace(0, 1, 100)

__all__ = ["N_SEM_DIVISOR", "FPR_GRID", "class_roc_auc"]


def class_roc_auc(preds, labels):
    """`(fpr, tpr, auc)` for one class's predictions, or `(None, None, nan)`.

    The single NaN-masking policy for a one-off (non-fold-averaged) ROC/AUC:
    mask NaN in EITHER `preds` or `labels`, matching
    `utils.gcv_common._compute_class_aucs`. Before 2026-09-10 this was computed
    three times with three different masks -- preds-only (`biclass_sf_gcv.py`),
    no mask at all (`blind_test_figures.py`), and preds+labels (`gcv_common`).

    The mask is DEFENSIVE. Labels never contain NaN (they come from the
    canonical `perturbed` column, always 0/1), and predictions should not
    either: rows whose complex has no AlphaFold3 structure are dropped at
    dataset-build time (`af3_failed`), and the GCV path raises rather than
    scoring an incomplete row. The one path that can still yield NaN is the
    VarChAMP blind test, which builds tensors with `require_complete=False` and
    prints exactly how many rows it excluded and why. Having one policy here
    means a hole in any of those arrays is handled the same way everywhere,
    instead of three call sites disagreeing about which rows count.

    Returns `(None, None, nan)` if there are fewer than 2 valid samples or only
    one label class present -- the standard "cannot draw a curve" case, treated
    as no-curve rather than an error.
    """
    import numpy as np
    from sklearn.metrics import auc, roc_curve

    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=float)
    valid = ~np.isnan(preds) & ~np.isnan(labels)
    p, l = preds[valid], labels[valid]
    if len(p) == 0 or len(np.unique(l)) < 2:
        return None, None, float("nan")
    fpr, tpr, _ = roc_curve(l, p)
    return fpr, tpr, auc(fpr, tpr)
