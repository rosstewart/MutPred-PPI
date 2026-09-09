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

__all__ = ["N_SEM_DIVISOR", "FPR_GRID"]
