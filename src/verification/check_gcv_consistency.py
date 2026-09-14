#!/usr/bin/env python
"""Flag GCV seeds whose AUCs do not look like the rest of their run.

A diagnostic for inspecting a run in progress -- it produces no figure, table or
deposited artifact, so it lives here rather than in `src/analysis/`.

Every GCV result is 30 independent seeds of the same procedure, so the seeds are
exchangeable: any one of them should sit inside the spread of the others. A seed
that does not is evidence that something changed underneath the run -- a rebuilt
cache, a different thread count, a code change midway -- rather than evidence
about the method.

    conda run -n ppi python src/verification/check_gcv_consistency.py
    conda run -n ppi python src/verification/check_gcv_consistency.py --baseline <dir>

Outliers are reported per (method, dataset, test class) as a robust z-score,
using the median and MAD rather than mean and standard deviation: a single bad
seed inflates the standard deviation enough to hide itself, which is exactly the
case this is meant to catch.

`--baseline` additionally compares against a directory of earlier results, for
checking that a deliberate change (a thread cap, the two-hop restriction) moved
the numbers no further than seed-to-seed noise already does.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

from utils.legacy_guard import DATASET_SUFFIX

from paths import GCV_RESULTS_DIR

CLASSES = ("C1", "C2", "C3")
# Robust z above which a seed is called out. The textbook MAD convention is 3.5,
# but per-seed AUCs are left-skewed -- an unlucky fold split costs more than a
# lucky one gains -- so the MAD understates the real spread and 3.5 flags the
# tail of a healthy run. 5.0 still catches a seed that has genuinely broken
# (0.5 among 0.85s) without crying wolf on ordinary variation.
Z_FLAG = 5.0


def _robust_z(col: np.ndarray) -> np.ndarray:
    """|x - median| / (1.4826 * MAD), 0 when the seeds are identical."""
    med = np.median(col)
    mad = np.median(np.abs(col - med))
    if mad == 0:
        return np.zeros_like(col)
    return np.abs(col - med) / (1.4826 * mad)


def load_runs(directory: str) -> dict[str, np.ndarray]:
    out = {}
    for f in sorted(glob.glob(os.path.join(directory, "*_micro_aucs.npy"))):
        stem = os.path.basename(f).replace("_micro_aucs.npy", "")
        try:
            a = np.load(f)
        except Exception:
            continue
        if a.ndim == 2 and a.shape[1] == 3:
            out[stem] = a
    return out


def check_new_seeds(runs: dict[str, np.ndarray], first_new: int) -> int:
    """Compare seeds >= `first_new` against the seeds that preceded them.

    This is the check that matters while a run is still filling in, or after
    something underneath it changed: the new seeds should be drawn from the same
    distribution as the old ones. It asks that directly instead of inferring it
    from the combined spread, which would dilute exactly the signal being looked
    for.
    """
    n_flagged = 0
    print(f"\ncomparing seeds >= {first_new} against earlier seeds of the same run")
    print("-" * 112)
    for stem, a in sorted(runs.items()):
        if a.shape[0] <= first_new:
            continue
        old, new = a[:first_new], a[first_new:]
        for c in range(3):
            o_col = old[:, c][~np.isnan(old[:, c])]
            n_col = new[:, c][~np.isnan(new[:, c])]
            if len(o_col) < 5 or len(n_col) < 1:
                continue
            spread = np.median(np.abs(o_col - np.median(o_col))) * 1.4826
            shift = abs(np.median(n_col) - np.median(o_col))
            flag = spread > 0 and shift > 3 * spread
            if flag:
                n_flagged += 1
                print(f"  !! {stem.replace(DATASET_SUFFIX, ''):<48} {CLASSES[c]} "
                      f"new median {np.median(n_col):.4f} vs old {np.median(o_col):.4f} "
                      f"(shift {shift:.4f}, old spread {spread:.4f}, n_new={len(n_col)})")
    if not n_flagged:
        print("  every run's new seeds sit inside the spread of its earlier ones")
    return n_flagged


def check(runs: dict[str, np.ndarray], baseline: dict[str, np.ndarray] | None,
          z_flag: float = Z_FLAG) -> int:
    n_flagged = 0
    print(f"{'run':<52} {'seeds':>6}  {'C1':>16} {'C2':>16} {'C3':>16}")
    print("-" * 112)
    for stem, a in sorted(runs.items()):
        cells = []
        for c in range(3):
            col = a[:, c]
            col = col[~np.isnan(col)]
            cells.append(f"{np.median(col):.4f}±{np.median(np.abs(col-np.median(col))):.4f}"
                         if len(col) else "      --      ")
        print(f"{stem.replace(DATASET_SUFFIX, ''):<52} {a.shape[0]:>3}/30  "
              f"{cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")

        # within-run outliers
        for c in range(3):
            col = a[:, c]
            ok = ~np.isnan(col)
            if ok.sum() < 5:
                continue
            z = _robust_z(col[ok])
            for seed, zz in zip(np.flatnonzero(ok), z):
                if zz > z_flag:
                    print(f"    !! seed {seed:>2} {CLASSES[c]} = {col[seed]:.4f} "
                          f"(robust z={zz:.1f} vs the other seeds)")
                    n_flagged += 1

        # against a baseline run of the same thing
        if baseline and stem in baseline:
            b = baseline[stem]
            for c in range(3):
                cur = a[:, c][~np.isnan(a[:, c])]
                old = b[:, c][~np.isnan(b[:, c])]
                if len(cur) < 3 or len(old) < 3:
                    continue
                shift = abs(np.median(cur) - np.median(old))
                spread = np.median(np.abs(old - np.median(old))) * 1.4826
                # A shift is only interesting once it exceeds the noise the
                # baseline's own seeds already show.
                if spread > 0 and shift > 2 * spread:
                    print(f"    ~~ {CLASSES[c]} median moved {shift:+.4f} vs baseline "
                          f"(baseline seed spread {spread:.4f})")
                    n_flagged += 1
    return n_flagged


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(GCV_RESULTS_DIR))
    ap.add_argument("--baseline", default=None,
                    help="directory of earlier *_micro_aucs.npy to compare against")
    ap.add_argument("--z", type=float, default=Z_FLAG)
    ap.add_argument("--new-from", type=int, default=None, metavar="SEED",
                    help="also check that seeds >= SEED match the earlier ones "
                         "in the same run -- use after a change midway through")
    args = ap.parse_args()

    runs = load_runs(args.results)
    if not runs:
        print(f"no GCV results under {args.results}")
        return 0
    base = load_runs(args.baseline) if args.baseline else None
    if args.baseline:
        print(f"baseline: {len(base)} run(s) from {args.baseline}\n")

    n = check(runs, base, args.z)
    if args.new_from is not None:
        n += check_new_seeds(runs, args.new_from)
    print()
    print(f"{len(runs)} run(s) checked, {n} thing(s) flagged"
          if n else f"{len(runs)} run(s) checked, nothing anomalous")
    return 0


if __name__ == "__main__":
    sys.exit(main())
