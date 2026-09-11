#!/usr/bin/env python
"""Validate our 573-dim eSIG-Net featurization against upstream's shipped output.

eSIG-Net publishes no feature-extraction code, so `predictors/esignet.py`
reconstructs it (AAC 20 + Conjoint-Triad 343 + auto-covariance 210).  Upstream
*does* ship the result of the code they withheld:
`datasets/embeddings/sdnn_corrected_ppi.h5`, 1094 proteins x 573 float64.

The h5 is keyed by `smile.<entrez id>` and ships no sequences, so a value-for-value
diff is not possible.  What is possible -- and decisive for the auto-covariance
block -- is a *structural* comparison:

  `_build_ac_norm()` z-scores each of the 7 properties across the 20 amino acids.
  A property whose table contains a wild outlier collapses after z-scoring (19
  values crushed together, 1 spike), so the 30 AC dimensions carrying it lose
  nearly all between-protein variance.  Upstream's per-property variance profile
  therefore tells us whether *their* table had such an outlier.  Ours must match.

Usage:
    python src/evaluation/predictors/validate_esignet_features.py [--n 300]

Lived in the gitignored `repro_test/` until 2026-09-10, where three
docstrings in `esignet.py` cited it as the evidence for the
reconstruction -- evidence no cloner could actually open.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import h5py
import numpy as np

# No sys.path bootstrap needed: `pip install -e .` puts src/ on the path.
_PUB = Path(__file__).resolve().parents[3]

# Resolved lazily -- this module must import without the eSIG-Net checkout.
def _upstream_h5():
    """Upstream's shipped 573-dim feature matrix, from the eSIG-Net checkout."""
    return method_dir("esignet") / "datasets" / "embeddings" / "sdnn_corrected_ppi.h5"


def _seq_fasta():
    """WT+variant sequences for the proteins we featurise."""
    return EXTERNAL_DIR / "swing_train" / "swing_train_wt_and_vt.fasta"

BLOCKS = {"AAC": (0, 20), "CT": (20, 363), "AC": (363, 573)}


def read_fasta(path: Path, limit: int) -> list[str]:
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    if len(seqs) >= limit:
                        return seqs
                    cur = []
            else:
                cur.append(line.strip())
    if cur and len(seqs) < limit:
        seqs.append("".join(cur))
    return seqs


def property_profile(ac_block: np.ndarray) -> np.ndarray:
    """Mean between-protein std of the 30 lags belonging to each of the 7 properties."""
    return ac_block.std(0).reshape(7, 30).mean(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="sequences to featurize")
    args = ap.parse_args()

    if not _upstream_h5().exists():
        print(f"missing upstream h5: {_upstream_h5()}", file=sys.stderr)
        return 2

    with h5py.File(_upstream_h5(), "r") as f:
        up = np.stack([f[k][:] for k in f.keys()])
    print(f"upstream: {up.shape[0]} proteins x {up.shape[1]} dims\n")

    # --- structural invariants: confirms our block boundaries are right ---
    print("=== block invariants (upstream) ===")
    for name, (a, b) in BLOCKS.items():
        B = up[:, a:b]
        print(f"  {name:4s} [{a}:{b}]  rowsum={B.sum(1).mean():7.4f}  "
              f"min={B.min():8.4f}  max={B.max():7.4f}  frac_neg={(B < 0).mean():.3f}")
    print("  (AAC and CT are frequency blocks -> rowsum 1.0, non-negative;"
          " AC is signed)\n")

    from evaluation.predictors import esignet as es

    seqs = read_fasta(_seq_fasta(), args.n)
    print(f"featurizing {len(seqs)} sequences from {_seq_fasta().name}\n")

    shipped = copy.deepcopy(es._AC_AA_PROPS)
    fixed = copy.deepcopy(shipped)
    fixed["Y"]["NCI"], fixed["Y"]["V"] = shipped["Y"]["V"], shipped["Y"]["NCI"]

    profiles = {}
    for label, table in (("shipped", shipped), ("Y-swapped", fixed)):
        es._AC_AA_PROPS = table
        es._AC_NORM = es._build_ac_norm()
        # _compute_573 memoizes by sequence; without this the second pass would
        # silently return the first pass's vectors and both tables would "agree".
        es._FEAT_CACHE.clear()
        ours = np.stack([es._compute_573(s) for s in seqs])
        profiles[label] = property_profile(ours[:, 363:573])
        if label == "shipped":
            print("=== our block invariants (shipped table) ===")
            for name, (a, b) in BLOCKS.items():
                B = ours[:, a:b]
                print(f"  {name:4s} [{a}:{b}]  rowsum={B.sum(1).mean():7.4f}  "
                      f"min={B.min():8.4f}  max={B.max():7.4f}  "
                      f"frac_neg={(B < 0).mean():.3f}")
            print()

    up_prof = property_profile(up[:, 363:573])

    print("=== auto-covariance: mean between-protein std per property ===")
    print(f"{'property':10s} {'upstream':>10s} {'ours(shipped)':>14s} "
          f"{'ours(Y-swapped)':>16s}")
    print("-" * 54)
    for i, prop in enumerate(es._AC_PROPERTIES):
        print(f"{prop:10s} {up_prof[i]:10.5f} {profiles['shipped'][i]:14.5f} "
              f"{profiles['Y-swapped'][i]:16.5f}")

    def ratio_spread(p):
        """How degenerate is the weakest property relative to the strongest."""
        return p.min() / p.max()

    print()
    print(f"weakest/strongest property ratio -- upstream {ratio_spread(up_prof):.3f}, "
          f"ours shipped {ratio_spread(profiles['shipped']):.3f}, "
          f"ours Y-swapped {ratio_spread(profiles['Y-swapped']):.3f}")

    # Correlate profile shape; the absolute scale differs (different protein sets).
    for label in ("shipped", "Y-swapped"):
        c = np.corrcoef(up_prof, profiles[label])[0, 1]
        print(f"profile correlation with upstream ({label:10s}): {c:+.4f}")

    print()
    nci = es._AC_PROPERTIES.index("NCI")
    if profiles["shipped"][nci] < 0.25 * profiles["shipped"].max() <= \
       profiles["Y-swapped"][nci] / 0.25:
        print("VERDICT: the shipped table degenerates the NCI block; the Y swap "
              "restores a profile consistent with upstream.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
