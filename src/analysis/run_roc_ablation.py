#!/usr/bin/env python
"""Run roc_plots in ablation mode (ABLATION=True, BOXPLOT=True) -- Figure S3.

The mode flags are module-level switches in `roc_plots`, so this sets them and
calls `main()`.

The 'Prior Best' arm is the previously published model rather than an ablation
of the current architecture, and its checkpoint ships in neither the repository
nor the Zenodo deposit, so it is omitted unless asked for.
"""
import argparse

from analysis import roc_plots


def main(include_prior_best: bool = False) -> int:
    roc_plots.ABLATION = True
    roc_plots.BOXPLOT = True
    roc_plots.INCLUDE_PRIOR_BEST = include_prior_best
    roc_plots.main()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-prior-best", action="store_true",
                    help="also draw the previously published model as a bar. "
                         "Off by default: its artifacts live under archive/ and "
                         "weights/v1_0/, neither of which is distributed, so the "
                         "bar cannot be reproduced from a clone.")
    args = ap.parse_args()
    raise SystemExit(main(args.include_prior_best))
