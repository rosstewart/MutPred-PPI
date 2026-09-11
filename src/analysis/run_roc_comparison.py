#!/usr/bin/env python
"""Run roc_plots in comparison mode (ABLATION=False, BOXPLOT=False).

Figure 3 and Figure S1 -- the per-class ROC bands across all methods.
"""
from analysis import roc_plots


def main():
    roc_plots.ABLATION = False
    roc_plots.BOXPLOT = False
    roc_plots.main()


if __name__ == "__main__":
    main()
