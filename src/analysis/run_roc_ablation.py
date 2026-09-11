#!/usr/bin/env python
"""Run roc_plots in ablation mode (ABLATION=True, BOXPLOT=True).

Figures S3 and the ablation panels. The mode flags are module-level switches in
`roc_plots`, so this sets them and calls `main()`.
"""
from analysis import roc_plots


def main():
    roc_plots.ABLATION = True
    roc_plots.BOXPLOT = True
    roc_plots.main()


if __name__ == "__main__":
    main()
