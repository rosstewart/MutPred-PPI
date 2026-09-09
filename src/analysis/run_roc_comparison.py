"""Run roc_plots.py in comparison mode (ABLATION=False, BOXPLOT=False)."""
import sys, os
import roc_plots
roc_plots.ABLATION = False
roc_plots.BOXPLOT  = False
roc_plots.main()
