#!/usr/bin/env python
"""Combined robustness figure: interface / pLDDT / single-vs-multi-domain, one
figure with 3 rows (A/B/C), each row the existing 3-panel (C1/C2/C3) ROC
comparison from the corresponding standalone script.

Reuses compute_curves()/plot_on_axes() from each of the three underlying
scripts — no duplicated computation or plotting logic. Each standalone script
still produces its own PNG/TSV independently; this just also assembles them
into one combined figure.

Output:
  results/robustness/combined_robustness_by_class.png

Usage:
  conda run -n ppi python src/analysis/combined_robustness_figure.py
"""
import os

import matplotlib
from analysis import plot_style
from analysis.plot_style import SAVE_DPI
plot_style.apply()   # shared rcParams + Agg backend
import matplotlib.pyplot as plt

from analysis import interface_analysis as ia
from analysis import plddt_stratification as ps
from analysis import protein_class_stratification as pc

OUT_DIR = ia.OUT_DIR
ROW_LABELS = ["Interface vs. non-interface", "AF3 pLDDT confidence", "Single- vs. multi-domain"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Computing interface curves...", flush=True)
    ia_curves, ia_vt_group = ia.compute_curves()

    print("Computing pLDDT curves...", flush=True)
    ps_curves, ps_vt_bin = ps.compute_curves()

    print("Computing protein-class curves...", flush=True)
    pc_curves, pc_vt_grp = pc.compute_curves()

    fig, axes = plt.subplots(3, 3, figsize=(13, 13.0), sharey="row")

    rows = [
        (ia.plot_on_axes, ia_curves, ia_vt_group),
        (ps.plot_on_axes, ps_curves, ps_vt_bin),
        (pc.plot_on_axes, pc_curves, pc_vt_grp),
    ]
    for row_idx, (plot_fn, curves, group_data) in enumerate(rows):
        plot_fn(axes[row_idx], curves, group_data)
        axes[row_idx, 0].text(
            -0.28, 1.12, f"({chr(65 + row_idx)})", transform=axes[row_idx, 0].transAxes,
            fontsize=16, fontweight="bold", va="top", ha="right",
        )
        # Row-level label above the middle panel
        axes[row_idx, 1].text(
            0.5, 1.18, ROW_LABELS[row_idx], transform=axes[row_idx, 1].transAxes,
            fontsize=12, fontweight="bold", ha="center", va="bottom",
        )

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "combined_robustness_by_class.png")
    plt.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
