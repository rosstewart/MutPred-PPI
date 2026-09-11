"""One matplotlib configuration for every figure in the paper.

Before 2026-09-10 the figure scripts disagreed with each other in three ways
that a reader sees directly:

* **Output resolution.** `savefig(dpi=...)` took five different values across
  manuscript figures -- 100, 150, 200, 300, and two script-local constants.
  Figure 3 shipped at 300 dpi while the pLDDT, interface and protein-class
  supplements shipped at 150, so the supplements were visibly softer than the
  main figures for no reason anyone had chosen.
* **Fonts and axes.** Two scripts set an `rcParams` block (identical, by
  copy-paste); the other thirteen used matplotlib's defaults, so font sizes and
  axis line weights differed between panels of the same paper.
* **Colours.** Three hand-maintained palette dicts, agreeing on shared keys only
  because someone kept them in step.

`apply()` is called by each figure script at import; `SAVE_DPI` is the single
number to change if a journal asks for a different resolution.

Changing these changes IMAGE FILES ONLY -- no number, curve or table value
depends on anything here.
"""
from __future__ import annotations

__all__ = ["SAVE_DPI", "FIGURE_DPI", "METHOD_COLORS", "CLASS_LABELS", "apply"]

# Output resolution for every saved figure. 300 dpi is the usual journal
# minimum for line art and was already what the main figures used.
SAVE_DPI = 300

# On-screen/backend resolution; only affects interactive rendering size.
FIGURE_DPI = 100

_RC = {
    "font.size": 11,
    "axes.labelsize": 12,
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": SAVE_DPI,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
    "savefig.bbox": "tight",
}

# The canonical per-method colours, previously duplicated across `roc_plots`,
# `varchamp_blind_test` and the ablation table. Keys are display names.
METHOD_COLORS = {
    "MutPred-PPI":       "#d7191c",
    "eSIG-Net":          "#fdae61",
    "MINT (seq)":        "#2c7bb6",
    "MINT (site diff)":  "#1a4f7a",
    "PPLM (seq)":        "#9970ab",
    "PPLM (site diff)":  "#762a83",
    "SWING":             "#4dac26",
    "SAAMBE-3D":         "#7b3294",
    "MutPPI":            "#008837",
    "MutPPI+":           "#80cdc1",
    "MutPred2":          "#bf812d",
}

CLASS_LABELS = {
    1: "C1 (both proteins seen)",
    2: "C2 (one protein seen)",
    3: "C3 (neither protein seen)",
}


def apply(agg: bool = True) -> None:
    """Install the shared style. Safe to call more than once.

    `agg=True` selects the non-interactive backend, which every figure script
    needs and twelve of them each set themselves.
    """
    import matplotlib
    if agg:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(_RC)
