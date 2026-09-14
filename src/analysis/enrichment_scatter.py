#!/usr/bin/env python3
"""Quasi-null against edgetic enrichment, one point per variant sample.

The main enrichment figure shows these two as separate stacked layers. Plotting
them against each other puts the mechanism on one pair of axes: to the right is
loss of function (the variant disrupts every partner), upward is rewiring (it
disrupts some and preserves others), and the lower left is no perturbation.

Both axes use the same statistic and the same bootstrap as the main figure, read
from the cache it writes:

    results/variant_dbs_all_data/all_bootstrap_results.pkl

so this adds no computation and cannot disagree with the figure it summarises.
Run `variant_db_charts.py --edgotype-bootstrap` first if the cache is absent.

This is a presentation figure, not a manuscript one. It is sized and weighted to
be read from across a room: large markers, large labels, no title.

    conda run -n ppi python src/analysis/enrichment_scatter.py
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from analysis import plot_style
from analysis.plot_style import SAVE_DPI
plot_style.apply()

import matplotlib.patheffects as pe   # noqa: E402
import matplotlib.pyplot as plt       # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
import numpy as np                    # noqa: E402

from paths import RESULTS_DIR         # noqa: E402

#: Column order of the bootstrap densities: quasi-wild-type, quasi-null, edgetic.
QWT, QN, EDGETIC = 0, 1, 2

DEFAULT_CACHE = RESULTS_DIR / "variant_dbs_all_data" / "all_bootstrap_results.pkl"
DEFAULT_OUT = RESULTS_DIR / "variant_dbs_all_data" / "enrichment_scatter.png"

#: (label, database key, subgroup index, colour). Colours match the main figure.
DISEASE = [
    ("Rare benign",    "clinvar", 0, "#0D47A1"),
    ("Benign",         "clinvar", 1, "#1976D2"),
    ("Pathogenic",     "clinvar", 2, "#C62828"),
    ("Pathogenic AR",  "clinvar", 4, "#00695C"),
    ("Pathogenic AD",  "clinvar", 5, "#6A1B9A"),
    ("Cancer drivers", "cosmic",  5, "#6A0000"),
]
#: gnomAD allele-frequency bins, rarest first, drawn as a trajectory.
GNOMAD_KEY = "gnomad_af"


def enrichment(obs: np.ndarray, base: np.ndarray, idx: int) -> np.ndarray:
    """Per-replicate enrichment of one component against the paired background."""
    f_o, f_b = obs[:, idx], base[:, idx]
    total = f_o + f_b
    return np.divide(f_o - f_b, total, out=np.zeros_like(total), where=total > 0)


def point(obs: np.ndarray, base: np.ndarray) -> tuple[float, float]:
    """(quasi-null enrichment, edgetic enrichment), as bootstrap medians."""
    return (float(np.median(enrichment(obs, base, QN))),
            float(np.median(enrichment(obs, base, EDGETIC))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE),
                    help="bootstrap cache written by variant_db_charts.py")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--demo-tier", action="store_true",
                    help="mark the figure as coming from the Sahni+Fragoza model")
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.exists():
        print(f"{cache} not found. Run:\n"
              f"  python src/analysis/variant_db_charts.py --data-dir "
              f"{cache.parent} --edgotype-bootstrap")
        return 1

    with open(cache, "rb") as fh:
        boot = pickle.load(fh)
    base = boot["gnomad"][0]

    disease, missing = [], []
    for label, db, idx, colour in DISEASE:
        if db not in boot or idx >= len(boot[db]):
            missing.append(label)
            continue
        disease.append((label, colour, point(boot[db][idx], base)))
    gnomad = [point(a, base) for a in boot.get(GNOMAD_KEY, [])]

    if missing:
        print(f"[skip] {len(missing)} group(s) absent from the cache: "
              f"{', '.join(missing)}", flush=True)
    if not disease:
        print("no disease groups in the cache; nothing to draw")
        return 1

    # Sized for a projector: the whole point of this figure is being legible at
    # a distance, so markers, labels and axis text are all well above the
    # defaults used for the manuscript figures.
    MARKER, GMARKER = 420, 150
    LABEL_FS, AXIS_FS, TICK_FS, LEGEND_FS = 20, 24, 18, 17

    fig, ax = plt.subplots(figsize=(14, 11))
    xs = [p[0] for p in [d[2] for d in disease] + gnomad]
    ys = [p[1] for p in [d[2] for d in disease] + gnomad]
    padx = 0.22 * max(abs(min(xs)), abs(max(xs)), 0.05)
    pady = 0.22 * max(abs(min(ys)), abs(max(ys)), 0.05)
    XL, XR = min(xs) - padx, max(xs) + padx
    YB, YT = min(ys) - pady, max(ys) + pady
    ax.set_xlim(XL, XR)
    ax.set_ylim(YB, YT)

    # Three quadrants imply some perturbation; only the lower left implies none.
    shade = dict(color="#E53935", alpha=0.055, zorder=0)
    ax.fill_between([0, XR], 0, YT, **shade)
    ax.fill_between([XL, 0], 0, YT, **shade)
    ax.fill_between([0, XR], YB, 0, **shade)
    ax.axhline(0, lw=1.6, ls="--", alpha=0.45, color="#555555", zorder=1)
    ax.axvline(0, lw=1.6, ls="--", alpha=0.45, color="#555555", zorder=1)

    stroke = [pe.withStroke(linewidth=3.5, foreground="white")]

    if gnomad:
        gx = [p[0] for p in gnomad]
        gy = [p[1] for p in gnomad]
        ax.plot(gx, gy, color="#bbbbbb", lw=1.8, ls=":", zorder=2, alpha=0.8)
        ax.scatter(gx, gy, s=GMARKER, color="#aaaaaa", edgecolors="#888888",
                   linewidths=1.0, alpha=0.75, zorder=3)
        if len(gnomad) > 1:
            ax.annotate("", xy=(gx[0], gy[0]), xytext=(gx[1], gy[1]),
                        arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.8,
                                        mutation_scale=18), zorder=3)

    for label, colour, (px, py) in disease:
        ax.scatter(px, py, s=MARKER, color=colour, zorder=5,
                   edgecolors="white", linewidths=2.0)

    # Place labels in the first of eight directions that misses those already
    # placed, so no label is dropped and none overlaps another.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Labels must miss three things: each other, the plotted markers, and the
    # edge of the axes. Avoiding only other labels left "Benign" sitting on the
    # gnomAD trajectory and pushed "Pathogenic AR" off the right-hand side.
    marker_boxes = []
    for px, py in [d[2] for d in disease] + gnomad:
        x0, y0 = ax.transData.transform((px, py))
        r = 26
        marker_boxes.append(Bbox.from_bounds(x0 - r, y0 - r, 2 * r, 2 * r))
    axes_box = ax.get_window_extent(renderer=renderer)

    to_place = list(disease)
    if gnomad:
        to_place += [("rare gnomAD", "#888888", gnomad[0]),
                     ("common gnomAD", "#888888", gnomad[-1])]

    placed = []
    CANDIDATES = [(18, 16), (-18, 16), (18, -18), (-18, -20), (28, 0), (-28, 0),
                  (0, 28), (0, -30), (34, 18), (-34, 18), (34, -20), (-34, -20)]
    for label, colour, (px, py) in sorted(to_place,
                                          key=lambda d: -(d[2][0] ** 2 + d[2][1] ** 2)):
        italic = label.endswith("gnomAD")
        best = None
        for dx, dy in CANDIDATES:
            t = ax.annotate(label, (px, py), xytext=(dx, dy),
                            textcoords="offset points",
                            fontsize=LEGEND_FS - 2 if italic else LABEL_FS,
                            color=colour,
                            fontweight="normal" if italic else "bold",
                            style="italic" if italic else "normal",
                            ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                            va="bottom" if dy > 0 else ("top" if dy < 0 else "center"),
                            zorder=6)
            bb = t.get_window_extent(renderer=renderer)
            inside = (bb.x0 >= axes_box.x0 and bb.x1 <= axes_box.x1
                      and bb.y0 >= axes_box.y0 and bb.y1 <= axes_box.y1)
            clear = not any(bb.overlaps(o) for o in placed + marker_boxes)
            if inside and clear:
                best = (t, bb)
                break
            t.remove()
        if best is None:
            # Nothing fits cleanly: take the first candidate that at least stays
            # inside the axes, so a label is never lost off the edge.
            for dx, dy in CANDIDATES:
                t = ax.annotate(label, (px, py), xytext=(dx, dy),
                                textcoords="offset points",
                                fontsize=LEGEND_FS - 2 if italic else LABEL_FS,
                                color=colour,
                                fontweight="normal" if italic else "bold",
                                style="italic" if italic else "normal",
                                ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                                va="bottom" if dy > 0 else ("top" if dy < 0 else "center"),
                                zorder=6)
                bb = t.get_window_extent(renderer=renderer)
                if (bb.x0 >= axes_box.x0 and bb.x1 <= axes_box.x1
                        and bb.y0 >= axes_box.y0 and bb.y1 <= axes_box.y1):
                    best = (t, bb)
                    break
                t.remove()
        if best is None:
            continue
        best[0].set_path_effects(stroke)
        placed.append(best[1])

    ax.set_xlabel("Quasi-null enrichment  (relative to gnomAD)",
                  fontsize=AXIS_FS, fontweight="bold", labelpad=12)
    ax.set_ylabel("Edgetic enrichment  (relative to gnomAD)",
                  fontsize=AXIS_FS, fontweight="bold", labelpad=12)
    ax.tick_params(labelsize=TICK_FS)

    soft = dict(color="#bbbbbb", style="italic", fontsize=LEGEND_FS, zorder=3)
    soft_stroke = [pe.withStroke(linewidth=4, foreground="white")]
    for t in (ax.text(0.0, YT, " ↑ network rewiring", ha="center", va="top", **soft),
              ax.text(XR, 0.0, "loss of function → ", ha="right", va="bottom", **soft),
              ax.text(XL, YB, " no perturbation", ha="left", va="bottom", **soft)):
        t.set_path_effects(soft_stroke)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, lw=0.9)

    if args.demo_tier:
        plot_style.demo_stamp(fig)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
