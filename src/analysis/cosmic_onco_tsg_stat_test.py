#!/usr/bin/env python3
"""COSMIC Onco/TSG QN vs Edgetic enrichment statistical test.

For each recurrence bin (Single, ≥2, ≥4, ≥8, ≥16, ≥32) in COSMIC Oncogene and
Tumour Suppressor Gene (TSG) variants, tests whether Quasi-null (QN) enrichment
significantly differs from Edgetic (E) enrichment using paired bootstrap distributions.

Method:
  For each bin, per-bootstrap-sample QN and Edgetic enrichments are computed
  (relative to gnomAD baseline). The difference distribution QN_enrich - E_enrich
  is used for a two-tailed test; p-value = 2 * min(mean(diff≤0), mean(diff≥0)).
  Bonferroni correction over 12 tests (6 bins × 2 categories).

Output:
    results/cosmic_stat_test/cosmic_onco_tsg_qn_vs_edgetic.tex

Usage:
    conda run -n ppi python src/analysis/cosmic_onco_tsg_stat_test.py
"""
from __future__ import annotations

import sys
import pickle

import numpy as np

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import REPO_ROOT  # noqa: E402


_PUB = REPO_ROOT
from variant_db_charts import calc_enrichment



# Must match the all-data model used for Fig 5 (weights/MutPred-PPI.pt). The old,
# variant_dbs_classified/ trees hold SF-model predictions; mixing the two
# across panels is what this path previously did.
BOOTSTRAP_PKL = _PUB / "results" / "variant_dbs_all_data" / "all_bootstrap_results.pkl"
CLASSIFIED_DIR = _PUB / "results" / "variant_dbs_all_data" / "cosmic"
OUT_DIR        = _PUB / "results" / "cosmic_stat_test"

BIN_LABELS   = ["Single", r"$\geq$2", r"$\geq$4", r"$\geq$8", r"$\geq$16", r"$\geq$32"]
BIN_KEYS     = ["single", "2+", "4+", "8+", "16+", "32+"]
CATEGORIES   = ["cosmic_onco", "cosmic_tsg"]
CAT_DISPLAY  = {"cosmic_onco": "Oncogene", "cosmic_tsg": "TSG"}
N_BONF       = 12  # 6 bins × 2 categories


def load_n_variants() -> dict[str, list[int]]:
    """Load n_variants per bin from posterior_ls pkl files."""
    ns: dict[str, list[int]] = {}
    for cat_key, cat_display in [("onco", "cosmic_onco"), ("tsg", "cosmic_tsg")]:
        ns[cat_display] = []
        for b in BIN_KEYS:
            fname = CLASSIFIED_DIR / f"cosmic_{cat_key}_{b}_posterior_ls.pkl"
            if fname.exists():
                with open(fname, "rb") as f:
                    d = pickle.load(f)
                ns[cat_display].append(len(d))
            else:
                ns[cat_display].append(0)
    return ns


def bootstrap_enrichment_per_bin(
    obs_list: list[np.ndarray],
    gnomad_boot: np.ndarray,
) -> list[np.ndarray]:
    """Return list of (n_boot, 3) enrichment arrays, one per bin."""
    n_boot = gnomad_boot.shape[0]
    result = []
    for obs_boot in obs_list:
        enr = np.array([calc_enrichment(obs_boot[i], gnomad_boot[i]) for i in range(n_boot)])
        result.append(enr)
    return result


def pval_str(p: float, alpha_bonf: float) -> str:
    if p < alpha_bonf:
        if p < 0.001:
            return r"$<$0.001$^{**}$"
        return f"{p:.3f}$^{{**}}$"
    elif p < 0.05:
        if p < 0.001:
            return r"$<$0.001$^{*}$"
        return f"{p:.3f}$^{{*}}$"
    else:
        if p > 0.99:
            return ">0.99 (n.s.)"
        return f"{p:.3f} (n.s.)"


def fmt_enr(median: float, p16: float, p84: float) -> str:
    sign = "+" if median >= 0 else ""
    return f"{sign}{median:.2f} [{p16:.2f}, {p84:.2f}]"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    alpha_bonf = 0.05 / N_BONF

    print("Loading bootstrap results...", flush=True)
    with open(BOOTSTRAP_PKL, "rb") as f:
        all_boot = pickle.load(f)

    gnomad_boot = all_boot["gnomad"][0]   # shape (n_boot, 3)
    n_boot      = gnomad_boot.shape[0]
    print(f"  n_bootstrap: {n_boot:,}", flush=True)

    print("Loading n_variants per bin...", flush=True)
    n_variants = load_n_variants()

    print("Computing enrichments and tests...", flush=True)
    rows = []
    for cat_key in CATEGORIES:
        obs_list = all_boot[cat_key]   # list of 6 arrays (n_boot, 3)
        enr_list = bootstrap_enrichment_per_bin(obs_list, gnomad_boot)
        ns_list  = n_variants.get(cat_key, [0] * 6)

        for bi, (bin_label, enr_boot) in enumerate(zip(BIN_LABELS, enr_list)):
            qn_boot  = enr_boot[:, 1]   # Quasi-null
            edg_boot = enr_boot[:, 2]   # Edgetic
            diff     = qn_boot - edg_boot

            qn_med  = float(np.median(qn_boot))
            qn_p16  = float(np.percentile(qn_boot, 16))
            qn_p84  = float(np.percentile(qn_boot, 84))
            edg_med = float(np.median(edg_boot))
            edg_p16 = float(np.percentile(edg_boot, 16))
            edg_p84 = float(np.percentile(edg_boot, 84))
            dif_med = float(np.median(diff))
            dif_p16 = float(np.percentile(diff, 16))
            dif_p84 = float(np.percentile(diff, 84))

            p = 2 * min(float(np.mean(diff <= 0)), float(np.mean(diff >= 0)))
            p = min(p, 1.0)

            rows.append({
                "category": cat_key,
                "cat_display": CAT_DISPLAY[cat_key],
                "bin": bin_label,
                "n": ns_list[bi],
                "qn_med": qn_med, "qn_p16": qn_p16, "qn_p84": qn_p84,
                "edg_med": edg_med, "edg_p16": edg_p16, "edg_p84": edg_p84,
                "dif_med": dif_med, "dif_p16": dif_p16, "dif_p84": dif_p84,
                "p": p,
            })
            print(f"  {CAT_DISPLAY[cat_key]} {bin_label.ljust(10)} "
                  f"QN={qn_med:+.3f}  E={edg_med:+.3f}  diff={dif_med:+.3f}  p={p:.4f}",
                  flush=True)

    # Build LaTeX table
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{COSMIC Oncogene and TSG: Quasi-null vs.\ Edgetic enrichment per "
        r"recurrence bin. Enrichment = $(f_\text{obs} - f_\text{gnomAD}) / "
        r"(f_\text{obs} + f_\text{gnomAD})$; values reported as median [16th, 84th pct] "
        r"over 10{,}000 bootstrap samples. "
        r"$^{**}$ Bonferroni-corrected $p < 0.05$ (12 tests); "
        r"$^{*}$ uncorrected $p < 0.05$.}",
        r"\label{tab:cosmic_onco_tsg_stat_test}",
        r"\small",
        r"\begin{tabular}{llrcccr}",
        r"\toprule",
        r"Category & Bin & $n$ & QN Enrichment & Edgetic Enrichment "
        r"& QN $-$ Edgetic & $p$-value \\",
        r"\midrule",
    ]

    prev_cat = None
    for row in rows:
        is_new_cat = row["category"] != prev_cat
        if prev_cat and is_new_cat:
            lines.append(r"\midrule")
        cat_str = row["cat_display"] if is_new_cat else ""
        prev_cat = row["category"]

        lines.append(
            r"{cat} & {bin} & {n:,} & {qn} & {edg} & {diff} & {pval} \\".format(
                cat   = cat_str,
                bin   = row["bin"],
                n     = row["n"],
                qn    = fmt_enr(row["qn_med"], row["qn_p16"], row["qn_p84"]),
                edg   = fmt_enr(row["edg_med"], row["edg_p16"], row["edg_p84"]),
                diff  = fmt_enr(row["dif_med"], row["dif_p16"], row["dif_p84"]),
                pval  = pval_str(row["p"], alpha_bonf),
            )
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out_tex = OUT_DIR / "cosmic_onco_tsg_qn_vs_edgetic.tex"
    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved → {out_tex}", flush=True)


if __name__ == "__main__":
    main()
