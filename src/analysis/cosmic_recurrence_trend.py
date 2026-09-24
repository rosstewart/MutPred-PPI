#!/usr/bin/env python3
"""Does edgetic enrichment rise with COSMIC recurrence?

Two questions, deliberately kept apart because they are not the same claim:

  (A) BIN LEVEL. Across the six recurrence strata drawn in Fig 5
      (Single, >=2, >=4, >=8, >=16, >=32), does edgetic enrichment relative to
      the gnomAD baseline increase with the stratum? Correlation is reported
      against the ordinal rank AND against the raw recurrence cut (1, 2, 4, 8,
      16, 32) and its log2. Spearman is invariant to any monotone transform of
      x, so its value is identical for all three; only Pearson distinguishes
      them, which is exactly why both are printed.

  (B) VARIANT LEVEL, UNBINNED. Using each variant's RAW recurrence count --
      no strata at all -- does recurrence predict being called Edgetic?
      Reported as Spearman rho, as Pearson r on both the raw count and its
      log2, and as a logistic fit giving an odds ratio per unit and per
      doubling. Spearman is identical for raw and log2 (rank invariance), so it
      appears once; Pearson and the logistic slope are what the transform
      changes.

WHY (A)'s p-VALUES ARE NOT REPORTED FROM THE CORRELATION ITSELF
---------------------------------------------------------------
Fig 5's strata are CUMULATIVE and therefore nested: a variant recurring 40
times is in >=2, >=4, >=8, >=16 and >=32 simultaneously. Six such points are
massively dependent, so the textbook Spearman p-value -- which assumes
independent pairs -- is meaningless here. Instead the existing 100,000-replicate
bootstrap is reused: rho is recomputed within each replicate, giving a
distribution whose spread already carries the dependence. The reported interval
is that distribution's 2.5/97.5 percentiles, and the one-sided level is the
fraction of replicates with rho <= 0.

(B) has no such problem: it does not bin, so there is nothing to nest -- one
row per variant, each counted once, against its own recurrence count.

Requires a COSMIC licence (recurrence comes from
`datasets/annotations_licensed/vt_to_tumor_site.pkl`).

    conda run -n ppi python src/analysis/cosmic_recurrence_trend.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.edgotypes import DEFAULT_THRESHOLD, classify
from paths import ANNOTATIONS_LICENSED_DIR, RESULTS_DIR, VARIANT_DBS_DIR

#: Index of the Edgetic column in the bootstrap density arrays
#: (Quasi-wild-type, Quasi-null, Edgetic) -- same order as variant_db_charts.
EDGETIC = 2

#: The recurrence cuts behind Fig 5's COSMIC strata, in stratum order.
#: Recurrence cuts per group. The full COSMIC set is binned to >=256; the
#: oncogene and TSG subsets stop at >=32, where they already hold only ~134 and
#: ~113 variants, so the cuts are NOT global.
CUTS_ALL = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CUTS_SUBSET = (1, 2, 4, 8, 16, 32)

#: Minimum scored partners for the restricted variant-level rows. Edgetic means
#: "some partners disrupted, some preserved", so one partner can never be
#: Edgetic and two decide it on a single disagreement. Requiring three makes the
#: call a property of the variant rather than of how well it was covered.
MIN_PARTNERS = 3

#: bootstrap key -> (per-variant file stems, display name)
GROUPS = {
    "cosmic":      ("cosmic",      "COSMIC (all)",     CUTS_ALL),
    "cosmic_onco": ("cosmic_onco", "COSMIC oncogenes", CUTS_SUBSET),
    "cosmic_tsg":  ("cosmic_tsg",  "COSMIC TSGs",      CUTS_SUBSET),
}

OUT_DIR = RESULTS_DIR / "cosmic_recurrence_trend"


def calc_enrichment(f_obs, f_base):
    """The Fig 5 statistic, bounded in [-1, 1]. Same formula as variant_db_charts."""
    return (f_obs - f_base) / (f_obs + f_base)


# ── (A) bin level ─────────────────────────────────────────────────────────────

def bin_level(bootstrap: dict, key: str, cuts: tuple) -> dict:
    """Correlation of edgetic enrichment with recurrence stratum, with a
    bootstrap interval on rho that respects the strata's nesting."""
    baseline = bootstrap["gnomad"][0]                 # (n_boot, 3)
    strata = bootstrap[key]                           # 6 x (n_boot, 3)
    if len(strata) != len(cuts):
        raise ValueError(f"{key}: {len(strata)} strata, expected {len(cuts)} "
                         f"({cuts})")

    # (n_boot, 6) edgetic enrichment, replicate-aligned across strata
    enr = np.column_stack([
        calc_enrichment(s[:, EDGETIC], baseline[:, EDGETIC]) for s in strata])

    point = enr.mean(axis=0)
    rank = np.arange(1, len(cuts) + 1)

    # rho within each replicate: the spread carries the nesting dependence
    rhos = np.array([spearmanr(rank, enr[i]).statistic for i in range(enr.shape[0])])
    rhos = rhos[np.isfinite(rhos)]

    sp = spearmanr(rank, point)
    return {
        "enrichment": point,
        "rho_point": float(sp.statistic),
        # Nominal p, assuming six INDEPENDENT points. The strata are nested, so
        # this is anti-conservative; `p_one_sided` below is the honest figure.
        "rho_p_nominal": float(sp.pvalue),
        "rho_median": float(np.median(rhos)),
        "rho_lo": float(np.percentile(rhos, 2.5)),
        "rho_hi": float(np.percentile(rhos, 97.5)),
        "p_one_sided": float(np.mean(rhos <= 0)),
        "pearson_rank": pearsonr(rank, point).statistic,
        "pearson_raw": pearsonr(np.array(cuts, float), point).statistic,
        "pearson_log2": pearsonr(np.log2(cuts), point).statistic,
        "n_boot": int(enr.shape[0]),
    }


# ── (B) variant level ─────────────────────────────────────────────────────────

def load_recurrence() -> dict[str, int]:
    """(uniprot, variant) -> recurrence count.

    `vt_to_tumor_site.pkl` holds one entry PER OCCURRENCE, so the count is the
    list length -- not the number of distinct tissues. Same reading as
    extract_variant_db_stats.py.
    """
    path = ANNOTATIONS_LICENSED_DIR / "vt_to_tumor_site.pkl"
    if not path.exists():
        sys.exit(f"{path} not found -- this analysis needs a COSMIC licence.")
    with open(path, "rb") as fh:
        return {k: len(v) for k, v in pickle.load(fh).items()}


def _variant_table(stem: str, recurrence: dict[str, int], threshold: float):
    """(recurrence, is_edgetic, n_partners) arrays, one entry per variant.

    The widest stratum on disk (`{stem}_2+`) is unioned with the singletons so
    that every recurrence level is represented; each variant appears once.
    """
    frames = []
    for name in (f"{stem}_single", f"{stem}_2+"):
        p = VARIANT_DBS_DIR / "cosmic" / f"{name}.csv.gz"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["uniprot", "variant", "partner"])

    rows = []
    for (u, v), g in df.groupby(["uniprot", "variant"], sort=False):
        n = recurrence.get(f"{u} {v}")
        if n is None:
            continue
        rows.append((n, classify(g["score"].to_numpy(), threshold) == "Edgetic",
                     len(g)))
    if len(rows) < 3:
        return None

    return (np.array([r[0] for r in rows], float),
            np.array([r[1] for r in rows], float),
            np.array([r[2] for r in rows]))


def variant_level(stem: str, recurrence: dict[str, int],
                  threshold: float) -> dict | None:
    """Recurrence vs Edgetic, UNBINNED: one row per variant, raw counts.

    Three framings of the same data, because they answer different questions:

      Spearman rho  rank association. Invariant to any monotone transform, so
                    the raw-count and log2-count values are IDENTICAL -- it is
                    reported once rather than twice to avoid implying otherwise.
      Pearson r     linear association. This one DOES change under log2, which
                    is the point of computing both: recurrence is heavily
                    right-skewed, so the raw-count r is dominated by a handful
                    of very high counts while the log2 r reflects fold-changes.
      logistic      P(Edgetic) modelled directly on the continuous predictor.
                    The log2 slope is the most interpretable number here: an
                    odds ratio per DOUBLING of recurrence.

    Every statistic is also recomputed on variants with at least `MIN_PARTNERS`
    scored partners. A single-partner variant cannot be Edgetic by
    construction, and a two-partner one is called Edgetic on a single
    disagreement, so including them makes recurrence partly a proxy for partner
    coverage rather than for edgotype.
    """
    table = _variant_table(stem, recurrence, threshold)
    if table is None:
        return None
    rec, edge, n_partners = table
    out = {}

    for tag, mask in (("all", np.ones(len(rec), bool)),
                      ("multi", n_partners >= MIN_PARTNERS)):
        r, e = rec[mask], edge[mask]
        if len(r) < 3 or e.std() == 0:
            continue
        lg = np.log2(r)
        # How far the "per doubling" odds ratio is being asked to reach. The
        # full span is wide but the bulk is not, so both are recorded: an OR
        # compounded over the full range is carried by a thin tail.
        out[f"doublings_span_{tag}"] = float(lg.max() - lg.min())
        out[f"doublings_iqr_{tag}"] = float(
            np.percentile(lg, 75) - np.percentile(lg, 25))
        out[f"frac_le3_{tag}"] = float(np.mean(r <= 3))
        sp = spearmanr(r, e)
        out[f"n_{tag}"] = int(len(r))
        out[f"n_edgetic_{tag}"] = int(e.sum())
        out[f"spearman_{tag}"] = float(sp.statistic)
        out[f"spearman_p_{tag}"] = float(sp.pvalue)
        pr_raw, pr_log = pearsonr(r, e), pearsonr(lg, e)
        out[f"pearson_raw_{tag}"] = float(pr_raw.statistic)
        out[f"pearson_raw_p_{tag}"] = float(pr_raw.pvalue)
        out[f"pearson_log2_{tag}"] = float(pr_log.statistic)
        out[f"pearson_log2_p_{tag}"] = float(pr_log.pvalue)
        for xname, x in (("raw", r), ("log2", lg)):
            fit = _logit(x, e)
            if fit is not None:
                out[f"or_{xname}_{tag}"], out[f"or_{xname}_lo_{tag}"], \
                    out[f"or_{xname}_hi_{tag}"], out[f"or_{xname}_p_{tag}"] = fit
    return out or None


def _logit(x: np.ndarray, y: np.ndarray):
    """(odds ratio, lo, hi, p) for a one-predictor logistic fit, or None.

    Returned as an odds ratio per unit of `x`, so with x = log2(recurrence) it
    reads as the multiplicative change in the odds of being Edgetic per
    doubling of recurrence.
    """
    import statsmodels.api as sm
    try:
        res = sm.Logit(y, sm.add_constant(x, has_constant="add")).fit(disp=0)
    except Exception:                                          # noqa: BLE001
        return None
    beta = float(res.params[1])
    lo, hi = (float(v) for v in res.conf_int()[1])
    return float(np.exp(beta)), float(np.exp(lo)), float(np.exp(hi)), float(res.pvalues[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(VARIANT_DBS_DIR))
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    pkl = Path(args.data_dir) / "all_bootstrap_results.pkl"
    if not pkl.exists():
        sys.exit(f"{pkl} not found -- run variant_db_charts.py --edgotype-bootstrap first.")
    with open(pkl, "rb") as fh:
        bootstrap = pickle.load(fh)

    recurrence = load_recurrence()
    out_rows: list[dict] = []
    bins: dict[str, dict] = {}
    variants: dict[str, dict] = {}

    print("(A) Edgetic enrichment vs recurrence stratum "
          "(cumulative bins, as in Fig 5)\n")
    for key, (stem, label, cuts) in GROUPS.items():
        if key not in bootstrap:
            print(f"  {label}: absent from the bootstrap cache, skipping")
            continue
        r = bin_level(bootstrap, key, cuts)
        bins[key] = r
        print(f"  {label}")
        print("     enrichment by stratum: " +
              "  ".join(f"{c}:{e:+.3f}" for c, e in zip(cuts, r["enrichment"])))
        print(f"     Spearman rho = {r['rho_point']:+.3f}   "
              f"bootstrap median {r['rho_median']:+.3f} "
              f"[{r['rho_lo']:+.3f}, {r['rho_hi']:+.3f}]   "
              f"P(rho<=0) = {r['p_one_sided']:.4f}   (n_boot={r['n_boot']:,})")
        print(f"     Pearson r: vs rank {r['pearson_rank']:+.3f}   "
              f"vs raw cut {r['pearson_raw']:+.3f}   "
              f"vs log2(cut) {r['pearson_log2']:+.3f}")
        out_rows.append({"group": label, "level": "bin", **
                         {k: v for k, v in r.items() if k != "enrichment"}})
        print()

    print("(B) Raw recurrence count vs Edgetic, unbinned, one row per variant\n")
    for key, (stem, label, _cuts) in GROUPS.items():
        r = variant_level(stem, recurrence, args.threshold)
        if r is None:
            print(f"  {label}: too few variants with recurrence, skipping")
            continue
        variants[key] = r
        print(f"  {label}")
        for tag, note in (("all", "all variants"),
                          ("multi", f">={MIN_PARTNERS} scored partners")):
            if f"n_{tag}" not in r:
                continue
            print(f"     {note} ({r[f'n_{tag}']:,}; "
                  f"{r[f'n_edgetic_{tag}']:,} Edgetic)")
            print(f"        Spearman rho      {r[f'spearman_{tag}']:+.4f}"
                  f"   p = {r[f'spearman_p_{tag}']:.3g}   (raw and log2 identical)")
            print(f"        Pearson r  raw    {r[f'pearson_raw_{tag}']:+.4f}"
                  f"   p = {r[f'pearson_raw_p_{tag}']:.3g}")
            print(f"        Pearson r  log2   {r[f'pearson_log2_{tag}']:+.4f}"
                  f"   p = {r[f'pearson_log2_p_{tag}']:.3g}")
            if f"or_raw_{tag}" in r:
                print(f"        OR per +1 count   {r[f'or_raw_{tag}']:.4f} "
                      f"[{r[f'or_raw_lo_{tag}']:.4f}, {r[f'or_raw_hi_{tag}']:.4f}]"
                      f"   p = {r[f'or_raw_p_{tag}']:.3g}")
            if f"or_log2_{tag}" in r:
                print(f"        OR per doubling   {r[f'or_log2_{tag}']:.4f} "
                      f"[{r[f'or_log2_lo_{tag}']:.4f}, {r[f'or_log2_hi_{tag}']:.4f}]"
                      f"   p = {r[f'or_log2_p_{tag}']:.3g}")
                print(f"        counts span {r[f'doublings_span_{tag}']:.1f} doublings "
                      f"(IQR {r[f'doublings_iqr_{tag}']:.2f}; "
                      f"{100 * r[f'frac_le3_{tag}']:.0f}% recur <=3 times)")
        out_rows.append({"group": label, "level": "variant", **r})
        print()

    print("=" * 78)
    print("SUMMARY\n")
    print(f"{'group':18s} {'partners':13s} {'binned rho':>10s} {'p(nom)':>9s} "
          f"{'P(rho<=0)':>10s} {'OR/doubling':>12s} {'p':>10s} "
          f"{'r(log2)':>9s} {'p':>10s}")
    for key, (stem, label, _cuts) in GROUPS.items():
        b = bins.get(key)
        v = variants.get(key)
        if b is None or v is None:
            continue
        for tag, note in (("all", "all"),
                          ("multi", f">={MIN_PARTNERS} partners")):
            if f"or_log2_{tag}" not in v:
                continue
            print(f"{label:18s} {note:13s} {b['rho_point']:+10.3f} "
                  f"{b['rho_p_nominal']:9.4f} {b['p_one_sided']:10.4f} "
                  f"{v[f'or_log2_{tag}']:12.3f} {v[f'or_log2_p_{tag}']:10.3g} "
                  f"{v[f'pearson_log2_{tag}']:+9.4f} "
                  f"{v[f'pearson_log2_p_{tag}']:10.3g}")
    any_v = next(iter(variants.values()), {})
    if "doublings_span_all" in any_v:
        print(f"\n  Binned rho is over the six cumulative Fig 5 strata; p(nom) assumes")
        print(f"  six independent points and the strata are nested, so P(rho<=0) from")
        print(f"  the {bins[next(iter(bins))]['n_boot']:,}-replicate bootstrap is the figure to quote.")
        print(f"  OR/doubling and r(log2) are unbinned, one row per variant.")
    print("=" * 78 + "\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "cosmic_recurrence_trend.tsv"
    pd.DataFrame(out_rows).to_csv(dest, sep="\t", index=False)
    print(f"Saved: {dest}")


if __name__ == "__main__":
    main()
