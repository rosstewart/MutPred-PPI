#!/usr/bin/env python3
"""Compute S-Table 1 (variant repository statistics) for MutPred-PPI paper.

Reads the prediction TSVs and classification source files to compute per-group
statistics: Proteins, Pairs, Variants, Triplets, Mean Partners.

Variants the model was trained on are excluded, exactly as they are for every
enrichment figure (see analysis/training_overlap.py). The table describes the
data the paper actually analyses, so its counts match the sample sizes printed
on Fig 5, S8, S9 and the stability figure; reporting the unfiltered inventory
here would put a different n next to the same group name in two places.

Writes figures/variant_db_stats_table.tex as a drop-in tabular block.
"""
import pickle
import pandas as pd

# --- repo-relative path resolution (see src/paths.py) ---
from paths import ANNOTATIONS_DIR, ANNOTATIONS_LICENSED_DIR, DATA_ROOT, REPO_ROOT  # noqa: E402
from analysis import training_overlap  # noqa: E402
from utils.legacy_guard import LegacyInputError  # noqa: E402


_PUB = REPO_ROOT
_BASE = DATA_ROOT
_HOME = _BASE / "home"
_OUT  = _PUB / "figures" / "variant_db_stats_table.tex"

PRED_DIR = _PUB / "results" / "variant_dbs_all_data"


def prediction_tsv(db):
    """One database's prediction TSV, in either supported layout.

    Inference writes `{DATA_ROOT}/{db}/mutpred_ppi_predictions.tsv`; the
    reproduction notebook collects them as `{db}_mutpred_ppi_predictions.tsv`.
    """
    collected = PRED_DIR / f"{db}_mutpred_ppi_predictions.tsv"
    if collected.exists():
        return collected
    return DATA_ROOT / db / "mutpred_ppi_predictions.tsv"

# Classification source files
CLINVAR_SUBSETS = {
    "pathogenic": ANNOTATIONS_DIR / "clinvar" / "pathogenic_dirbind_variant_subset.pkl",
    "benign":     ANNOTATIONS_DIR / "clinvar" / "benign_dirbind_variant_subset.pkl",
    "vus":        ANNOTATIONS_DIR / "clinvar" / "vus_dirbind_variant_subset.pkl",
}
BENIGN_AF_FILE = ANNOTATIONS_DIR / "benign_allele_frequencies.tsv"
RARE_BENIGN_THRESHOLD = 0.01

GNOMAD_AF_FILE   = ANNOTATIONS_DIR / "gnomad_allele_frequencies.tsv"
VT_TO_TUMOR_SITE = ANNOTATIONS_LICENSED_DIR / "vt_to_tumor_site.pkl"
ONCO_TSG_FILE    = ANNOTATIONS_LICENSED_DIR / "onco_tsg_dict.pkl"
NEURODEV_LABELS  = ANNOTATIONS_DIR / "neurodev" / "variant_label_dict.pkl"
HGMD_SUBSET      = ANNOTATIONS_LICENSED_DIR / "hgmd_variant_subset.pkl"
AR_AD_FILE       = ANNOTATIONS_DIR / "clingen_ar_ad_uniprot_sets.pkl"


def parse_preds(tsv_path) -> pd.DataFrame:
    """Load a predictions TSV as interactor / partner / variant / score.

    Current files carry the two accessions in separate columns. This used to
    split a welded `complex_id` on its first underscore, which raises on every
    current file and is wrong for any accession containing the delimiter.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    if "complex_id" in df.columns:
        raise LegacyInputError(
            f"{tsv_path} uses the retired complex_id/variant/score schema; "
            f"re-score the database rather than parsing it.")
    df = df.rename(columns={"mutation": "variant"})
    return _drop_training_overlap(df, tsv_path)


def _drop_training_overlap(df: pd.DataFrame, tsv_path) -> pd.DataFrame:
    """Exclude variants the model was trained on, as the figures do."""
    known = training_overlap.training_variants_or_none()
    if known is None:
        print(f"  [warn] {tsv_path}: training table unavailable, so trained-on "
              f"variants are NOT excluded -- these counts will not match the "
              f"sample sizes on the enrichment figures", flush=True)
        return df
    keep = [(i, v) not in known
            for i, v in zip(df["interactor"].astype(str), df["variant"].astype(str))]
    n = len(df) - sum(keep)
    if n:
        print(f"  training-overlap filter [{tsv_path.name if hasattr(tsv_path, 'name') else tsv_path}]: "
              f"dropped {n:,} of {len(df):,} rows ({100 * n / len(df):.2f}%)", flush=True)
    return df.loc[keep].reset_index(drop=True)


def stats(df: pd.DataFrame, subset=None) -> dict:
    """Compute Proteins/Pairs/Variants/Triplets/MeanPartners for a filtered df.

    subset: optional set of (interactor, variant, partner) tuples to filter to.
    """
    if subset is not None:
        mask = df.apply(lambda r: (r.interactor, r.variant, r.partner) in subset, axis=1)
        df = df[mask]
    if len(df) == 0:
        return dict(proteins=0, pairs=0, variants=0, triplets=0, mean_partners=0.0)
    # Every protein the group touches, on either side of a pair -- matching
    # `generate_training_table.py`. Counting only interactors made the same
    # column mean different things in Table 1 and Table S1, and understated a
    # group whose variants sit in few proteins with many distinct partners.
    n_proteins = len(set(df["interactor"]) | set(df["partner"]))
    n_pairs    = df.groupby(["interactor", "partner"]).ngroups
    n_variants = df.groupby(["interactor", "variant"]).ngroups
    n_triplets = len(df)
    mean_partners = df.groupby(["interactor", "variant"])["partner"].count().mean()
    return dict(proteins=n_proteins, pairs=n_pairs, variants=n_variants,
                triplets=n_triplets, mean_partners=round(mean_partners, 1))


def fmt(n, decimals=None) -> str:
    if n is None or (isinstance(n, float) and n != n):
        return r"-"
    if decimals is not None:
        return f"{n:.{decimals}f}"
    return f"{int(n):,}".replace(",", "{,}")


def row(label: str, s: dict) -> str:
    return (rf"\quad {label} & {fmt(s['proteins'])} & {fmt(s['pairs'])} & "
            rf"{fmt(s['variants'])} & {fmt(s['triplets'])} & {fmt(s['mean_partners'], 1)} \\")


def main() -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        (r"\textbf{Dataset} & \textbf{Proteins} & \textbf{Pairs} & "
         r"\textbf{Variants} & \textbf{Triplets} & \textbf{Mean Partners} \\"),
        r"\hline",
    ]

    # ── ClinVar ──────────────────────────────────────────────────────────────
    print("Processing ClinVar...", flush=True)
    cv_df = parse_preds(prediction_tsv("clinvar"))
    subsets = {k: pickle.load(open(v, "rb")) for k, v in CLINVAR_SUBSETS.items()}
    rare_benign_set = set()
    if BENIGN_AF_FILE.exists():
        af = {}
        with open(BENIGN_AF_FILE) as f:
            for ln in f:
                parts = ln.strip().split("\t")
                if len(parts) >= 2:
                    af[parts[0]] = float(parts[1])
        for (u, v, p) in subsets["benign"]:
            if af.get(f"{u} {v}", 1.0) <= RARE_BENIGN_THRESHOLD:
                rare_benign_set.add((u, v, p))
    else:
        print("  WARNING: benign AF file not found — rare_benign empty", flush=True)

    ar_uniprots, ad_uniprots = set(), set()
    if AR_AD_FILE.exists():
        ar_ad = pickle.load(open(AR_AD_FILE, "rb"))
        ar_uniprots, ad_uniprots = ar_ad.get("AR", set()), ar_ad.get("AD", set())
    else:
        print(f"  WARNING: {AR_AD_FILE} not found — AR/AD rows will be empty "
              "(run build_ar_ad_gene_sets.py first)", flush=True)
    ar_pathogenic_set = {(u, v, p) for (u, v, p) in subsets["pathogenic"] if u in ar_uniprots}
    ad_pathogenic_set = {(u, v, p) for (u, v, p) in subsets["pathogenic"] if u in ad_uniprots}

    lines += [r"\multicolumn{6}{l}{\textit{ClinVar}} \\"]
    lines.append(row("Rare Benign",  stats(cv_df, rare_benign_set)))
    lines.append(row("Benign",       stats(cv_df, subsets["benign"])))
    lines.append(row("Pathogenic",   stats(cv_df, subsets["pathogenic"])))
    lines.append(row("VUS",          stats(cv_df, subsets["vus"])))
    lines.append(row("Pathogenic AR", stats(cv_df, ar_pathogenic_set)))
    lines.append(row("Pathogenic AD", stats(cv_df, ad_pathogenic_set)))
    lines.append(r"\hline")

    # ── COSMIC ───────────────────────────────────────────────────────────────
    print("Processing COSMIC...", flush=True)
    cos_df = parse_preds(prediction_tsv("cosmic"))
    # Despite the file name, the list holds one entry PER OCCURRENCE (a site repeats once per sample), so len() is the recurrence -- the number of times the variant was observed, NOT the number of distinct tissues.
    vt_to_sites = pickle.load(open(VT_TO_TUMOR_SITE, "rb"))
    onco_tsg    = pickle.load(open(ONCO_TSG_FILE, "rb"))
    onco_vts    = onco_tsg["oncogene"]
    tsg_vts     = onco_tsg["TSG"]

    def recurrence(r) -> int:
        return len(vt_to_sites.get(f"{r.interactor} {r.variant}", []))

    cos_df = cos_df.copy()
    cos_df["recurrence"] = cos_df.apply(recurrence, axis=1)
    cos_df["is_onco"]    = cos_df.apply(lambda r: f"{r.interactor} {r.variant}" in onco_vts, axis=1)
    cos_df["is_tsg"]     = cos_df.apply(lambda r: f"{r.interactor} {r.variant}" in tsg_vts, axis=1)

    thresholds = [1, 2, 4, 8, 16, 32]
    labels_rec = ["Single-occurrence", r"Recurrence $\geq$ 2", r"Recurrence $\geq$ 4",
                  r"Recurrence $\geq$ 8", r"Recurrence $\geq$ 16", r"Recurrence $\geq$ 32"]

    lines += [r"\multicolumn{6}{l}{\textit{COSMIC}} \\"]
    for thr, lbl in zip(thresholds, labels_rec):
        sub = cos_df[cos_df["recurrence"] == thr] if thr == 1 else cos_df[cos_df["recurrence"] >= thr]
        lines.append(row(lbl, stats(sub)))
    lines.append(r"\hline")

    lines += [r"\multicolumn{6}{l}{\textit{COSMIC (Oncogenes)}} \\"]
    onco_df = cos_df[cos_df["is_onco"]]
    for thr, lbl in zip(thresholds, labels_rec):
        sub = onco_df[onco_df["recurrence"] == thr] if thr == 1 else onco_df[onco_df["recurrence"] >= thr]
        lines.append(row(lbl, stats(sub)))
    lines.append(r"\hline")

    lines += [r"\multicolumn{6}{l}{\textit{COSMIC (Tumor Suppressor Genes)}} \\"]
    tsg_df = cos_df[cos_df["is_tsg"]]
    for thr, lbl in zip(thresholds, labels_rec):
        sub = tsg_df[tsg_df["recurrence"] == thr] if thr == 1 else tsg_df[tsg_df["recurrence"] >= thr]
        lines.append(row(lbl, stats(sub)))
    lines.append(r"\hline")

    # ── HGMD ─────────────────────────────────────────────────────────────────
    print("Processing HGMD...", flush=True)
    hgmd_tsv = prediction_tsv("hgmd")
    if hgmd_tsv.exists():
        hgmd_df = parse_preds(hgmd_tsv)
        lines += [r"\multicolumn{6}{l}{\textit{HGMD}} \\"]
        lines.append(row("All", stats(hgmd_df)))
        if HGMD_SUBSET.exists():
            hgmd_subset = pickle.load(open(HGMD_SUBSET, "rb"))
            ar_hgmd_set = {(u, v, p) for (u, v, p) in hgmd_subset if u in ar_uniprots}
            ad_hgmd_set = {(u, v, p) for (u, v, p) in hgmd_subset if u in ad_uniprots}
            lines.append(row("AR", stats(hgmd_df, ar_hgmd_set)))
            lines.append(row("AD", stats(hgmd_df, ad_hgmd_set)))
        lines.append(r"\hline")

    # ── gnomAD ────────────────────────────────────────────────────────────────
    print("Processing gnomAD...", flush=True)
    gn_df = parse_preds(prediction_tsv("gnomad"))
    af_dict = {}
    with open(GNOMAD_AF_FILE) as f:
        for ln in f:
            parts = ln.strip().split("\t")
            if len(parts) >= 2:
                af_dict[parts[0]] = float(parts[1])
    gn_df = gn_df.copy()
    gn_df["af"] = gn_df.apply(lambda r: af_dict.get(f"{r.interactor} {r.variant}", None), axis=1)

    lines += [r"\multicolumn{6}{l}{\textit{gnomAD}} \\"]
    lines.append(row("All", stats(gn_df)))
    gn_af = gn_df.dropna(subset=["af"])
    bins = [(None, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, None)]
    bin_labels = [
        r"AF $\leq$ 1e-6",
        r"1e-6 $<$ AF $\leq$ 1e-5",
        r"1e-5 $<$ AF $\leq$ 1e-4",
        r"1e-4 $<$ AF $\leq$ 1e-3",
        r"1e-3 $<$ AF $\leq$ 1e-2",
        r"1e-2 $<$ AF",
    ]
    for (lo, hi), lbl in zip(bins, bin_labels):
        if lo is None:
            sub = gn_af[gn_af["af"] <= hi]
        elif hi is None:
            sub = gn_af[gn_af["af"] > lo]
        else:
            sub = gn_af[(gn_af["af"] > lo) & (gn_af["af"] <= hi)]
        lines.append(row(lbl, stats(sub)))
    lines.append(r"\hline")

    # ── NDD & ASD ─────────────────────────────────────────────────────────────
    print("Processing NDD / ASD...", flush=True)
    ndd_df = parse_preds(prediction_tsv("neurodev"))
    label_dict = pickle.load(open(NEURODEV_LABELS, "rb"))
    ndd_case_set, ndd_ctrl_set = set(), set()
    for (u, v, p) in ndd_df.apply(
            lambda r: (r.interactor, r.variant, r.partner), axis=1):
        key = f"{u} {v}"
        if key in label_dict:
            (ndd_case_set if label_dict[key] == 1 else ndd_ctrl_set).add((u, v, p))

    lines += [r"\multicolumn{6}{l}{\textit{Neurodevelopmental Disorders}} \\"]
    lines.append(row("Case",    stats(ndd_df, ndd_case_set)))
    lines.append(row("Control", stats(ndd_df, ndd_ctrl_set)))
    lines.append(r"\hline")

    # ASD is its own database (Fu et al. de novo cases), not a slice of NeuroDev.
    # This row used to be `stats(ndd_df, neurodev/variant_subset.pkl)` -- the
    # NeuroDev predictions filtered through an AlphaFold folding-budget cap
    # (<=10 partners per interactor, stopping at 600 complexes), whose 290
    # variants are a strict subset of NeuroDev's. That is what produced the
    # published 270-variant / 1,111-triplet row.
    asd_tsv = prediction_tsv("asd")
    lines += [r"\multicolumn{6}{l}{\textit{Autism Spectrum Disorder}} \\"]
    if asd_tsv.exists():
        lines.append(row("Case", stats(parse_preds(asd_tsv))))
    else:
        print(f"  ASD: {asd_tsv} not found, row omitted", flush=True)
    lines.append(r"\hline")

    lines.append(r"\end{tabular}")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote → {_OUT}")


if __name__ == "__main__":
    main()
