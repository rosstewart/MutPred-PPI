#!/usr/bin/env python3
"""Generate the LaTeX training-data table for the MutPred-PPI paper.

Counts come from the mapped source CSVs (`datasets/source_mapping/datasets/`),
one per row of the figure, and therefore describe each dataset as collected.
The separate **Modelled** column is the triplet count that survives the
AlphaFold-coverage filter -- the rows in `datasets/training_eval/` that every
method is actually trained and scored on.

Reading the canonical row table instead would report only the modelled figure
under the heading "Triplets", understating every dataset with nothing to show
that it had done so. The CSVs are deduplicated and carry `perturbed` as a real
label, so each count is a direct read rather than a difference between
overlapping label files.

Accessions are split with `utils.identifiers.split_variant_id` rather than by
guessing a separator, which matters for pairs whose accession itself contains a
hyphen or underscore (`NP_005190_KRTAP10-7`, and every isoform suffix).

Writes figures/training_data_table.tex as a drop-in tabular block.
"""
from __future__ import annotations

import pickle

import pandas as pd

# --- repo-relative path resolution (see src/paths.py) ---
from paths import DATA_ROOT, MAPPING_DIR, REPO_ROOT  # noqa: E402
from utils.identifiers import split_variant_id  # noqa: E402


_PUB = REPO_ROOT
_MS = DATA_ROOT / "megascale_preprocessed"
_OUT = _PUB / "figures" / "training_data_table.tex"

# Table row -> canonical dataset. Each is a real table, so no row of the figure
# is derived by subtracting one file from another any more.
AF3_FAILED_COL = "af3_failed"
_MAPPED_CSV_DIR = MAPPING_DIR / "datasets"


def _load_mapped_csv(dataset: str):
    """The mapped source CSV for a dataset, wherever the mapping wrote it.

    Combined sets sit at the top level; single-source ones under single_source/.
    """
    for candidate in (_MAPPED_CSV_DIR / f"{dataset}.csv",
                      _MAPPED_CSV_DIR / "single_source" / f"{dataset}.csv"):
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError(
        f"no mapped CSV for {dataset} under {_MAPPED_CSV_DIR}; "
        f"run notebooks/map_ppi_datasets_090826.py then "
        f"src/data_processing/annotate_af3_coverage.py")


_SAHNI = "sahni_only_mapped090826"
_FRAGOZA = "fragoza_only_mapped090826"
_VARCHAMP = "varchamp_all_mapped090826"
_SF = "sahni_fragoza_mapped090826"
_SFVC = "sahni_fragoza_varchamp_all_mapped090826"


def fmt(n) -> str:
    if n is None:
        return r"-"
    return f"{int(n):,}".replace(",", "{,}")


def stats(dataset: str) -> dict:
    """Dataset counts, reported BEFORE the AlphaFold-coverage filter.

    Read from the mapped source CSV rather than the canonical row table, because
    the row table has already had `af3_failed` rows dropped -- so counting it
    describes what the model was trained on, not what the dataset contains, and
    understates every column with no indication that it has done so.

    `modelled` is the triplet count that survives the filter: pairs AlphaFold 3
    produced a structure for, which is what `datasets/training_eval/` holds and
    what every method is scored on.

    `interactor` and `partner` are columns, so a pair is a tuple of two fields
    rather than a string that has to be taken apart again later. Accessions are
    used exactly as the table stores them -- collapsing an isoform onto its
    parent here would merge two rows the model treats as distinct.
    """
    df = _load_mapped_csv(dataset)
    failed = df[AF3_FAILED_COL].astype(bool) if AF3_FAILED_COL in df.columns else None
    if failed is None:
        raise ValueError(
            f"{dataset}: mapped CSV has no '{AF3_FAILED_COL}' column. Run\n"
            f"  python src/data_processing/annotate_af3_coverage.py")
    n_dis = int((df["perturbed"] == 1).sum())
    out = dict(
        proteins=len(set(df["interactor"]) | set(df["partner"])),
        pairs=len(set(zip(df["interactor"], df["partner"]))),
        variants=len(set(zip(df["interactor"], df["mutation"]))),
        triplets=len(df),
        modelled=int((~failed).sum()),
        dis=n_dis,
        non=len(df) - n_dis,
    )
    print(f"{dataset}: {out}", flush=True)
    return out


def megascale_stats() -> dict:
    """Tsuboyama stability pretraining set: proteins, rows, and ddG sign split.

    `vt_ids` here are `'{construct} {mutation}'`, the space-delimited variant id
    of `utils.identifiers`, so the protein side is read with `split_variant_id`
    rather than an ad-hoc `split(" ")[0]`.
    """
    ms = pickle.loads((_MS / "preprocessed.pkl").read_bytes())
    vt_ids, ddg = ms["vt_ids"], ms["ddg_labels"]
    out = dict(
        proteins=len({split_variant_id(vt)[0] for vt in vt_ids}),
        total=len(vt_ids),
        dis=int((ddg < 0).sum()),
        non=int((ddg >= 0).sum()),
    )
    print(f"megascale: {out}", flush=True)
    return out


def main() -> None:
    sahni = stats(_SAHNI)
    fragoza = stats(_FRAGOZA)
    varchamp = stats(_VARCHAMP)
    sf = stats(_SF)
    sfvc = stats(_SFVC)
    ms = megascale_stats()

    # The combined tables are smaller than the sum of their parts: a triple that
    # appears in two sources is one row, not two. Reported so the table is not
    # read as an arithmetic error.
    print(f"\nnote: {_SF} has {sf['triplets']} rows vs "
          f"{sahni['triplets'] + fragoza['triplets']} summed over its two "
          f"sources; the difference is deduplicated overlap.", flush=True)

    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        (r"\textbf{Dataset} & \textbf{Proteins} & \textbf{Pairs} & "
         r"\textbf{Variants} & \textbf{Triplets} & \textbf{Modelled} & "
         r"\textbf{Disruptive} & \textbf{Non-disruptive} \\"),
        r"\hline",
        r"\multicolumn{8}{l}{\textit{PPI Perturbation Data}} \\",
        (rf"Sahni \textit{{et al.}}~(Mendelian) & {fmt(sahni['proteins'])} & "
         rf"{fmt(sahni['pairs'])} & {fmt(sahni['variants'])} & "
         rf"{fmt(sahni['triplets'])} & {fmt(sahni['modelled'])} & {fmt(sahni['dis'])} & {fmt(sahni['non'])} \\"),
        (rf"Fragoza \textit{{et al.}}~(Population) & {fmt(fragoza['proteins'])} & "
         rf"{fmt(fragoza['pairs'])} & {fmt(fragoza['variants'])} & "
         rf"{fmt(fragoza['triplets'])} & {fmt(fragoza['modelled'])} & {fmt(fragoza['dis'])} & {fmt(fragoza['non'])} \\"),
        (rf"VarChAMP (IGVF) & {fmt(varchamp['proteins'])} & {fmt(varchamp['pairs'])} & "
         rf"{fmt(varchamp['variants'])} & {fmt(varchamp['triplets'])} & "
         rf"{fmt(varchamp['modelled'])} & {fmt(varchamp['dis'])} & "
         rf"{fmt(varchamp['non'])} \\"),
        r"\hline",
        r"\multicolumn{8}{l}{\textit{Combined Training Sets}} \\",
        (rf"Sahni, Fragoza & {fmt(sf['proteins'])} & {fmt(sf['pairs'])} & "
         rf"{fmt(sf['variants'])} & {fmt(sf['triplets'])} & "
         rf"{fmt(sf['modelled'])} & {fmt(sf['dis'])} & {fmt(sf['non'])} \\"),
        (rf"Sahni, Fragoza, VarChAMP & {fmt(sfvc['proteins'])} & "
         rf"{fmt(sfvc['pairs'])} & {fmt(sfvc['variants'])} & "
         rf"{fmt(sfvc['triplets'])} & {fmt(sfvc['modelled'])} & {fmt(sfvc['dis'])} & "
         rf"{fmt(sfvc['non'])} \\"),
        r"\hline",
        r"\multicolumn{8}{l}{\textit{Stability Pretraining Data}} \\",
        # Disruptive/Non-disruptive count PPI perturbation labels, which the
        # stability set does not carry -- its labels are a ddG sign split, a
        # different quantity, so the columns are left empty rather than filled
        # with a number that does not mean the same thing.
        (rf"Tsuboyama \textit{{et al.}} & {fmt(ms['proteins'])} & - & "
         rf"{fmt(ms['total'])} & - & - & - & - \\"),
        r"\hline",
        r"\end{tabular}",
    ]

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(lines) + "\n")
    print(f"\nWrote → {_OUT}")


if __name__ == "__main__":
    main()
